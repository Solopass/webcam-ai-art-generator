import cv2
import torch
import numpy as np
from PIL import Image
import time
import argparse
import os
import threading
import queue

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderTiny

import sys
import mediapipe as mp
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from audio_sync import AudioTracker

# --- THREAD QUEUES ---
Q_IN = queue.Queue(maxsize=1)
Q_OUT = queue.Queue(maxsize=1)

class ThreadedCamera:
    def __init__(self, src=0):
        if isinstance(src, int):
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                print("DirectShow failed. Falling back to default MSMF backend.")
                self.capture = cv2.VideoCapture(src)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.capture = cv2.VideoCapture(src)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        self.is_mock = False
        self.frame_counter = 0
        self.new_frame_event = threading.Event()
        if not self.capture.isOpened():
            print(f"Warning: Could not open camera {src}. Using Mock Camera for headless testing!")
            self.is_mock = True
            self.frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            self.status = True
            self.capture.release()
            self.capture = None
        else:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.status, self.frame = self.capture.read()

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        import time
        while self.running:
            if not self.is_mock and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    self.status, self.frame = True, frame
                    self.frame_counter += 1
                    self.new_frame_event.set()
            elif self.is_mock:
                self.frame = np.roll(self.frame, 5, axis=1)
                self.frame_counter += 1
                self.new_frame_event.set()
                time.sleep(0.033) 

    def read(self, wait=True):
        if wait:
            self.new_frame_event.wait()
            self.new_frame_event.clear()
        return self.status, self.frame, self.frame_counter

    def isOpened(self):
        return self.is_mock or (self.capture is not None and self.capture.isOpened())

    def release(self):
        self.running = False
        self.thread.join()
        if self.capture:
            self.capture.release()

def camera_thread(cap, args):
    import mediapipe.python.solutions as mp_solutions
    segmenter = mp_solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    face_detector = mp_solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    
    current_x = -1
    current_y = -1
    target_x = -1
    target_y = -1
    
    # Warmup
    for _ in range(4):
        cap.read(wait=True)
        
    while True:
        ret, frame, _ = cap.read(wait=True)
        if not ret: break
        
        if args.mirror_camera:
            frame = cv2.flip(frame, 1)
            
        h, w = frame.shape[:2]
        size = min(h, w)
        
        if target_x == -1:
            target_x = (w - size) // 2
            target_y = (h - size) // 2
            current_x = target_x
            current_y = target_y
            
        # Face detection every 5th frame
        if cap.frame_counter % 5 == 0:
            small_rgb = cv2.cvtColor(cv2.resize(frame, (w // 4, h // 4)), cv2.COLOR_BGR2RGB)
            results = face_detector.process(small_rgb)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                face_cx = int((bbox.xmin + bbox.width / 2) * w)
                face_cy = int((bbox.ymin + bbox.height / 2) * h)
                target_x = max(0, min(w - size, face_cx - (size // 2)))
                target_y = max(0, min(h - size, face_cy - int(size * 0.4)))
        
        current_x = int(current_x * 0.8 + target_x * 0.2)
        current_y = int(current_y * 0.8 + target_y * 0.2)
        
        cropped = frame[current_y:current_y+size, current_x:current_x+size]
        resized = cv2.resize(cropped, (512, 512))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        original_frame_rgb = frame_rgb.copy()
        
        # Selfie Mask (For Post-Processing Compositing ONLY)
        try:
            small_rgb = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            res = segmenter.process(small_rgb)
            if res.segmentation_mask is not None:
                soft_mask = cv2.resize(cv2.GaussianBlur(res.segmentation_mask, (7, 7), 0), (512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                soft_mask = np.ones((512, 512), dtype=np.float32)
        except Exception as e:
            soft_mask = np.ones((512, 512), dtype=np.float32)
        
        if Q_IN.full():
            try: Q_IN.get_nowait()
            except: pass
        Q_IN.put((frame_rgb, soft_mask, original_frame_rgb))

# --- POST-PROCESSING THREAD ---
def postprocess_thread(args, zmq_socket, vcam):
    anchor_frame = None
    bg_img_cache = None
    if args.bg_image and os.path.exists(args.bg_image):
        bg_img_cache = cv2.resize(cv2.imread(args.bg_image), (512, 512))
        
    start_time = time.time()
    frame_count = 0
    
    while True:
        out_frame, soft_mask, original_frame_rgb = Q_OUT.get()
        
        gaussian = cv2.GaussianBlur(out_frame, (0, 0), 1.5)
        out_frame = cv2.addWeighted(out_frame, 1.4, gaussian, -0.4, 0)
        
        # Vivid Anime Color Pop
        hsv = cv2.cvtColor(out_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 20)
        v = cv2.add(v, 10)
        out_frame = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        
        # Alpha Compositing
        alpha = soft_mask[..., np.newaxis]
        bg_to_use = bg_img_cache if bg_img_cache is not None else cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
        out_frame = (out_frame.astype(np.float32) * alpha + bg_to_use.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        # Output Routing
        if vcam:
            hd = cv2.resize(out_frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            vcam.send(cv2.cvtColor(hd, cv2.COLOR_BGR2RGB))

        if zmq_socket:
            _, buffer = cv2.imencode('.jpg', out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            zmq_socket.send(buffer.tobytes())
            
        frame_count += 1
        if time.time() - start_time > 5.0:
            fps = frame_count / (time.time() - start_time)
            print(f"[Engine] Rendering at {fps:.1f} FPS")
            start_time = time.time()
            frame_count = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="cat girl, masterpiece, ultra-detailed")
    parser.add_argument("--negative_prompt", type=str, default="lowres, bad anatomy, bad hands, text, error")
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--lora", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--freeze_threshold", type=float, default=0.95)
    parser.add_argument("--mirror_camera", action='store_true')
    parser.add_argument("--virtual_camera", action='store_true')
    parser.add_argument("--zmq_port", type=int, default=-1)
    parser.add_argument("--bg_image", type=str, default="")
    args = parser.parse_args()
    
    args.frame_buffer = 1
    
    zmq_socket = None
    if args.zmq_port > 0:
        import zmq
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUB)
        zmq_socket.bind(f"tcp://127.0.0.1:{args.zmq_port}")
        
    pipe = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to("cuda")

    engine_dir = f"engines_tinyvae_fb{args.frame_buffer}"
    if args.lora and args.lora.lower() != "none":
        lora_path = os.path.join("loras", args.lora)
        if os.path.exists(lora_path):
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora()
            safe_lora_name = "".join([c for c in args.lora if c.isalnum() or c in ('-', '_')]).replace('safetensors', '')
            engine_dir = f"engines_tinyvae_{safe_lora_name}_fb{args.frame_buffer}"

    stream = StreamDiffusion(
        pipe,
        t_index_list=[38],
        torch_dtype=torch.float16,
        cfg_type="none",
        do_add_noise=True,
        use_denoising_batch=True,
        frame_buffer_size=args.frame_buffer,
    )
    
    stream.load_lcm_lora()
    
    stream.prepare(
        prompt=args.prompt + ", closed mouth",
        negative_prompt=args.negative_prompt,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
    )
    
    # Pre-compute Audio Lip-Sync CLIP embeddings to prevent 200ms stutters when talking
    closed_mouth_embed = stream.prompt_embeds.clone()
    stream.update_prompt(args.prompt + ", open mouth, speaking, talking, conversational")
    open_mouth_embed = stream.prompt_embeds.clone()
    stream.prompt_embeds.copy_(closed_mouth_embed)

    stream = accelerate_with_tensorrt(
        stream,
        engine_dir,
        max_batch_size=args.frame_buffer,
        use_cuda_graph=True,
        engine_build_options={"opt_batch_size": args.frame_buffer}
    )

    # stream.enable_similar_image_filter(threshold=args.freeze_threshold, max_skip_frame=10)

    cam_id = int(args.camera) if args.camera.isdigit() else args.camera
    cap = ThreadedCamera(cam_id)
    
    vcam = None
    if args.virtual_camera:
        import pyvirtualcam
        try: vcam = pyvirtualcam.Camera(width=1024, height=1024, fps=30)
        except: pass
        
    audio_tracker = None
    try: audio_tracker = AudioTracker()
    except: pass
    
    # Launch Threads
    threading.Thread(target=camera_thread, args=(cap, args), daemon=True).start()
    threading.Thread(target=postprocess_thread, args=(args, zmq_socket, vcam), daemon=True).start()

    is_currently_speaking = False

    # AI INFERENCE THREAD (Main Thread)
    while True:
        frame_rgb, soft_mask, original_frame_rgb = Q_IN.get()
        
        if audio_tracker:
            speaking_now = audio_tracker.is_speaking
            if speaking_now != is_currently_speaking:
                is_currently_speaking = speaking_now
                if speaking_now:
                    stream.prompt_embeds.copy_(open_mouth_embed)
                else:
                    stream.prompt_embeds.copy_(closed_mouth_embed)

        output_image = stream(Image.fromarray(frame_rgb))
        
        if isinstance(output_image, list) and len(output_image) > 0:
            output_image = output_image[0]
            
        if output_image is not None:
            if not isinstance(output_image, np.ndarray):
                out_np = postprocess_image(output_image, output_type="np")[0]
            else:
                out_np = output_image
            
            out_np = (out_np * 255).astype(np.uint8) if out_np.dtype == np.float32 else out_np
            out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        if Q_OUT.full():
            try: Q_OUT.get_nowait()
            except: pass
        Q_OUT.put((out_frame, soft_mask, original_frame_rgb))

if __name__ == "__main__":
    main()
