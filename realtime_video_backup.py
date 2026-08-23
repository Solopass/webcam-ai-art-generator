import sys
import time
import cv2
import torch
import tensorrt
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

import threading

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print(f"Warning: CAP_DSHOW failed for camera {src}. Falling back to default backend...")
            self.capture = cv2.VideoCapture(src)
            
        self.is_mock = False
        self.frame_counter = 0
        self.new_frame_event = threading.Event()
        if not self.capture.isOpened():
            print(f"Warning: Could not open camera {src}. Using Mock Camera for headless testing!")
            self.is_mock = True
            self.status = True
            # Create a mock 1080p frame (noise/random image)
            self.frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        else:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
            self.status, self.frame = self.capture.read()
            
        self.running = True
        if self.status:
            self.thread = threading.Thread(target=self.update, args=(), daemon=True)
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
                else:
                    time.sleep(0.01)
            elif self.is_mock:
                # Add some motion to mock camera
                self.frame = np.roll(self.frame, 5, axis=1)
                self.frame_counter += 1
                self.new_frame_event.set()
                time.sleep(0.033) # 30 FPS mock

    def read(self, wait=True):
        if wait:
            self.new_frame_event.wait()
            self.new_frame_event.clear()
        return self.status, self.frame, self.frame_counter

    def isOpened(self):
        return self.is_mock or self.capture.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if not self.is_mock:
            self.capture.release()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="StreamDiffusion Realtime Video")
    parser.add_argument("--prompt", type=str, default="1girl, masterpiece, best quality, beautiful lighting, anime style, highly detailed", help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--guidance_scale", type=float, default=1.2, help="CFG Guidance scale")
    parser.add_argument("--delta", type=float, default=1.0, help="StreamDiffusion Delta (AI Creativity)")
    parser.add_argument("--frame_buffer", type=int, default=1, help="Frame Buffer Size")
    parser.add_argument("--freeze_threshold", type=float, default=0.95, help="Similar Image Filter Threshold")
    parser.add_argument("--mirror_camera", action="store_true", help="Mirror the webcam input horizontally")
    parser.add_argument("--virtual_camera", action="store_true", help="Output to OBS Virtual Camera instead of a window")
    parser.add_argument("--zmq_port", type=int, default=0, help="Stream frames over ZMQ to the specified port")
    parser.add_argument("--audio_sync", action="store_true", help="Enable audio-driven lip sync")
    parser.add_argument("--emotion_sync", action="store_true", help="Enable emotion-driven dynamic prompting")
    parser.add_argument("--lora", type=str, default="None", help="Character LoRA filename (must be in loras/ folder)")
    parser.add_argument("--bg_image", type=str, default="", help="Path to custom background image for chroma keying")
    args = parser.parse_args()
    
    # CRITICAL: For 1-Step LCM-LoRA (t_index_list=[26]), frame_buffer MUST mathematically be 1.
    # Any other value causes TensorRT to expect multiple frames and crash with a ShapeMachine error.
    args.frame_buffer = 1
    
    zmq_socket = None
    if args.zmq_port > 0:
        import zmq
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUB)
        zmq_socket.bind(f"tcp://127.0.0.1:{args.zmq_port}")
        print(f"ZMQ Video Stream bound to port {args.zmq_port}")

    print("Initializing StreamDiffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    print("Loading TinyVAE for VTuber Fluidity...")
    from diffusers import AutoencoderTiny
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to("cuda")

    engine_dir = f"engines_tinyvae_fb{args.frame_buffer}"
    if args.lora and args.lora.lower() != "none":
        lora_path = os.path.join("loras", args.lora)
        if os.path.exists(lora_path):
            print(f"Loading Character LoRA: {args.lora}")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora()
            safe_lora_name = "".join([c for c in args.lora if c.isalnum() or c in ('-', '_')]).replace('safetensors', '')
            engine_dir = f"engines_tinyvae_{safe_lora_name}_fb{args.frame_buffer}"
            print(f"Targeting LoRA-specific TensorRT Engine: {engine_dir}")

    stream = StreamDiffusion(
        pipe,
        t_index_list=[26],
        torch_dtype=torch.float16,
        cfg_type="none",
        do_add_noise=True,
        use_denoising_batch=True,
        frame_buffer_size=args.frame_buffer,
    )

    print("Loading LCM-LoRA...")
    stream.load_lcm_lora()
    stream.fuse_lora()

    print("Applying TensorRT acceleration (this may take 5-15 mins on first run)...")
    from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
    
    # Critical Fix: TRT batch size MUST match args.frame_buffer when cfg_type is "none".
    stream = accelerate_with_tensorrt(
        stream,
        engine_dir,
        max_batch_size=args.frame_buffer,
        use_cuda_graph=False,
        engine_build_options={"opt_batch_size": args.frame_buffer}
    )

    print("Enabling VTuber Temporal Stability (Similar Image Filter)...")
    stream.enable_similar_image_filter(threshold=args.freeze_threshold, max_skip_frame=10)

    print(f"Preparing StreamDiffusion for real-time inference with prompt: {args.prompt}")
    stream.prepare(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
    )

    print(f"Opening webcam {args.camera} in background thread...")
    cap = ThreadedCamera(args.camera)
    if not cap.isOpened():
        print(f"\n[CRITICAL ERROR] Could not open webcam at index {args.camera}!")
        print("This usually means another app (like Discord, Zoom, or OBS) is using the camera.")
        print("Or you typed the wrong Camera Index. Let's scan for available cameras...")
        
        # Scan for cameras
        available_cams = []
        for i in range(5):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                available_cams.append(i)
                test_cap.release()
                
        if available_cams:
            print(f"SUCCESS! Found available cameras at indices: {available_cams}")
            print(f"Please change the 'Camera Index' in the GUI to one of the above numbers!")
        else:
            print("FAILED! No cameras detected on this system at all.")
            
        if not cap.is_mock:
            sys.exit(1)

    audio_tracker = None
    if args.audio_sync:
        print("Initializing Audio Lip-Sync (FFT)...")
        from audio_sync import AudioTracker
        audio_tracker = AudioTracker()

    vcam = None

    print("Starting real-time loop. Press 'q' to quit.")
    
    # Warmup
    for _ in range(4):
        ret, frame, _ = cap.read()
        if not ret: continue
        if args.mirror_camera:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        cropped = frame[y:y+size, x:x+size]
        resized = cv2.resize(cropped, (512, 512))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        stream(img)

    fps = 0
    start_time = time.time()
    frame_count = 0
    is_currently_speaking = False
    last_processed_counter = -1
    
    while True:
        # Blocks instantly and cleanly until the background thread sets the event
        ret, frame, current_counter = cap.read(wait=True)
        if not ret:
            print("Failed to grab frame.")
            break

        if args.mirror_camera:
            frame = cv2.flip(frame, 1)

        # --- Dynamic Prompting (Audio Lip-Sync) ---
        if audio_tracker:
            speaking_now = audio_tracker.is_speaking
            if speaking_now != is_currently_speaking:
                is_currently_speaking = speaking_now
                dynamic_prompt = args.prompt
                if speaking_now:
                    dynamic_prompt += ", open mouth, speaking, talking, conversational"
                else:
                    dynamic_prompt += ", closed mouth"
                
                # Re-prepare the prompt embeds dynamically (takes ~10ms)
                stream.prepare(
                    prompt=dynamic_prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    delta=args.delta,
                )

        # --- Smart Face-Tracking Camera Crop ---
        h, w = frame.shape[:2]
        size = min(h, w)
        
        # Initialize default center crop
        target_y = (h - size) // 2
        target_x = (w - size) // 2
        
        try:
            if not hasattr(cap, 'face_detector'):
                import mediapipe as mp
                cap.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
                cap.current_x = target_x
                cap.current_y = target_y
                cap.target_x = target_x
                cap.target_y = target_y
                cap.frame_counter = 0
            
            cap.frame_counter += 1
            
            # Only run heavy face detection every 5th frame to save CPU and restore 30+ FPS
            if cap.frame_counter % 5 == 0:
                small_frame = cv2.resize(frame, (w // 4, h // 4))
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = cap.face_detector.process(small_rgb)
                
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    face_cx = int((bbox.xmin + bbox.width / 2) * w)
                    face_cy = int((bbox.ymin + bbox.height / 2) * h)
                    
                    cap.target_x = face_cx - (size // 2)
                    cap.target_y = face_cy - int(size * 0.4)
                    
                    cap.target_x = max(0, min(w - size, cap.target_x))
                    cap.target_y = max(0, min(h - size, cap.target_y))
            
            # Smooth Camera Panning (Exponential Moving Average) updates EVERY frame
            cap.current_x = int(cap.current_x * 0.8 + cap.target_x * 0.2)
            cap.current_y = int(cap.current_y * 0.8 + cap.target_y * 0.2)
            
            y = cap.current_y
            x = cap.current_x
        except Exception as e:
            y, x = target_y, target_x # Fallback to center
            
        cropped = frame[y:y+size, x:x+size]
        resized = cv2.resize(cropped, (512, 512))

        # Convert to RGB
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # --- Pre-Processing: Lighting Normalization (CLAHE) ---
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # --- Pre-Processing: Facial Feature Lock (Input Sharpening) ---
        gaussian_in = cv2.GaussianBlur(frame_rgb, (0, 0), 1.5)
        frame_rgb = cv2.addWeighted(frame_rgb, 1.5, gaussian_in, -0.5, 0)
        
        # --- Pre-Processing: Background Annihilation (Virtual Green Screen) ---
        try:
            if not hasattr(cap, 'segmenter'):
                import mediapipe as mp
                cap.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) # Fast Model
            
            # Downscale for ultra-fast CPU inference (Saves ~25ms)
            small_rgb = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            results = cap.segmenter.process(small_rgb)
            
            # Blur the small mask (much faster than blurring 512x512)
            small_mask = cv2.GaussianBlur(results.segmentation_mask, (7, 7), 0)
            
            # Upscale the soft mask back to 512x512
            soft_mask = cv2.resize(small_mask, (512, 512), interpolation=cv2.INTER_LINEAR)
            cap.last_mask = soft_mask # Save float mask for soft chroma key injection
            
            # Save original frame for compositing later
            cap.original_frame_rgb = frame_rgb.copy()
            
            # Smoothly multiply webcam frame by mask to fade background to pure black without jagged pixels
            frame_rgb = (frame_rgb * soft_mask[..., None]).astype(np.uint8)
        except Exception as e:
            pass # Fallback if MediaPipe fails

        # --- Pre-Processing: Temporal Webcam Denoising ---
        if not hasattr(cap, 'prev_frame'):
            cap.prev_frame = frame_rgb.astype(np.float32)
        else:
            cv2.accumulateWeighted(frame_rgb.astype(np.float32), cap.prev_frame, alpha=0.3)
            frame_rgb = cv2.convertScaleAbs(cap.prev_frame)

        # Run Inference
        input_image = Image.fromarray(frame_rgb)
        output_image = stream(input_image)
        
        if isinstance(output_image, list) and len(output_image) > 0:
            output_image = output_image[0]
        
        if output_image is not None:
            if not isinstance(output_image, np.ndarray):
                out_np = postprocess_image(output_image, output_type="np")[0]
            else:
                out_np = output_image
            
            out_np = (out_np * 255).astype(np.uint8) if out_np.dtype == np.float32 else out_np
            out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

            # --- Professional VTuber Post-Processing ---
            # 1. Subtle Unsharp Mask: Crisps up the anime lines after the TinyVAE softness
            gaussian = cv2.GaussianBlur(out_frame, (0, 0), 1.5)
            out_frame = cv2.addWeighted(out_frame, 1.4, gaussian, -0.4, 0)
            
            # 2. Reinhard Color Transfer: Prevent AI Color Flickering
            if not hasattr(cap, 'anchor_frame'):
                cap.anchor_frame = out_frame.copy().astype(np.float32)
            else:
                out_frame = color_transfer(cap.anchor_frame.astype(np.uint8), out_frame)
                # Gracefully adapt to long-term room lighting changes (1% per frame) while stopping fast flicker
                cv2.accumulateWeighted(out_frame.astype(np.float32), cap.anchor_frame, alpha=0.01)

            # 3. Vivid Anime Color Pop
            hsv = cv2.cvtColor(out_frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, 20)
            v = cv2.add(v, 10)
            hsv = cv2.merge((h, s, v))
            out_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # --- Post-Processing: Selective AI Masking (Real Background) ---
            if hasattr(cap, 'last_mask') and hasattr(cap, 'original_frame_rgb'):
                alpha = cv2.resize(cap.last_mask, (out_frame.shape[1], out_frame.shape[0]))
                alpha = alpha[..., np.newaxis] # Shape (H, W, 1)
                
                # Check for custom background image, otherwise use unmodified webcam background
                bg_to_use = None
                if hasattr(args, 'bg_image') and args.bg_image and os.path.exists(args.bg_image):
                    if not hasattr(cap, 'bg_img_cache'):
                        bg = cv2.imread(args.bg_image)
                        if bg is not None:
                            cap.bg_img_cache = cv2.resize(bg, (out_frame.shape[1], out_frame.shape[0]))
                        else:
                            cap.bg_img_cache = None
                    bg_to_use = cap.bg_img_cache
                
                if bg_to_use is None:
                    # Use original real-world room background!
                    bg_to_use = cv2.cvtColor(cap.original_frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Cinematic Alpha Blending: Anime Avatar composited over real room!
                out_frame = (out_frame * alpha + bg_to_use * (1.0 - alpha)).astype(np.uint8)

            # Calculate and log FPS (every 5 seconds to avoid spam)
            frame_count += 1
            if time.time() - start_time > 5.0:
                fps = frame_count / (time.time() - start_time)
                print(f"[Engine] Rendering at {fps:.1f} FPS")
                start_time = time.time()
                frame_count = 0
            
            # --- Output Routing (Virtual Camera vs Window vs ZMQ) ---
            if args.virtual_camera:
                if vcam is None:
                    import pyvirtualcam
                    try:
                        # Initialize Virtual Camera at 1024x1024 High Definition
                        vcam = pyvirtualcam.Camera(width=1024, height=1024, fps=30)
                        print(f"OBS Virtual Camera started: {vcam.device} (HD 1024x1024)")
                    except Exception as e:
                        print(f"\n[ERROR] Virtual Camera failed: {e}")
                        print("Please ensure you have OBS Studio installed and have started the 'Virtual Camera' at least once!")
                        args.virtual_camera = False # Fallback to window mode
                        continue
                
                # Upscale using hardware-accelerated LINEAR for max FPS
                hd_frame = cv2.resize(out_frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                # PyVirtualCam requires RGB format
                rgb_out = cv2.cvtColor(hd_frame, cv2.COLOR_BGR2RGB)
                vcam.send(rgb_out)
                vcam.sleep_until_next_frame()
            elif zmq_socket is not None:
                # Compress frame to JPEG and send over ZMQ
                _, buffer = cv2.imencode('.jpg', out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                zmq_socket.send(buffer.tobytes())
            else:
                cv2.imshow("StreamDiffusion AI WebCam", out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if audio_tracker:
        audio_tracker.close()
    if vcam:
        vcam.close()


def color_transfer(source, target):
    # Reinhard Color Transfer: Forces the target image to match the color palette of the source image
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)
    
    # Avoid division by zero
    tgt_std = np.where(tgt_std == 0, 1e-5, tgt_std)

    l = (tgt_lab[:, :, 0] - tgt_mean[0][0]) * (src_std[0][0] / tgt_std[0][0]) + src_mean[0][0]
    a = (tgt_lab[:, :, 1] - tgt_mean[1][0]) * (src_std[1][0] / tgt_std[1][0]) + src_mean[1][0]
    b = (tgt_lab[:, :, 2] - tgt_mean[2][0]) * (src_std[2][0] / tgt_std[2][0]) + src_mean[2][0]

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b]).astype("uint8")
    return cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)

def import_numpy_array(pil_image):
    return np.array(pil_image)

if __name__ == "__main__":
    main()
