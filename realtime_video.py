"""
TensorRT AI VTuber Engine — headless inference backend.

Pipeline (3 threads):
  camera_thread      : capture -> face-track crop -> selfie mask -> Q_IN
  main thread        : TinyVAE encode -> TensorRT UNet -> TinyVAE decode -> Q_OUT
  postprocess_thread : color stabilise -> composite -> ZMQ / virtual camera
"""

import argparse
import os
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from streamdiffusion.image_utils import postprocess_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# --- THREAD QUEUES ---
Q_IN = queue.Queue(maxsize=1)
Q_OUT = queue.Queue(maxsize=1)
STOP = threading.Event()
FAILED = threading.Event()


LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
_log_file = None


def open_log():
    """Every run gets its own file plus logs/engine-latest.log, so a crash can
    be read back after the window is gone."""
    global _log_file
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        paths = [os.path.join(LOG_DIR, f"engine-{stamp}.log"),
                 os.path.join(LOG_DIR, "engine-latest.log")]
        _log_file = [open(p, "w", encoding="utf-8", buffering=1) for p in paths]
        log(f"[Engine] Logging to logs/engine-{stamp}.log")
    except Exception as e:
        print(f"[Engine] Could not open log file: {e}", flush=True)
        _log_file = None


def log(msg):
    """Unbuffered print so the GUI log box updates live; mirrored to disk."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode('utf-8', 'replace').decode('cp1252', 'replace'), flush=True)
    if _log_file:
        line = f"{time.strftime('%H:%M:%S')} {msg}\n"
        for fh in _log_file:
            try:
                fh.write(line)
            except Exception:
                pass


def log_exception(where, exc):
    log(f"[FATAL] {where} raised {type(exc).__name__}: {exc}")
    log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())


def log_environment(args):
    log("--- environment ---")
    log(f"argv: {' '.join(sys.argv[1:])}")
    log(f"python: {sys.version.split()[0]}  platform: {sys.platform}")
    # numpy and onnxruntime are here for a reason: a numpy 2.x / onnxruntime
    # mismatch breaks the ONNX export, and you only find out ~15 minutes into an
    # engine rebuild. This line tells you before you start.
    for mod in ("torch", "numpy", "diffusers", "tensorrt", "onnxruntime",
                "cv2", "mediapipe", "zmq"):
        try:
            m = __import__(mod)
            log(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception as e:
            log(f"{mod}: NOT IMPORTABLE ({e})")
    try:
        import torch as _t
        if _t.cuda.is_available():
            free, total = _t.cuda.mem_get_info()
            log(f"gpu: {_t.cuda.get_device_name(0)}  vram free {free/2**30:.1f}/{total/2**30:.1f} GiB")
        else:
            log("gpu: CUDA NOT AVAILABLE")
    except Exception as e:
        log(f"gpu: probe failed ({e})")
    log(f"resolved: t_index={args.t_index} cfg_type={args.cfg_type} "
        f"guidance={args.guidance_scale} delta={args.delta} "
        f"freeze={args.freeze_threshold} composite={args.composite} "
        f"cuda_graph={args.cuda_graph} audio_sync={args.audio_sync} seed={args.seed}")
    log("-------------------")


def _thread_excepthook(a):
    log(f"[FATAL] Thread '{a.thread.name if a.thread else '?'}' died.")
    log("".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)).rstrip())
    # Without FAILED, a dead worker just tripped STOP, the main loop exited
    # normally and the process returned 0 — so the GUI reported a hard crash as
    # "=== Engine Shut Down ===".
    FAILED.set()
    STOP.set()


threading.excepthook = _thread_excepthook


def _signal_handler(signum, _frame):
    log(f"[Engine] Received signal {signum}; shutting down.")
    STOP.set()


def install_signal_handlers():
    """Clean shutdown for console runs (Ctrl-C, SIGTERM from a shell,
    smoke_test.py).

    The GUI does NOT rely on this: it is spawned with CREATE_NO_WINDOW, and
    console control events cannot be delivered to a process with no console.
    That path uses the ZMQ command channel — see cmd_listener_thread."""
    import signal
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------- camera ----
class ThreadedCamera:
    def __init__(self, src=0, mock=False):
        self.src = src
        self.capture = None
        self.is_mock = False
        self.is_screen = False
        self.sct = None
        self.monitor = None
        self.frame_counter = 0
        self.new_frame_event = threading.Event()
        self.status = False
        self.frame = None

        if mock:
            self._become_mock()
        elif str(src).lower().startswith("screen"):
            screen_idx = 1
            if " " in str(src):
                try:
                    screen_idx = int(str(src).split(" ")[1])
                except ValueError:
                    pass
            self._become_screen(screen_idx)
        else:
            if isinstance(src, int) and sys.platform == "win32":
                self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not self.capture.isOpened():
                    log("[Camera] DirectShow failed, falling back to the default backend.")
                    self.capture.release()
                    self.capture = cv2.VideoCapture(src)
            else:
                self.capture = cv2.VideoCapture(src)

            if not self.capture.isOpened():
                self.capture.release()
                self.capture = None
            else:
                # BUFFERSIZE must be set on every backend or latency piles up.
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
                self.status, self.frame = self.capture.read()
                if not self.status:
                    log("[Camera] Opened the device but could not read a frame from it.")
                    self.capture.release()
                    self.capture = None

        self.running = self.is_mock or self.is_screen or self.capture is not None
        self.thread = None
        if self.running:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()

    def _become_mock(self):
        log("[Camera] Running with the MOCK camera (synthetic test pattern).")
        self.is_mock = True
        self.status = True
        # A structured gradient, not noise: random noise makes the diffusion
        # model hallucinate garbage and looks like the engine is broken.
        gx = np.linspace(0, 255, 640, dtype=np.uint8)
        base = np.repeat(gx[None, :], 480, axis=0)
        self.frame = cv2.merge((base, base[::-1], np.full_like(base, 128)))

    def _become_screen(self, screen_idx=1):
        log(f"[Camera] Capturing Screen {screen_idx} (Desktop) instead of a webcam.")
        import mss
        self.is_screen = True
        self.sct = mss.mss()
        if screen_idx < len(self.sct.monitors):
            self.monitor = self.sct.monitors[screen_idx]
        else:
            log(f"[Camera] Screen {screen_idx} not found. Falling back to Screen 1.")
            self.monitor = self.sct.monitors[1]
        self.status = True
        self.frame = np.array(self.sct.grab(self.monitor))[:, :, :3]

    def update(self):
        while self.running and not STOP.is_set():
            if self.is_mock:
                self.frame = np.roll(self.frame, 5, axis=1)
                self.frame_counter += 1
                self.new_frame_event.set()
                time.sleep(0.033)
            elif self.is_screen:
                sct_img = self.sct.grab(self.monitor)
                self.frame = np.array(sct_img)[:, :, :3]
                self.status = True
                self.frame_counter += 1
                self.new_frame_event.set()
                time.sleep(0.033)
            else:
                ret, frame = self.capture.read()
                if ret:
                    self.status, self.frame = True, frame
                    self.frame_counter += 1
                    self.new_frame_event.set()
                else:
                    # If this is a video file, loop it seamlessly for preview!
                    if isinstance(self.src, str) and os.path.isfile(self.src):
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        # Do not spin the CPU at 100% on a disconnected device.
                        time.sleep(0.01)

    def read(self, wait=True, timeout=2.0):
        if wait:
            if not self.new_frame_event.wait(timeout=timeout):
                return False, self.frame, self.frame_counter
            self.new_frame_event.clear()
        return self.status, self.frame, self.frame_counter

    def isOpened(self):
        return self.is_mock or self.is_screen or (self.capture is not None and self.capture.isOpened())

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                # cv2.VideoCapture is not thread-safe; releasing it while the
                # capture thread is still blocked inside read() (a stalled or
                # unplugged UVC device can block for seconds) is a native
                # use-after-free. Leaking the handle is the lesser evil.
                log("[Camera] Capture thread is stuck in read(); leaving the "
                    "device open rather than releasing it underneath.")
                return
        if self.capture is not None:
            self.capture.release()
        if getattr(self, "sct", None) is not None:
            self.sct.close()


def scan_cameras(limit=6):
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found



# ---------------------------------------------------------- camera thread ----
def camera_thread(cap, args, state_dict, controlnet_aux_models):
    import mediapipe.python.solutions as mp_solutions

    # Always use selfie segmentation to isolate the user from the background
    # before AI generation, avoiding background bleed-in (AI Green Screen).
    # However, if capturing a screen, we want to stylize the whole screen, not just people!
    segmenter = None if getattr(cap, "is_screen", False) else mp_solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    face_detector = None
    face_mesh = None
    if not args.no_face_track:
        face_detector = mp_solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        face_mesh = mp_solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    current_x = current_y = target_x = target_y = -1
    prev_input = None
    mouth_open_state = False
    smiling_state = False
    eyes_closed_state = False
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) if args.normalize_lighting else None

    for _ in range(4):  # warm up the capture device
        cap.read(wait=True)

    # The main loop reports "wait Nms" but could never say WHY the producer was
    # late. cap.read() does not quantize to camera frames (the new-frame event
    # stays set while we work, so the next read returns at once), which means
    # the producer's cycle is pure CPU — and at 10.0 FPS with 76ms of inference
    # and 21ms of wait, that CPU is the actual bottleneck. These timers say
    # which stage owns it instead of leaving it to guesswork.
    cam_t0 = time.time()
    cam_n = 0
    cam_stage = {"read": 0.0, "face": 0.0, "prep": 0.0, "seg": 0.0,
                 "mesh": 0.0, "cond": 0.0}
    seg_cache = None
    prev_emotions = []

    while not STOP.is_set():
        _t = time.perf_counter()
        ret, frame, counter = cap.read(wait=True)
        cam_stage["read"] += time.perf_counter() - _t
        if not ret or frame is None:
            continue

        if args.mirror_camera:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        zoom = state_dict.get("zoom", getattr(args, "zoom", 1.0))
        size = int(min(h, w) / zoom)

        if target_x == -1:
            target_x = (w - size) // 2
            target_y = (h - size) // 2
            current_x, current_y = target_x, target_y

        _t = time.perf_counter()
        if face_detector is not None and counter % 5 == 0:
            small_rgb = cv2.cvtColor(cv2.resize(frame, (w // 4, h // 4)), cv2.COLOR_BGR2RGB)
            try:
                results = face_detector.process(small_rgb)
            except Exception:
                results = None
            if results is not None and results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                face_cx = int((bbox.xmin + bbox.width / 2) * w)
                face_cy = int((bbox.ymin + bbox.height / 2) * h)
                target_x = max(0, min(w - size, face_cx - (size // 2)))
                target_y = max(0, min(h - size, face_cy - int(size * 0.4)))

        current_x = int(current_x * 0.8 + target_x * 0.2)
        current_y = int(current_y * 0.8 + target_y * 0.2)

        if args.no_face_track:
            # Zoom used to be dropped entirely here while its slider still said
            # it zoomed. Crop a centred rect of the SAME aspect ratio so the
            # framing stays full-frame. zoom == 1.0 is a byte-identical no-op.
            if zoom > 1.001:
                cw, ch = max(16, int(w / zoom)), max(16, int(h / zoom))
                ox, oy = (w - cw) // 2, (h - ch) // 2
                cropped = frame[oy:oy + ch, ox:ox + cw]
            else:
                cw, ch = w, h
                ox = oy = 0
                cropped = frame
        else:
            cw, ch = size, size
            cropped = frame[current_y:current_y + size, current_x:current_x + size]
        
        cam_stage["face"] += time.perf_counter() - _t
        _t = time.perf_counter()
        resized = cv2.resize(cropped, (512, 512))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Untouched copy kept for background compositing.
        original_frame_rgb = frame_rgb.copy()

        if clahe is not None:
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            frame_rgb = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

        # Temporal denoise: kills webcam grain so the model stops re-inventing
        # detail over static noise (this is the "Temporal Hardware Denoising"
        # the README advertises — it had been dropped in the threaded rewrite).
        temporal_denoise = float(state_dict.get("temporal_denoise", args.temporal_denoise))
        if temporal_denoise > 0.0:
            f32 = frame_rgb.astype(np.float32)
            if prev_input is None:
                prev_input = f32
            else:
                cv2.accumulateWeighted(f32, prev_input, temporal_denoise)
                frame_rgb = cv2.convertScaleAbs(prev_input)

        cam_stage["prep"] += time.perf_counter() - _t
        _t = time.perf_counter()
        soft_mask = None
        # Live-toggleable: the segmenter object is built once, but whether we
        # USE it is re-read every frame so "Paint Whole Frame" works mid-run.
        # Leaving soft_mask as None disables both the grey-screen below and the
        # silhouette mask in postprocess_thread's HD paste-back.
        use_segmenter = segmenter is not None and not state_dict.get("no_segment", args.no_segment)
        # --segment_every N reuses the previous mask on the frames in between.
        # At 30fps a silhouette barely moves in 33ms, so N=2 halves the cost of
        # the most expensive stage for a difference you have to look for.
        # Default 1 = run every frame, exactly as before.
        seg_every = max(1, int(getattr(args, "segment_every", 1)))
        if use_segmenter and seg_every > 1 and counter % seg_every != 0 and seg_cache is not None:
            soft_mask = seg_cache
            alpha = soft_mask[..., np.newaxis].astype(np.float32)
            frame_rgb = (frame_rgb.astype(np.float32) * alpha
                         + np.float32(64.0) * (1.0 - alpha)).astype(np.uint8)
            use_segmenter = False
        if use_segmenter:
            try:
                small_rgb = cv2.resize(original_frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
                res = segmenter.process(small_rgb)
                if res.segmentation_mask is not None:
                    # Edge softness of the cutout. Was a fixed 7px blur.
                    k = int(state_dict.get("mask_feather", args.mask_feather))
                    k = max(1, min(31, k | 1))          # odd, in range
                    small_mask = cv2.GaussianBlur(res.segmentation_mask, (k, k), 0)
                    soft_mask = cv2.resize(small_mask, (512, 512), interpolation=cv2.INTER_LINEAR)
                    
                    # AI Green Screen: Isolate the character on a neutral dark gray background 
                    # before sending it to the AI. This stops the AI from hallucinating 
                    # the real room into the generated image. Dark gray blends better in latent space.
                    # float32 throughout: the int64 literal used to promote the
                    # whole expression to float64, which is measurably slower on
                    # a 512x512x3 frame for identical output.
                    alpha = soft_mask[..., np.newaxis].astype(np.float32)
                    grey = np.float32(64.0)
                    frame_rgb = (frame_rgb.astype(np.float32) * alpha
                                 + grey * (1.0 - alpha)).astype(np.uint8)
                    seg_cache = soft_mask
            except Exception as e:
                log(f"[Camera] Segmentation failed ({e}); compositing the full AI frame.")
                soft_mask = None

        cam_stage["seg"] += time.perf_counter() - _t
        _t = time.perf_counter()
        emotions = []
        mesh_every = max(1, int(getattr(args, "mesh_every", 1)))
        run_mesh = face_mesh is not None and counter % mesh_every == 0
        if face_mesh is not None and not run_mesh:
            # Reuse the last expression state rather than dropping to neutral,
            # which would make the mouth flicker shut every other frame.
            emotions = list(prev_emotions)
        if run_mesh:
            try:
                fm_res = face_mesh.process(frame_rgb)
                if fm_res.multi_face_landmarks:
                    landmarks = fm_res.multi_face_landmarks[0].landmark
                    sens_overrides = state_dict.get("sens_overrides", {})
                    mouth_sens = sens_overrides.get("mouth", 0.030)
                    smile_sens = sens_overrides.get("smile", 0.010)
                    eyes_sens = sens_overrides.get("eyes", 0.019)
                    
                    # Calculate vertical mouth opening (landmarks 13 and 14)
                    mouth_top = np.array([landmarks[13].x, landmarks[13].y])
                    mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])
                    mouth_open = np.linalg.norm(mouth_top - mouth_bottom)
                    
                    if mouth_open > (mouth_sens * 0.6 if mouth_open_state else mouth_sens):
                        mouth_open_state = True
                    elif mouth_open < mouth_sens * 0.6:
                        mouth_open_state = False
                    emotions.append("open mouth" if mouth_open_state else "closed mouth")

                    # Calculate smile (mouth corners moving up relative to center)
                    left_corner = landmarks[61].y
                    right_corner = landmarks[291].y
                    center_lip = landmarks[13].y
                    smile = min(center_lip - left_corner, center_lip - right_corner)
                    if smile > (smile_sens * 0.55 if smiling_state else smile_sens):
                        smiling_state = True
                    elif smile < smile_sens * 0.55:
                        smiling_state = False
                    if smiling_state:
                        emotions.append("smiling")

                    # Calculate blink (left eye: 159, 145 / right eye: 386, 374)
                    left_eye_open = landmarks[145].y - landmarks[159].y
                    right_eye_open = landmarks[374].y - landmarks[386].y
                    eye_open = max(left_eye_open, right_eye_open)
                    if eye_open < (eyes_sens if eyes_closed_state else eyes_sens * 0.7):
                        eyes_closed_state = True
                    elif eye_open > eyes_sens:
                        eyes_closed_state = False
                    if eyes_closed_state:
                        emotions.append("closed eyes")
            except Exception:
                pass

        # Calculate ControlNet Depth Map (Async to GPU render)
        cond_final = None
        if len(controlnet_aux_models) > 0:
            import torchvision.transforms.functional as TF
            import torch
            from PIL import Image
            cond_tensors = []
            for net_type in ["depth", "canny", "lineart", "openpose"]:
                if net_type in controlnet_aux_models:
                    aux = controlnet_aux_models[net_type]
                    # ControlNet detectors typically take PIL Image or Numpy
                    with torch.no_grad():
                        cond_pil = aux(Image.fromarray(frame_rgb))
                    c_tensor = TF.to_tensor(cond_pil).unsqueeze(0).to(dtype=torch.float16, device="cuda")
                    # True Green Screen: Multiply depth map by soft_mask so background stays empty
                    if soft_mask is not None:
                        # soft_mask is [H, W] float32 in [0, 1]
                        mask_tensor = torch.from_numpy(soft_mask).unsqueeze(0).unsqueeze(0).to(device=c_tensor.device, dtype=c_tensor.dtype)
                        c_tensor = c_tensor * mask_tensor
                    cond_tensors.append(c_tensor)
            if len(cond_tensors) > 0:
                cond_final = torch.cat(cond_tensors, dim=1)

        # Always keep the freshest frame waiting in the slot. Skipping the work
        # whenever the slot was full (an earlier "optimisation" of mine) served
        cam_stage["cond"] += time.perf_counter() - _t
        prev_emotions = list(emotions)
        cam_n += 1
        _el = time.time() - cam_t0
        if _el > 5.0 and cam_n:
            per = {k: v / cam_n * 1000.0 for k, v in cam_stage.items()}
            total = sum(per.values())
            log("[Camera] {:.1f} fps produced | {:.1f}ms/frame = ".format(cam_n / _el, total)
                + "  ".join(f"{k} {per[k]:.1f}" for k in
                            ("read", "face", "prep", "seg", "mesh", "cond")))
            cam_t0 = time.time()
            cam_n = 0
            cam_stage = {k: 0.0 for k in cam_stage}

        # the camera wait and the CPU preprocessing *in series* with the GPU
        # instead of overlapping them, which cost more throughput than the
        # MediaPipe calls it saved. The producer must stay ahead of the GPU.
        if Q_IN.full():
            try:
                Q_IN.get_nowait()
            except queue.Empty:
                pass
        try:
            # The paste-back origin must be the origin we actually cropped at.
            # Hardcoding 0,0 for Full Frame Mode was correct only while zoom was
            # ignored there; now that zoom crops a centred rect, a 0,0 paste puts
            # the stylised image in the top-left corner, half a frame off.
            crop_origin = (ox, oy) if args.no_face_track else (current_x, current_y)
            Q_IN.put_nowait((frame_rgb, soft_mask, original_frame_rgb, emotions,
                             cond_final, frame, (crop_origin[0], crop_origin[1], cw, ch)))
        except queue.Full:
            pass


# ----------------------------------------------------- postprocess thread ----
def postprocess_thread(args, zmq_socket, vcam, state_dict):
    bg_img_cache = None
    if args.bg_image and os.path.exists(args.bg_image):
        bg = cv2.imread(args.bg_image)
        if bg is not None:
            bg_img_cache = cv2.resize(bg, (512, 512))
        else:
            log(f"[Engine] Could not read background image: {args.bg_image}")

    current_smoothed_frame = None
    target_frame = None
    frame_count = 0
    cost = 0.0
    start_time = time.time()
    # This thread owns `vcam` from here on: it may close and reopen it, so main
    # must not assume its own reference is still live.
    vcam_retry_at = 0.0
    vcam_failures = 0
    full_frame = None
    crop_coords = None
    pristine_full_frame = None
    
    last_target_time = time.time()
    engine_fps_estimate = 30.0

    while not STOP.is_set():
        t0 = time.perf_counter()
        try:
            item = Q_OUT.get_nowait()
            if item is None:
                break
            
            is_frozen = False
            full_frame = None
            crop_coords = None
            pristine_full_frame = None      # reset per dequeued frame
            if len(item) == 7:
                raw_out_frame, soft_mask, original_frame_rgb, cond_final, is_frozen, full_frame, crop_coords = item
            elif len(item) == 5:
                raw_out_frame, soft_mask, original_frame_rgb, cond_final, is_frozen = item
            elif len(item) == 4:
                raw_out_frame, soft_mask, original_frame_rgb, cond_final = item
            else:
                raw_out_frame, soft_mask, original_frame_rgb = item
                cond_final = None
    
            # NOT gated on args.post_processing. That flag is store_true and
            # build_command never emitted it, so this entire stage was dead in
            # every GUI run: Sharpness, Saturation and Brightness each logged
            # their new value, passed the self-test, and changed nothing.
            #
            # The defaults are now neutral (0 / 0 / 0) rather than the old
            # hardcoded 1.0 / +20 / +10, so an untouched slider is a true no-op
            # and the picture matches what it was while this code was dead.
            # Turning the stage on WITHOUT that change would have suddenly
            # sharpened and saturated everyone's output.
            sharpness = float(state_dict.get("sharpness", args.sharpness))
            if abs(sharpness) > 0.01:
                amount = 0.4 * sharpness
                gaussian = cv2.GaussianBlur(raw_out_frame, (0, 0), 1.5)
                out_frame = cv2.addWeighted(raw_out_frame, 1.0 + amount,
                                            gaussian, -amount, 0)
            else:
                out_frame = raw_out_frame

            sat = int(state_dict.get("saturation", args.saturation))
            bright = int(state_dict.get("brightness", args.brightness))
            if sat or bright:
                hsv = cv2.cvtColor(out_frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                if sat:
                    s = cv2.add(s, sat) if sat > 0 else cv2.subtract(s, -sat)
                if bright:
                    v = cv2.add(v, bright) if bright > 0 else cv2.subtract(v, -bright)
                out_frame = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


            if state_dict.pop("save_snapshot", False):
                save_dir = os.path.join(SCRIPT_DIR, "snapshots")
                os.makedirs(save_dir, exist_ok=True)
                ts = int(time.time() * 1000)
                cv2.imwrite(os.path.join(save_dir, f"snap_{ts}_ai.png"), out_frame)
                cv2.imwrite(os.path.join(save_dir, f"snap_{ts}_webcam.png"), cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR))
                if cond_final is not None:
                    # --controlnet multi concatenates two detectors, so this is
                    # 6 channels, not 3, and cvtColor raised cv2.error. The only
                    # handler around this block catches queue.Empty, so the
                    # exception escaped postprocess_thread, tripped FAILED and
                    # took the engine down with exit 1 — one Snapshot press.
                    try:
                        cond_img = (cond_final[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        ch_n = cond_img.shape[2] if cond_img.ndim == 3 else 1
                        if ch_n >= 3:
                            cond_img = cv2.cvtColor(cond_img[:, :, :3], cv2.COLOR_RGB2BGR)
                        elif ch_n == 1:
                            cond_img = cv2.cvtColor(cond_img[:, :, 0], cv2.COLOR_GRAY2BGR)
                        else:
                            cond_img = cv2.cvtColor(cond_img[:, :, 0], cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(os.path.join(save_dir, f"snap_{ts}_edges.png"), cond_img)
                    except Exception as e:
                        log(f"[Engine] Snapshot: skipped the ControlNet preview ({e}).")
                log(f"[Engine] Saved high-res snapshot to snapshots/snap_{ts}_*.png")
    
            # Only a fallback now. When full_frame/crop_coords are present the
            # HD paste-back below composites at full resolution, and doing it
            # here as well applied the mask TWICE: a pixel with alpha 0.5 came
            # out 0.25*AI instead of 0.5*AI, so the feathered silhouette edge
            # was thinner and washed out compared with the Mask Feather value.
            if soft_mask is not None and (full_frame is None or crop_coords is None):
                alpha = soft_mask[..., np.newaxis].astype(np.float32)
                if bg_img_cache is not None:
                    out_frame = (out_frame.astype(np.float32) * alpha + bg_img_cache.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                elif state_dict.get("composite", args.composite):
                    bg_to_use = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
                    bokeh = state_dict.get("bokeh_blur", args.bokeh_blur)
                    if bokeh > 0.01:
                        blur_kernel = int(bokeh * 40)
                        if blur_kernel % 2 == 0: blur_kernel += 1
                        bg_to_use = cv2.GaussianBlur(bg_to_use, (blur_kernel, blur_kernel), 0)
                    out_frame = (out_frame.astype(np.float32) * alpha + bg_to_use.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            
            target_frame = out_frame.astype(np.float32)
            
            now = time.time()
            dt = now - last_target_time
            if dt > 0.001:
                engine_fps_estimate = engine_fps_estimate * 0.8 + (1.0 / dt) * 0.2
            last_target_time = now
            
            if current_smoothed_frame is None:
                current_smoothed_frame = target_frame.copy()
        except queue.Empty:
            pass

        if target_frame is None:
            time.sleep(0.033)
            continue

        # full_frame is only valid for the iteration that dequeued it. It was
        # left set on `queue.Empty`, so the HD composite below re-ran on the
        # array it had already written into — ~3x per engine frame at 10fps.
        # Screen/Color Dodge/Overlay compound and saturate to white (badly
        # during a 30-60s LoRA refit, when nothing dequeues at all), and VFX
        # Opacity renders far stronger than its value. Re-composite from the
        # pristine copy instead.
        if full_frame is not None:
            if pristine_full_frame is None:
                pristine_full_frame = full_frame.copy()
            full_frame = pristine_full_frame.copy()

        # Lerp current frame towards target frame for buttery 30 FPS motion blur
        alpha = state_dict.get("motion_smoothing", args.motion_smoothing)
        current_smoothed_frame = cv2.addWeighted(target_frame, 1.0 - alpha, current_smoothed_frame, alpha, 0)
        display_frame = current_smoothed_frame.astype(np.uint8)

        vfx_blend_mode = state_dict.get("vfx_blend_mode", "Normal")
        # Unclamped, a value above 1.0 extrapolates highlights past 255 and
        # the uint8 cast at the end of the blend wraps them to black.
        vfx_opacity = max(0.0, min(1.0, float(state_dict.get("vfx_opacity", 1.0))))
        
        # HD VFX Compositing
        if full_frame is not None and crop_coords is not None:
            cx, cy, cw, ch = crop_coords
            # Protect bounds
            fh, fw = full_frame.shape[:2]
            cx = max(0, min(cx, fw - 1))
            cy = max(0, min(cy, fh - 1))
            cw = min(cw, fw - cx)
            ch = min(ch, fh - cy)
            
            if cw > 0 and ch > 0:
                ai_resized = cv2.resize(display_frame, (cw, ch))
                
                hd_region = full_frame[cy:cy+ch, cx:cx+cw].copy()
                
                mask_resized = None
                if soft_mask is not None:
                    mask_resized = cv2.resize(soft_mask, (cw, ch))[..., np.newaxis]
                
                base_region = hd_region.astype(np.float32)
                ai_region = ai_resized.astype(np.float32)
                
                # The dropdown offers twelve modes; only three were implemented
                # and the rest silently fell through to Normal — while the
                # listener still logged the name, so the self-test passed them.
                if vfx_blend_mode == "Multiply":
                    blended = base_region * ai_region / 255.0
                elif vfx_blend_mode == "Darken":
                    blended = np.minimum(base_region, ai_region)
                elif vfx_blend_mode == "Lighten":
                    blended = np.maximum(base_region, ai_region)
                elif vfx_blend_mode == "Difference":
                    blended = np.abs(base_region - ai_region)
                elif vfx_blend_mode == "Exclusion":
                    blended = base_region + ai_region - 2.0 * base_region * ai_region / 255.0
                elif vfx_blend_mode == "Linear Dodge (Add)":
                    blended = np.clip(base_region + ai_region, 0, 255)
                elif vfx_blend_mode == "Hard Light":
                    m = ai_region < 128
                    blended = np.empty_like(base_region)
                    blended[m] = 2.0 * base_region[m] * ai_region[m] / 255.0
                    blended[~m] = 255.0 - 2.0 * (255.0 - base_region[~m]) * (255.0 - ai_region[~m]) / 255.0
                elif vfx_blend_mode == "Soft Light":
                    b, a = base_region / 255.0, ai_region / 255.0
                    blended = np.clip(((1 - 2 * a) * b * b + 2 * a * b) * 255.0, 0, 255)
                elif vfx_blend_mode == "Screen":
                    blended = 255.0 - ((255.0 - base_region) * (255.0 - ai_region) / 255.0)
                elif vfx_blend_mode == "Color Dodge":
                    blended = np.clip(base_region / (1.0001 - ai_region/255.0), 0, 255)
                elif vfx_blend_mode == "Overlay":
                    mask = base_region < 128
                    blended = np.empty_like(base_region)
                    blended[mask] = 2.0 * base_region[mask] * ai_region[mask] / 255.0
                    blended[~mask] = 255.0 - 2.0 * (255.0 - base_region[~mask]) * (255.0 - ai_region[~mask]) / 255.0
                else:
                    blended = ai_region
                
                final_ai = (blended * vfx_opacity) + (base_region * (1.0 - vfx_opacity))
                
                if mask_resized is not None:
                    # The background put behind the subject at 512 used to be
                    # discarded here: this line pasted the real camera room back
                    # wherever the mask was ~0. A BG image was therefore never
                    # visible at all, and Background Bokeh blurred a background
                    # nobody ever saw. Both now apply at full resolution.
                    bg_hd = base_region
                    if bg_img_cache is not None:
                        bg_hd = cv2.resize(bg_img_cache, (cw, ch)).astype(np.float32)
                    else:
                        bokeh = state_dict.get("bokeh_blur", args.bokeh_blur)
                        if bokeh > 0.01:
                            bk = max(1, int(bokeh * 40)) | 1
                            bg_hd = cv2.GaussianBlur(hd_region, (bk, bk), 0).astype(np.float32)
                    final_ai = (final_ai * mask_resized) + (bg_hd * (1.0 - mask_resized))

                    
                full_frame[cy:cy+ch, cx:cx+cw] = final_ai.astype(np.uint8)
                display_frame = full_frame

        # One transient send failure used to disable the virtual camera for the
        # rest of the session — OBS briefly grabbing the device during startup
        # was enough, and the only sign was a single log line while the GUI
        # preview carried on working. Reopen it on a cooldown instead.
        if vcam is None and args.virtual_camera and time.time() >= vcam_retry_at:
            vcam_retry_at = time.time() + 5.0
            try:
                import pyvirtualcam
                vcam = pyvirtualcam.Camera(width=1024, height=1024, fps=30)
                log(f"[Engine] Virtual camera reconnected: {vcam.device}")
                vcam_failures = 0
            except Exception as e:
                vcam_failures += 1
                if vcam_failures in (1, 6, 60):
                    log(f"[Engine] Virtual camera reconnect failed "
                        f"(attempt {vcam_failures}): {e}")

        # A/B: show the untouched camera frame while the key is held.
        if state_dict.get("ab_raw", False) and original_frame_rgb is not None:
            display_frame = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
            if full_frame is not None and crop_coords is not None:
                cx2, cy2, cw2, ch2 = crop_coords
                fh2, fw2 = full_frame.shape[:2]
                cx2, cy2 = max(0, min(cx2, fw2 - 1)), max(0, min(cy2, fh2 - 1))
                cw2, ch2 = min(cw2, fw2 - cx2), min(ch2, fh2 - cy2)
                if cw2 > 0 and ch2 > 0:
                    full_frame[cy2:cy2 + ch2, cx2:cx2 + cw2] = cv2.resize(display_frame, (cw2, ch2))
                    display_frame = full_frame

        if vcam is not None:
            # display_frame became the full HD camera frame when the HD
            # paste-back landed; squeezing 1280x720 into the 1024x1024 device
            # stretched everyone vertically by 1.78x in OBS while the GUI
            # preview (which is not resized) still looked correct. Letterbox
            # instead, so nothing is distorted and nothing is cropped away.
            _dh, _dw = display_frame.shape[:2]
            if _dw == _dh:
                hd = cv2.resize(display_frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            else:
                _sc = min(1024.0 / _dw, 1024.0 / _dh)
                _nw, _nh = max(1, int(round(_dw * _sc))), max(1, int(round(_dh * _sc)))
                hd = np.zeros((1024, 1024, 3), dtype=np.uint8)
                _ox, _oy = (1024 - _nw) // 2, (1024 - _nh) // 2
                hd[_oy:_oy + _nh, _ox:_ox + _nw] = cv2.resize(
                    display_frame, (_nw, _nh), interpolation=cv2.INTER_LINEAR)
            try:
                vcam.send(cv2.cvtColor(hd, cv2.COLOR_BGR2RGB))
                vcam.sleep_until_next_frame()
            except Exception as e:
                log(f"[Engine] Virtual camera send failed: {e}")
                try:
                    vcam.close()
                except Exception:
                    pass
                vcam = None
                vcam_retry_at = time.time() + 5.0

        if zmq_socket is not None:
            ok, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                try:
                    status_byte = b'\x01' if is_frozen else b'\x00'
                    zmq_socket.send(status_byte + buffer.tobytes())
                except Exception:
                    pass
        
        # Measure the work BEFORE the pacing sleep. Including the sleep made
        # this stat read ~37ms every single time regardless of what the thread
        # was actually doing, which is worse than having no stat at all.
        cost += time.perf_counter() - t0
        frame_count += 1

        if vcam is None:
            time.sleep(0.033)

        elapsed = time.time() - start_time
        if elapsed > 10.0 and frame_count:
            log(f"[Engine] output {frame_count / elapsed:.1f} fps, "
                f"{cost / frame_count * 1000:.1f}ms work/frame "
                f"(composite {'on' if state_dict.get('composite', args.composite) else 'off'}, "
                f"smoothing on)")
            start_time = time.time()
            frame_count = 0
            cost = 0.0

    # Close the camera this thread is actually holding, which may not be the
    # object main was handed if a reconnect happened.
    if vcam is not None:
        try:
            vcam.close()
        except Exception:
            pass

# ------------------------------------------------------------------ main ----
def build_args():
    parser = argparse.ArgumentParser(description="TensorRT AI VTuber engine")
    parser.add_argument("--prompt", type=str, default="cat girl, masterpiece, ultra-detailed")
    parser.add_argument("--negative_prompt", type=str,
                        default="lowres, bad anatomy, bad hands, text, error")
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--lora", type=str, default="None")
    parser.add_argument("--guidance_scale", type=float, default=1.4)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--freeze_threshold", type=float, default=0.98)
    parser.add_argument("--motion_smoothing", type=float, default=0.6)
    parser.add_argument("--bokeh_blur", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--expr_overrides", type=str, default="{}")
    parser.add_argument("--sens_overrides", type=str, default="{}")
    parser.add_argument("--t_index", type=int, default=32,
                        help="Denoise start step out of 50. Lower = more AI "
                             "stylisation, higher = closer to the raw webcam.")
    parser.add_argument("--frame_buffer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cfg_type", type=str, default="full",
                        choices=["none", "self", "initialize", "full"])
    parser.add_argument("--mirror_camera", action="store_true")
    parser.add_argument("--post_processing", action="store_true")
    parser.add_argument("--virtual_camera", action="store_true")
    parser.add_argument("--zmq_port", type=int, default=-1)
    parser.add_argument("--cmd_port", type=int, default=-1)
    parser.add_argument("--bg_image", type=str, default="")
    # These were being sent by launcher.py but never declared here, so argparse
    # exited with code 2 and the engine died instantly whenever they were on.
    parser.add_argument("--audio_sync", action="store_true")
    parser.add_argument("--composite", action="store_true",
                        help="Keep the real webcam background and composite the "
                             "AI avatar over it.")
    parser.add_argument("--no_face_track", action="store_true")
    # Live image controls. Defaults reproduce the previously hardcoded values.
    parser.add_argument("--sharpness", type=float, default=0.0)
    parser.add_argument("--saturation", type=int, default=0)
    parser.add_argument("--brightness", type=int, default=0)
    parser.add_argument("--mask_feather", type=int, default=7)
    parser.add_argument("--stillness_blend", type=float, default=0.3)
    parser.add_argument("--vfx_opacity", type=float, default=1.0)
    parser.add_argument("--vfx_blend_mode", type=str, default="Normal")
    # Was hardcoded to 2 inside StreamDiffusion.prepare(); exposing it makes a
    # given prompt + seed reproducible between runs.
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--lora_strength", type=float, default=1.0)
    parser.add_argument("--segment_every", type=int, default=1,
                        help="Run the segmenter every Nth frame and reuse the "
                             "mask in between. 1 = every frame (unchanged).")
    parser.add_argument("--mesh_every", type=int, default=1,
                        help="Run the FaceMesh expression tracker every Nth "
                             "frame. 1 = every frame (unchanged).")
    parser.add_argument("--no_segment", action="store_true",
                        help="Skip MediaPipe selfie segmentation entirely. Without "
                             "it the input is grey-screened before inference AND the "
                             "result is masked to your silhouette during the HD "
                             "paste-back, so the AI can never cover the whole frame. "
                             "Default off — existing behaviour is unchanged.")
    parser.add_argument("--normalize_lighting", action="store_true")
    parser.add_argument("--temporal_denoise", type=float, default=0.4,
                        help="0 disables. Lower = smoother but laggier input.")
    parser.add_argument("--mock_camera", action="store_true")
    parser.add_argument("--cuda_graph", action="store_true",
                        help="~10%% faster but has been observed to emit stale / "
                             "corrupted frames on some driver versions.")
    parser.add_argument("--controlnet", type=str, default="none",
                        choices=["none", "depth", "canny", "lineart", "openpose", "multi"],
                        help="Enable ControlNet preprocessor and engine.")

    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"[Engine] Ignoring unknown arguments: {' '.join(unknown)}")

    
    # NOTE: --no_color_lock and the Reinhard color_transfer() it controlled were
    # removed. The 2-step/full-CFG output does not strobe the way the 1-step
    # output did, so the stabiliser was taken out of postprocess_thread — and a
    # flag that silently controls nothing is exactly how this project lost a day
    # to three inert sliders. If flicker ever returns, the implementation is in
    # realtime_video_backup.py and in git history.
    # A background image only makes sense if we are compositing — but only
    # honour one that actually exists. A stale path used to silently switch
    # compositing on and then fall back to the real webcam background, i.e. the
    # worst of both worlds.
    if args.bg_image and not os.path.exists(args.bg_image):
        args.bg_image = None
    if args.bg_image:
        args.composite = True
    # Floor of 2, not 0: the two-step schedule needs room for a distinct first
    # step below t_index. The GUI never sends below 12.
    args.t_index = max(2, min(49, args.t_index))
    # frame_buffer > 1 duplicates ONE frame (`x_in.repeat(fb, ...)`) and then
    # keeps only `output_image[-1]`, so it is 2x the UNet work for an identical
    # picture. Nothing in the GUI selects it; clamp so a stray CLI flag cannot
    # silently halve the frame rate.
    if args.frame_buffer > 1:
        log(f"[Engine] frame_buffer={args.frame_buffer} ignored: it duplicates a "
            f"single frame and discards the extra output. Using 1.")
        args.frame_buffer = 1
    return args


def cmd_listener_thread(port, state_dict):
    import zmq
    import json
    import time
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://127.0.0.1:{port}")
    try:
        while not STOP.is_set():
            while True:
                try:
                    msg = socket.recv_string(flags=zmq.NOBLOCK)
                    try:
                        cmd = json.loads(msg)
                        if cmd.get("cmd") == "stop":
                            log("[Engine] Stop requested by the launcher.")
                            STOP.set()
                            break
                        if "negative_prompt" in cmd:
                            if cmd["negative_prompt"] != state_dict["negative_prompt"]:
                                state_dict["negative_prompt"] = cmd["negative_prompt"]
                                state_dict["prompt_dirty"] = True
                                log(f"[Engine] Negative Prompt: {cmd['negative_prompt']}")
                        if "prompt" in cmd:
                            if cmd["prompt"] != state_dict["base_prompt"]:
                                state_dict["base_prompt"] = cmd["prompt"]
                                state_dict["prompt_dirty"] = True
                                log(f"[Engine] Prompt: {cmd['prompt']}")
                        if "freeze_threshold" in cmd:
                            state_dict["freeze_threshold_dirty"] = cmd["freeze_threshold"]
                            log(f"[Engine] Freeze Threshold: {cmd['freeze_threshold']}")
                        if "motion_smoothing" in cmd:
                            state_dict["motion_smoothing"] = cmd["motion_smoothing"]
                            log(f"[Engine] Motion Smoothing: {cmd['motion_smoothing']}")
                        if "bokeh_blur" in cmd:
                            state_dict["bokeh_blur"] = cmd["bokeh_blur"]
                            log(f"[Engine] Bokeh Blur: {cmd['bokeh_blur']}")
                        if "zoom" in cmd:
                            state_dict["zoom"] = cmd["zoom"]
                            log(f"[Engine] Zoom: {cmd['zoom']}")
                        if "guidance_scale" in cmd:
                            state_dict["guidance_scale"] = cmd["guidance_scale"]
                            log(f"[Engine] CFG: {cmd['guidance_scale']}")
                        if "expr_override" in cmd:
                            if "expr_overrides" not in state_dict:
                                state_dict["expr_overrides"] = {}
                            state_dict["expr_overrides"].update(cmd["expr_override"])
                            state_dict["prompt_dirty"] = True
                            log(f"[Engine] Expression Override: {cmd['expr_override']}")
                        if "sens_override" in cmd:
                            if "sens_overrides" not in state_dict:
                                state_dict["sens_overrides"] = {}
                            state_dict["sens_overrides"].update(cmd["sens_override"])
                            log(f"[Engine] Sensitivity Override: {cmd['sens_override']}")
                        # --- commands the launcher has always sent but nothing
                        # here ever handled, so they were parsed and dropped.
                        # Each consumer already reads these state_dict keys.
                        if "composite" in cmd:
                            state_dict["composite"] = bool(cmd["composite"])
                            log(f"[Engine] Composite Real Background: {state_dict['composite']}")
                        if "no_segment" in cmd:
                            state_dict["no_segment"] = bool(cmd["no_segment"])
                            log(f"[Engine] Paint Whole Frame: {state_dict['no_segment']}")
                        if "manual_freeze" in cmd:
                            state_dict["manual_freeze"] = bool(cmd["manual_freeze"])
                            log(f"[Engine] Manual Freeze: {state_dict['manual_freeze']}")
                        if "save_snapshot" in cmd:
                            state_dict["save_snapshot"] = True
                            log("[Engine] Snapshot requested.")
                        if "vfx_opacity" in cmd:
                            state_dict["vfx_opacity"] = float(cmd["vfx_opacity"])
                            log(f"[Engine] VFX Opacity: {state_dict['vfx_opacity']:.2f}")
                        if "vfx_blend_mode" in cmd:
                            state_dict["vfx_blend_mode"] = cmd["vfx_blend_mode"]
                            log(f"[Engine] VFX Blend Mode: {state_dict['vfx_blend_mode']}")
                        for _k, _label in (("sharpness", "Sharpness"),
                                           ("saturation", "Saturation"),
                                           ("brightness", "Brightness"),
                                           ("mask_feather", "Mask Feather"),
                                           ("temporal_denoise", "Temporal Denoise"),
                                           ("stillness_blend", "Stillness Blend"),
                                           ("lora_strength", "LoRA Strength")):
                            if _k in cmd:
                                state_dict[_k] = cmd[_k]
                                log(f"[Engine] {_label}: {cmd[_k]}")
                        if "lora_refresh" in cmd:
                            # Re-fuse the CURRENT LoRA at the current strength.
                            # The strength slider used to log its new value and
                            # then do nothing until you happened to change LoRA.
                            state_dict["lora_refresh"] = True
                            log("[Engine] LoRA Refresh: re-fusing at the current strength.")
                        if "ab_raw" in cmd:
                            state_dict["ab_raw"] = bool(cmd["ab_raw"])
                        if "lora" in cmd:
                            # Consumed by the main loop, which owns the TRT refit.
                            state_dict["lora"] = cmd["lora"]
                            log(f"[Engine] LoRA requested: {cmd['lora']}")
                    except Exception as e:
                        log(f"[Engine] Bad command: {e}")
                except zmq.Again:
                    break
            time.sleep(0.05)
    finally:
        try:
            socket.close(linger=0)
            context.destroy(linger=0)
        except Exception:
            pass



class EngineStopped(Exception):
    """STOP arrived during a long startup step."""


def _abort_if_stopped(where):
    """STOP was honoured only AFTER the whole model load returned, so a stop
    during a 5-15 minute build was ignored, the launcher's 6s grace expired and
    it fell through to TerminateProcess — 'exited with code 1 (see the error
    above)' with no error above, and a risk of a truncated unet.engine.

    These checkpoints sit between the startup steps. The TensorRT build itself
    is a single opaque call that cannot be interrupted, so a stop landing
    inside it still has to wait it out; everything around it now exits clean."""
    if STOP.is_set():
        log(f"[Engine] Stop received during startup ({where}); exiting cleanly.")
        raise EngineStopped(where)


def load_model_and_engine(args, lora_name):
    from diffusers import AutoencoderTiny, StableDiffusionPipeline
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
    
    log("[Engine] Loading base model (kohaku-v2.1)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    log("[Engine] Swapping in TinyVAE...")
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd", torch_dtype=torch.float16
    ).to("cuda")

    embeddings_dir = os.path.join(SCRIPT_DIR, "embeddings")
    if os.path.isdir(embeddings_dir):
        for emb in os.listdir(embeddings_dir):
            if emb.endswith(".safetensors") or emb.endswith(".pt"):
                emb_path = os.path.join(embeddings_dir, emb)
                token = os.path.splitext(emb)[0]
                log(f"[Engine] Loading Textual Inversion: {token}")
                pipe.load_textual_inversion(emb_path, token=token)

    # cfg_type changes trt_unet_batch_size, so it MUST be part of the cache key
    # or a non-default run silently reuses a wrong-shaped engine. The suffix is
    # omitted for "full" so every engine already on disk stays valid.
    cfg_suffix = "" if args.cfg_type == "full" else f"_cfg{args.cfg_type}"
    engine_dir = os.path.join(
        SCRIPT_DIR,
        f"engines_tinyvae_base_fb{args.frame_buffer}_steps{args.steps}{cfg_suffix}")

    if args.steps == 4:
        # Evenly spaced from lo to t_index. The old form was
        # [t-30, t-20, t-10, t] clamped at 0, which collapsed to duplicate
        # timesteps at high AI Strength — e.g. t_index 14 gave [0, 0, 4, 14],
        # one wasted UNet pass and near-maximum noise. This is IDENTICAL to the
        # old list for t_index >= 32 (where nothing clamped) and only differs
        # where the old one was already broken.
        lo = max(2, args.t_index - 30)
        span = args.t_index - lo
        t_list = [lo + round(span * i / 3) for i in range(4)]
        # guarantee strict ascent even in the degenerate small-span case
        for i in range(1, 4):
            if t_list[i] <= t_list[i - 1]:
                t_list[i] = t_list[i - 1] + 1
        t_list[-1] = max(t_list[-1], args.t_index)
    else:
        step1 = min(args.t_index - 2, max(10, args.t_index - 15))
        step1 = max(0, step1)
        t_list = [step1, args.t_index]
    
    stream = StreamDiffusion(
        pipe,
        t_index_list=t_list,
        torch_dtype=torch.float16,
        cfg_type=args.cfg_type,
        do_add_noise=True,
        use_denoising_batch=True,
        frame_buffer_size=args.frame_buffer,
    )

    log("[Engine] Loading LCM-LoRA...")
    stream.load_lcm_lora()
    stream.fuse_lora()
    
    log(f"[Engine] Applying TensorRT acceleration ({os.path.basename(engine_dir)}).")
    if not os.path.exists(os.path.join(engine_dir, "unet.engine")):
        log("[Engine] No cached engine found — the first build takes 5-15 minutes. Please wait.")
    _abort_if_stopped("before the TensorRT build")
    stream = accelerate_with_tensorrt(
        stream,
        engine_dir,
        max_batch_size=stream.trt_unet_batch_size,
        use_cuda_graph=args.cuda_graph,
        engine_build_options={
            "opt_batch_size": stream.trt_unet_batch_size,
            "build_enable_refit": True,
            "timing_cache": os.path.join(SCRIPT_DIR, "trt_global_timing.cache")
        },
    )

    # The user LoRA only ever reached the TensorRT UNet (via refit_lora_to_trt),
    # so stream.pipe.text_encoder kept BASE weights forever and LoRA trigger
    # tokens never conditioned anything. Fuse the text-encoder half here.
    #
    # Deliberately after acceleration: doing it before would bake the LoRA into
    # a freshly built engine, and engine_dir is shared "base" across every LoRA.
    # fuse_unet=False keeps the UNet untouched either way. Guarded, because the
    # old pre-PEFT diffusers 0.24 backend is fussy — a failure here degrades to
    # the previous behaviour rather than breaking a working engine.
    if lora_name and str(lora_name).lower() != "none":
        lora_path = os.path.join(SCRIPT_DIR, "loras", lora_name)
        if os.path.exists(lora_path):
            scale = float(getattr(args, "lora_strength", 1.0))
            try:
                # NOT pipe.load_lora_weights(): that always calls
                # load_lora_into_unet FIRST, and accelerate_with_tensorrt does
                # `del stream.pipe.unet` without ever restoring it. Diffusers'
                # __getattr__ then falls through to the config, so pipe.unet is
                # the tuple ('diffusers', 'UNet2DConditionModel') and every run
                # logged "'tuple' object has no attribute 'load_attn_procs'" —
                # the text encoder never saw the LoRA and trigger tokens did
                # nothing. fuse_unet=False could not help: the crash happened
                # in load_lora_weights, before fuse_lora was reached.
                # Loading straight into the text encoder skips the dead UNet.
                sd_lora, alphas = StableDiffusionPipeline.lora_state_dict(lora_path)
                StableDiffusionPipeline.load_lora_into_text_encoder(
                    sd_lora, network_alphas=alphas,
                    text_encoder=pipe.text_encoder, lora_scale=scale)
                pipe.fuse_lora(fuse_unet=False, fuse_text_encoder=True,
                               lora_scale=scale)
                log(f"[Engine] Fused '{lora_name}' into the TEXT ENCODER "
                    f"(strength {scale:.2f}) — trigger tokens now condition.")
            except Exception as e:
                log(f"[Engine] Text-encoder LoRA fuse skipped ({e}). The UNet "
                    f"refit still applies; only trigger words are weaker.")
        else:
            log(f"[Engine] LoRA not found for text-encoder fuse: {lora_path}")

    _abort_if_stopped("after the TensorRT build")
    controlnet_aux_models = {}
    if args.controlnet != "none":
        from streamdiffusion.acceleration.tensorrt.engine import UNet2DConditionModelEngine
        from controlnet_aux import MidasDetector, CannyDetector, LineartDetector, OpenposeDetector
        
        controlnet_engine_path = os.path.join(SCRIPT_DIR, "engines_controlnet", f"unet_{args.controlnet}.engine")
        if not os.path.exists(controlnet_engine_path):
            log(f"[FATAL] ControlNet engine not found at {controlnet_engine_path}. Please run compile_controlnet_fused.py --type {args.controlnet}")
            sys.exit(1)
            
        log(f"[Engine] Swapping base UNet engine with fused ControlNet {args.controlnet} engine...")
        stream.unet = UNet2DConditionModelEngine(
            controlnet_engine_path, 
            stream.unet.stream, 
            use_cuda_graph=args.cuda_graph
        )
        
        log(f"[Engine] Loading {args.controlnet} Estimator(s)...")
        if args.controlnet in ["depth", "multi"]:
            controlnet_aux_models["depth"] = MidasDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
        if args.controlnet in ["canny", "multi"]:
            controlnet_aux_models["canny"] = CannyDetector()
        if args.controlnet == "lineart":
            controlnet_aux_models["lineart"] = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
        if args.controlnet == "openpose":
            controlnet_aux_models["openpose"] = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

    return pipe, stream, controlnet_aux_models

def refit_prepare(stream, args, lora_name, state=None):
    """The slow half of a LoRA refit. Touches NO TensorRT state.

    Reloads the PyTorch pipeline, fuses the LoRAs into it, and exports the
    optimized ONNX the refit will read. Everything here works on a private
    pipeline and on files under `engine_dir/onnx/`; the only thing it reads
    from `stream` is `trt_unet_batch_size`, an int fixed at build time. It
    never touches the live engine, which is what makes it safe to run on a
    worker thread while inference keeps drawing frames.

    Returns `(base_onnx, refit_opt_onnx)` for `refit_apply`, or None when
    there is nothing to refit.
    """
    state = state if state is not None else {"lora_strength": getattr(args, "lora_strength", 1.0)}
    if args.controlnet != "none":
        log(f"[WARNING] LoRA hot-swapping is currently not supported when using Fused ControlNet ({args.controlnet}). Ignoring LoRA {lora_name}.")
        return None

    import gc
    from diffusers import StableDiffusionPipeline
    from streamdiffusion.acceleration.tensorrt.models import UNet
    from streamdiffusion.acceleration.tensorrt.utilities import export_onnx, optimize_onnx
    
    log(f"[Engine] Starting dynamic TRT Refit for LoRA: {lora_name}")
    # Must match load_model_and_engine's key exactly, cfg_suffix included, or a
    # non-default --cfg_type refits against the wrong ONNX directory and fails.
    cfg_suffix = "" if args.cfg_type == "full" else f"_cfg{args.cfg_type}"
    engine_dir = os.path.join(
        SCRIPT_DIR,
        f"engines_tinyvae_base_fb{args.frame_buffer}_steps{args.steps}{cfg_suffix}")
    
    base_onnx = os.path.join(engine_dir, "onnx", "unet.opt.onnx")
    refit_onnx = os.path.join(engine_dir, "onnx", "unet_refit.onnx")
    refit_opt_onnx = os.path.join(engine_dir, "onnx", "unet_refit.opt.onnx")

    # A copied engine cache has no onnx/ beside it, and the refit reads the
    # base graph from there. Fail here with a reason the user can act on,
    # rather than deep inside TensorRT after a 60-second export.
    if not os.path.exists(base_onnx):
        raise FileNotFoundError(
            f"{base_onnx} is missing, so this engine cannot be refitted. It only "
            f"exists on the machine that built the engine; delete {engine_dir} and "
            f"let it rebuild, or choose the LoRA before pressing START.")

    pipe = None
    try:
        log("[Engine] Reloading PyTorch UNet into RAM...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "KBlueLeaf/kohaku-v2.1",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        if lora_name and lora_name.lower() != "none":
            lora_path = os.path.join(SCRIPT_DIR, "loras", lora_name)
            if os.path.exists(lora_path):
                log(f"[Engine] Fusing {lora_name} into UNet...")
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=float(state.get("lora_strength", 1.0)))

        log("[Engine] Fusing LCM-LoRA...")
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()

        log("[Engine] Exporting temporary refit ONNX...")
        unet_model = UNet(
            fp16=True,
            device=pipe.device,
            max_batch_size=stream.trt_unet_batch_size,
            min_batch_size=1,
            embedding_dim=pipe.text_encoder.config.hidden_size,
            unet_dim=pipe.unet.config.in_channels,
        )

        export_onnx(pipe.unet, refit_onnx, unet_model, 512, 512, stream.trt_unet_batch_size, 17)

        log("[Engine] Optimizing refit ONNX...")
        optimize_onnx(refit_onnx, refit_opt_onnx, unet_model)
    finally:
        # Free the second copy of the UNet here, not at the call site: the
        # refit reads its ONNX off disk and needs nothing from `pipe`. Giving
        # back ~2GB before we go anywhere near the engine also keeps the VRAM
        # peak down, which matters now that this runs alongside live inference.
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return base_onnx, refit_opt_onnx


def refit_apply(stream, base_onnx, refit_opt_onnx):
    """The only part of a refit that mutates the live engine.

    MUST run on the inference thread. `engine.refit` rewrites the weights the
    execution context reads, so it cannot overlap a `stream()` call, and
    clearing `cuda_graph_instance` forces a re-capture — which is only legal
    when no other thread is issuing CUDA work.
    """
    log("[Engine] Refitting TensorRT Engine in VRAM... (Expect a 1-second freeze)")
    stream.unet.engine.refit(base_onnx, refit_opt_onnx)
    stream.unet.engine.cuda_graph_instance = None
    log("[Engine] Dynamic Refit Complete! Resuming stream.")


def refit_lora_to_trt(stream, args, lora_name, state=None):
    """Blocking prepare + apply, for the startup path where nothing is
    streaming yet. The hot-swap path in `main` runs the two halves separately
    so the picture keeps moving through the slow one."""
    prepared = refit_prepare(stream, args, lora_name, state)
    if prepared is None:
        return
    refit_apply(stream, *prepared)

def main():
    args = build_args()
    os.chdir(SCRIPT_DIR)
    open_log()
    install_signal_handlers()
    log_environment(args)

    zmq_socket = None
    if args.zmq_port > 0:
        import zmq
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUB)
        zmq_socket.setsockopt(zmq.SNDHWM, 2)     # drop stale preview frames
        zmq_socket.setsockopt(zmq.LINGER, 0)
        zmq_socket.bind(f"tcp://127.0.0.1:{args.zmq_port}")
        log(f"[Engine] Preview stream bound to port {args.zmq_port}")

    if not torch.cuda.is_available():
        log("[FATAL] No CUDA device found. This engine needs an NVIDIA GPU.")
        sys.exit(1)

    import json
    try:
        expr_overrides = json.loads(args.expr_overrides)
    except Exception:
        expr_overrides = {}

    try:
        sens_overrides = json.loads(args.sens_overrides)
    except Exception:
        sens_overrides = {}

    state_dict = {"base_prompt": args.prompt,
                  "negative_prompt": args.negative_prompt,
                  "expr_overrides": expr_overrides,
                  "sens_overrides": sens_overrides,
                  "composite": args.composite,
                  "no_segment": args.no_segment,
                  "sharpness": args.sharpness,
                  "saturation": args.saturation,
                  "brightness": args.brightness,
                  "mask_feather": args.mask_feather,
                  "temporal_denoise": args.temporal_denoise,
                  "stillness_blend": args.stillness_blend,
                  "vfx_opacity": args.vfx_opacity,
                  "vfx_blend_mode": args.vfx_blend_mode,
                  "lora_strength": args.lora_strength,
                  "prompt_dirty": True}

    # Bind the command socket BEFORE the model load. It needs nothing from the
    # model, and starting it afterwards meant there was no listener during the
    # 5-15 minute TensorRT build — so STOP failed silently, the launcher waited
    # out its 6s timeout and then TerminateProcess'd the engine (exit code 1,
    # phantom "see the error above", and a risk of a truncated unet.engine).
    if args.cmd_port > 0:
        cmd_t = threading.Thread(target=cmd_listener_thread, args=(args.cmd_port, state_dict), daemon=True)
        cmd_t.start()
        log(f"[Engine] Listening for commands on port {args.cmd_port}")

    try:
        pipe, stream, controlnet_aux_models = load_model_and_engine(args, args.lora)
    except EngineStopped:
        return
    if STOP.is_set():
        log("[Engine] Stop received during startup; exiting before inference.")
        return

    current_lora = args.lora if (args.lora and args.lora.lower() != "none") else "None"
    if current_lora != "None":
        # Blocking is correct here: nothing is streaming yet, so there is no
        # picture to protect, and START must not begin before the weights the
        # user asked for are actually in the engine.
        _abort_if_stopped("before the startup LoRA refit")
        refit_lora_to_trt(stream, args, args.lora, state_dict)
        if STOP.is_set():
            log("[Engine] Stop received after the LoRA refit; exiting cleanly.")
            return

    # In-flight background LoRA prep, or None. Only ever one at a time — a
    # second swap request stays in state_dict until this one lands, rather
    # than racing two exports over the same unet_refit.onnx.
    refit_job = None

    # The live CFG value. Must be a separate variable from args.guidance_scale,
    # because the freeze branch in the loop rewrites stream.guidance_scale every
    # iteration and would otherwise clobber whatever the slider sent.
    live_cfg = args.guidance_scale

    log(f"[Engine] Preparing (t_index={args.t_index}, cfg={args.cfg_type}, "
        f"guidance={args.guidance_scale}, delta={args.delta})")
    stream.prepare(
        prompt=state_dict["base_prompt"] + (", closed mouth" if args.audio_sync else ""),
        negative_prompt=args.negative_prompt,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
        seed=args.seed,
    )



    cam_id = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = ThreadedCamera(cam_id, mock=args.mock_camera)
    if not cap.isOpened():
        log(f"\n[FATAL] Could not open camera '{args.camera}'.")
        log("Another app (Discord, Zoom, OBS, Teams) may be holding it, or the index is wrong.")
        found = scan_cameras()
        if found:
            log(f"Cameras that DID open: {found} — set Camera Index to one of these.")
        else:
            log("No cameras responded at all on indices 0-5.")
        sys.exit(1)

    vcam = None
    if args.virtual_camera:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=1024, height=1024, fps=30)
            log(f"[Engine] OBS Virtual Camera started: {vcam.device}")
        except Exception as e:
            log(f"[Engine] Virtual camera unavailable ({e}). Start OBS's virtual cam once, then retry.")

    audio_tracker = None
    if args.audio_sync:
        try:
            from audio_sync import AudioTracker
            audio_tracker = AudioTracker()
            log("[Engine] Audio lip-sync active.")
        except Exception as e:
            log(f"[Engine] Audio lip-sync unavailable ({e}).")

    cam_t = threading.Thread(target=camera_thread, args=(cap, args, state_dict, controlnet_aux_models),
                             name="camera", daemon=True)
    post_t = threading.Thread(target=postprocess_thread, args=(args, zmq_socket, vcam, state_dict),
                              name="postprocess", daemon=True)
    cam_t.start()
    post_t.start()

    log("[Engine] READY — streaming frames.")
    starved = 0
    nan_frames = 0
    stat_n = 0
    stat_wait = 0.0
    stat_infer = 0.0
    warmup_until = time.time() + 5.0   # TensorRT's first calls are not typical
    stat_start = warmup_until
    
    current_emotions = []
    embed_shape_warned = False
    prev_frame_rgb = None
    out_np = None

    try:
        while not STOP.is_set():
            # NOTE: do NOT pop "freeze_threshold_dirty" here. The only
            # consumer is ~130 lines below (`freeze_threshold = state_dict.get(...)`)
            # and this pop threw the value away first, so the slider reverted to
            # the launch-time value after at most one frame — taking the
            # Stillness Blend slider with it, since that only applies while
            # is_frozen. The listener overwrites the key, so it stays current.
            #
            # We do NOT use StreamDiffusion's similar_image_filter because its
            # cosine similarity metric conflicts with our structural MSE metric,
            # causing 1-FPS ghosting. We handle freezing entirely via is_frozen.
            
            # The CFG slider writes live_cfg, NOT stream.guidance_scale. The
            # freeze branch below reassigns stream.guidance_scale on every
            # single iteration, so anything written here was overwritten ~140
            # lines later, before stream() ever saw it: the slider logged its
            # new value and changed nothing. Same trap as the freeze threshold.
            # (This block used to appear twice, verbatim; the first pop() made
            # the second copy unreachable.)
            if "guidance_scale" in state_dict:
                live_cfg = state_dict.pop("guidance_scale")
                # When turning off CFG, we MUST empty the prompt tensor so StreamDiffusion disables CFG!
                if live_cfg <= 1.0 and stream.prompt_embeds.shape[0] > stream.batch_size:
                    stream.prompt_embeds = stream.prompt_embeds[stream.batch_size:]
                # When turning ON CFG, we must re-prepare!
                elif live_cfg > 1.0 and stream.prompt_embeds.shape[0] == stream.batch_size:
                    state_dict["prompt_dirty"] = True

            t_wait = time.perf_counter()
            
            manual_freeze = state_dict.get("manual_freeze", False)
            if manual_freeze and 'frame_rgb' in locals():
                # If manually frozen, skip reading from webcam queue and reuse the last frame!
                # We also must clear Q_IN so the camera thread doesn't fill it with stale frames.
                try:
                    while True:
                        Q_IN.get_nowait()
                except queue.Empty:
                    pass
            else:
                try:
                    item = Q_IN.get(timeout=1.0)
                    frame_rgb, soft_mask, original_frame_rgb, emotions, cond_final, full_frame, crop_coords = item
                    starved = 0
                except queue.Empty:
                    # A silent hang used to look identical to a slow first frame.
                    starved += 1
                    if starved in (5, 15):
                        log(f"[Engine] No frames from the camera thread for {starved}s "
                            f"(camera_alive={cap.thread.is_alive() if cap.thread else False}, "
                            f"producer_alive={cam_t.is_alive()}).")
                    if starved >= 30:
                        # A stolen or unplugged webcam used to hang here forever at
                        # 0 FPS with the GUI frozen on the last frame.
                        log("[FATAL] No camera frames for 30s - the device was most "
                            "likely unplugged or taken by another application.")
                        FAILED.set()
                        STOP.set()
                    continue

            if audio_tracker is not None:
                audio_sens = state_dict.get("sens_overrides", {}).get("audio", 1.5)
                audio_tracker.set_threshold(audio_sens)
                
                if audio_tracker.is_speaking:
                    # The face mesh now always emits one of "open mouth"/"closed
                    # mouth", so appending blindly produced a prompt asking for both
                    # at once. Replace rather than add.
                    if "open mouth" not in emotions:
                        emotions = ["open mouth" if e == "closed mouth" else e
                                    for e in emotions]
                        if "open mouth" not in emotions:
                            emotions.append("open mouth")
                    
            if emotions != current_emotions or state_dict["prompt_dirty"]:
                current_emotions = emotions.copy()
                state_dict["prompt_dirty"] = False
                overrides = state_dict.get("expr_overrides", {})
                mapped = [overrides.get(e, e) for e in emotions]
                mapped = [m for m in mapped if m.strip()] # filter out empty strings if user wants to disable an emotion
                emotion_str = ", ".join(mapped) if mapped else ""
                full_prompt = state_dict["base_prompt"] + (f", {emotion_str}" if emotion_str else "")
                
                # We must encode BOTH positive and negative prompts for cfg_type=args.cfg_type,
                # and use .copy_() to overwrite the existing tensor in-place.
                # Wrap in torch.no_grad() to prevent massive VRAM leak when emotions change rapidly!
                with torch.no_grad():
                    encoder_output = stream.pipe.encode_prompt(
                        prompt=full_prompt,
                        device=stream.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=state_dict["negative_prompt"],
                    )
                cond_embeds = encoder_output[0].repeat(stream.batch_size, 1, 1)
                # Match whatever prepare() actually built instead of assuming the
                # [uncond, cond] layout. At guidance_scale exactly 1.0 (the CFG
                # slider's old minimum) StreamDiffusion disables CFG and
                # prompt_embeds is half this size, so the assumption made copy_()
                # raise a shape error and killed the engine on the next prompt
                # change — one click of the slider away.
                if stream.prompt_embeds.shape[0] == cond_embeds.shape[0] * 2:
                    uncond_embeds = encoder_output[1].repeat(stream.batch_size, 1, 1)
                    new_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
                else:
                    new_embeds = cond_embeds

                if new_embeds.shape == stream.prompt_embeds.shape:
                    stream.prompt_embeds.copy_(new_embeds)
                elif not embed_shape_warned:
                    embed_shape_warned = True
                    log(f"[Engine] Prompt update skipped: built "
                        f"{tuple(new_embeds.shape)} but the pipeline expects "
                        f"{tuple(stream.prompt_embeds.shape)}.")
                # No torch.cuda.empty_cache() here: torch.no_grad() above is what
                # fixed the VRAM leak. empty_cache() is a device-wide sync, and
                # since a blink flips "closed eyes" in and out of the prompt it
                # was stalling the loop several times a second.

            t_got = time.perf_counter()

            # Dynamic Stillness Enhancement
            is_frozen = False
            if hasattr(stream, "similar_filter") and stream.similar_filter is not None:
                stream.similar_filter.set_threshold(-1.0)
                
            similarity = 0.0
            if prev_frame_rgb is not None:
                # Calculate structural similarity. 1.0 = identical, drops towards 0.0 when moving.
                # Compute on the raw webcam frame (original_frame_rgb) to avoid black-background bias in composite mode.
                similarity = 1.0 / (1.0 + (np.mean((original_frame_rgb.astype(np.float32) - prev_frame_rgb.astype(np.float32)) ** 2) / 255.0))
            prev_frame_rgb = original_frame_rgb.copy()
            
            freeze_threshold = state_dict.get("freeze_threshold_dirty", args.freeze_threshold)
            if similarity > freeze_threshold and out_np is not None:
                is_frozen = True
                
            if state_dict.get("manual_freeze", False) and out_np is not None:
                is_frozen = True
                
            if is_frozen:
                stream.guidance_scale = min(live_cfg * 1.5, 4.0)
                # How much raw webcam gets folded back in while you hold
                # still. Was a hidden 0.3; 0.0 disables the feedback entirely.
                sb = float(state_dict.get("stillness_blend", args.stillness_blend))
                sb = max(0.0, min(1.0, sb))
                refinement_frame = cv2.addWeighted(out_np, 1.0 - sb, frame_rgb, sb, 0)
                # Skip PIL conversion and go straight to GPU Tensor
                x_in = torch.from_numpy(refinement_frame).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0
            else:
                stream.guidance_scale = live_cfg
                # Skip PIL conversion and go straight to GPU Tensor
                x_in = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0
            
            if args.frame_buffer > 1:
                input_imgs = x_in.repeat(args.frame_buffer, 1, 1, 1)
            else:
                input_imgs = x_in

            kwargs = {}
            if cond_final is not None:
                if args.frame_buffer > 1:
                    cond_final = cond_final.repeat(args.frame_buffer, 1, 1, 1)
                kwargs["controlnet_cond"] = cond_final

            output_image = stream(input_imgs, **kwargs)
            
            t_done = time.perf_counter()

            # --- LoRA hot-swap, phase 1: start the slow half off-thread -------
            # Everything expensive (pipeline reload, LoRA fuse, ONNX export and
            # optimize) runs on a worker while this loop keeps producing frames.
            # Only the engine refit itself comes back to this thread, below.
            # Guarded by refit_job: popping while a prep is in flight would
            # consume and DISCARD a swap the user asked for. test 16 caught
            # exactly that when this was written without the guard.
            pending_lora = None
            if refit_job is None:
                if "lora" in state_dict and state_dict["lora"] != current_lora:
                    pending_lora = state_dict.pop("lora")
                elif state_dict.pop("lora_refresh", False) and current_lora != "None":
                    # Same machinery, same LoRA, new strength.
                    pending_lora = current_lora
            if refit_job is None and pending_lora is not None:
                new_lora = pending_lora
                log(f"[Engine] LoRA hot-swap requested: {current_lora} -> {new_lora}")
                log(f"[Engine] === HOT-SWAP: preparing '{new_lora}' in the background. "
                    "The stream keeps running (slower — the GPU is shared) for "
                    "30-60s, then freezes ~1s for the engine refit. ===")
                refit_job = {"lora": new_lora, "result": None, "error": None}

                def _refit_worker(job=refit_job):
                    try:
                        job["result"] = refit_prepare(stream, args, job["lora"], state_dict)
                    except Exception as e:
                        job["error"] = e

                refit_job["thread"] = threading.Thread(
                    target=_refit_worker, daemon=True, name="refit-prep")
                refit_job["thread"].start()

            # --- phase 2: apply on THIS thread once the worker has finished ---
            if refit_job is not None and not refit_job["thread"].is_alive():
                job, refit_job = refit_job, None
                failure = None

                if job["error"] is not None:
                    log_exception("refit_prepare", job["error"])
                    failure = "preparation"
                elif job["result"] is None:
                    # ControlNet is fused into the engine; nothing to refit, but
                    # the name still advances so we don't retry every frame.
                    current_lora = job["lora"]
                else:
                    try:
                        refit_apply(stream, *job["result"])
                    except Exception as e:
                        log_exception("refit_apply", e)
                        failure = "the refit step"
                    else:
                        current_lora = job["lora"]
                        state_dict["prompt_dirty"] = True
                        log("[Engine] Re-preparing TensorRT embeddings after hot-swap...")
                        stream.prepare(
                            prompt=state_dict["base_prompt"] + (", closed mouth" if args.audio_sync else ""),
                            negative_prompt=state_dict["negative_prompt"],
                            num_inference_steps=50,
                            guidance_scale=args.guidance_scale,
                            delta=args.delta,
                            seed=args.seed,
                        )

                if failure:
                    # A failed refit must not kill a working stream — the old
                    # weights are still resident, so carry on with them, and
                    # leave current_lora naming what is actually loaded.
                    log(f"[Engine] Hot-swap to '{job['lora']}' FAILED during {failure}; "
                        f"continuing with the previously loaded weights "
                        f"('{current_lora}').")
                # The launcher greys out the LoRA dropdown on send and watches
                # for this line to re-enable it, so EVERY path above has to
                # reach it — including the failures. Bailing out early here is
                # what used to leave the dropdown greyed out for good.
                log(f"[Engine] LoRA hot-swap complete: now using '{current_lora}'.")

            if time.time() > warmup_until:
                stat_n += 1
                stat_wait += t_got - t_wait
                stat_infer += t_done - t_got
                elapsed = time.time() - stat_start
                if elapsed > 5.0 and stat_n:
                    # Splits the frame time into "starved waiting for the camera
                    # thread" vs "GPU busy", which is the only way to tell a
                    # producer problem from an inference problem.
                    log(f"[Engine] {stat_n / elapsed:.1f} FPS | "
                        f"wait {stat_wait / stat_n * 1000:.1f}ms  "
                        f"infer {stat_infer / stat_n * 1000:.1f}ms" +
                        (f" | {nan_frames} NaN frames" if nan_frames else ""))
                    stat_n, stat_wait, stat_infer = 0, 0.0, 0.0
                    stat_start = time.time()

            if output_image is None:
                continue

            if isinstance(output_image, np.ndarray):
                out_np = output_image
                if out_np.dtype != np.uint8:
                    out_np = (np.clip(out_np, 0.0, 1.0) * 255).astype(np.uint8)
                out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            else:
                # Fast GPU to CPU conversion, taking the newest frame
                out_t = output_image[-1] if len(output_image.shape) == 4 else output_image
                out_t = (out_t / 2 + 0.5).clamp(0, 1) # denormalize
                out_t = (out_t * 255).byte().permute(1, 2, 0)
                out_np = out_t.cpu().numpy()
                out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

            if Q_OUT.full():
                try:
                    Q_OUT.get_nowait()
                except queue.Empty:
                    pass
            try:
                Q_OUT.put((out_frame, soft_mask, original_frame_rgb, cond_final, is_frozen, full_frame, crop_coords), timeout=1.0)
            except queue.Full:
                pass
    except KeyboardInterrupt:
        log("[Engine] Interrupted.")
    finally:
        STOP.set()
        try:
            Q_OUT.put_nowait(None)
        except Exception:
            pass
        # Join BEFORE closing anything. postprocess_thread can be mid vcam.send()
        # or zmq send; tearing those down underneath it is a native-level crash.
        for t in (cam_t, post_t):
            t.join(timeout=3.0)
            if t.is_alive():
                log(f"[Engine] Worker '{t.name}' did not stop; leaving its "
                    f"resources open rather than freeing them underneath it.")
        # A refit prep thread has no cancel point — it is inside a torch ONNX
        # export — so give it a short grace period and then let it go. It owns
        # no camera, socket or log handle we are about to close; the only cost
        # of abandoning it is a half-written unet_refit.onnx, which the next
        # export overwrites before anything reads it.
        if refit_job is not None and refit_job["thread"].is_alive():
            log("[Engine] Waiting up to 2s for the background LoRA prep...")
            refit_job["thread"].join(timeout=2.0)
            if refit_job["thread"].is_alive():
                log("[Engine] LoRA prep still running at shutdown; abandoning it.")
        cap.release()
        if audio_tracker is not None:
            audio_tracker.close()
        # postprocess_thread owns vcam and closes its own reference on exit;
        # this is only a backstop for the case where that thread died. Wrapped
        # because closing an already-closed camera can raise, and an exception
        # here would abort the rest of the shutdown.
        if vcam is not None and not post_t.is_alive():
            try:
                vcam.close()
            except Exception:
                pass
        if zmq_socket is not None and not post_t.is_alive():
            zmq_socket.close(linger=0)
        log("[Engine] Shut down cleanly.")


def _exit_code_check():
    if FAILED.is_set():
        log("[Engine] Exiting non-zero: a worker thread failed.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
        # Checked HERE rather than at the end of main(), because main() has
        # early returns. The "Stop received during startup" return in
        # particular fired when a worker died during the model load (a
        # cmd_port collision sets FAILED and STOP from the excepthook), and
        # the process then exited 0 — reporting a hard startup failure to the
        # GUI as a clean shutdown.
        _exit_code_check()
    except SystemExit:
        raise
    except BaseException as e:
        # Model load and TensorRT failures land here. Without this the
        # traceback went to a stderr pipe that vanished with the window.
        log_exception("main()", e)
        sys.exit(1)
    finally:
        if _log_file:
            for _fh in _log_file:
                try:
                    _fh.close()
                except Exception:
                    pass
