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
    print(msg, flush=True)
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
        f"cuda_graph={args.cuda_graph} audio_sync={args.audio_sync}")
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
def camera_thread(cap, args, state_dict):
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

    while not STOP.is_set():
        ret, frame, counter = cap.read(wait=True)
        if not ret or frame is None:
            continue

        if args.mirror_camera:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        size = min(h, w)

        if target_x == -1:
            target_x = (w - size) // 2
            target_y = (h - size) // 2
            current_x, current_y = target_x, target_y

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

        cropped = frame[current_y:current_y + size, current_x:current_x + size]
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
        if args.temporal_denoise > 0.0:
            f32 = frame_rgb.astype(np.float32)
            if prev_input is None:
                prev_input = f32
            else:
                cv2.accumulateWeighted(f32, prev_input, args.temporal_denoise)
                frame_rgb = cv2.convertScaleAbs(prev_input)

        soft_mask = None
        if segmenter is not None:
            try:
                small_rgb = cv2.resize(original_frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
                res = segmenter.process(small_rgb)
                if res.segmentation_mask is not None:
                    small_mask = cv2.GaussianBlur(res.segmentation_mask, (7, 7), 0)
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
            except Exception as e:
                log(f"[Camera] Segmentation failed ({e}); compositing the full AI frame.")
                soft_mask = None

        emotions = []
        if face_mesh is not None:
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

        # Always keep the freshest frame waiting in the slot. Skipping the work
        # whenever the slot was full (an earlier "optimisation" of mine) served
        # the camera wait and the CPU preprocessing *in series* with the GPU
        # instead of overlapping them, which cost more throughput than the
        # MediaPipe calls it saved. The producer must stay ahead of the GPU.
        if Q_IN.full():
            try:
                Q_IN.get_nowait()
            except queue.Empty:
                pass
        try:
            Q_IN.put_nowait((frame_rgb, soft_mask, original_frame_rgb, emotions))
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

    while not STOP.is_set():
        t0 = time.perf_counter()
        try:
            item = Q_OUT.get_nowait()
            if item is None:
                break
            raw_out_frame, soft_mask, original_frame_rgb = item
    
            # Post-process the newly arrived frame
            gaussian = cv2.GaussianBlur(raw_out_frame, (0, 0), 1.5)
            out_frame = cv2.addWeighted(raw_out_frame, 1.4, gaussian, -0.4, 0)
    
            hsv = cv2.cvtColor(out_frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, 20)
            v = cv2.add(v, 10)
            out_frame = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
            if soft_mask is not None:
                alpha = soft_mask[..., np.newaxis].astype(np.float32)
                if bg_img_cache is not None:
                    out_frame = (out_frame.astype(np.float32) * alpha + bg_img_cache.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                elif args.composite:
                    bg_to_use = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
                    bokeh = state_dict.get("bokeh_blur", args.bokeh_blur)
                    if bokeh > 0.01:
                        blur_kernel = int(bokeh * 40)
                        if blur_kernel % 2 == 0: blur_kernel += 1
                        bg_to_use = cv2.GaussianBlur(bg_to_use, (blur_kernel, blur_kernel), 0)
                    out_frame = (out_frame.astype(np.float32) * alpha + bg_to_use.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            
            target_frame = out_frame.astype(np.float32)
            if current_smoothed_frame is None:
                current_smoothed_frame = target_frame.copy()
        except queue.Empty:
            pass

        if target_frame is None:
            time.sleep(0.033)
            continue

        # Lerp current frame towards target frame for buttery 30 FPS motion blur
        alpha = state_dict.get("motion_smoothing", args.motion_smoothing)
        current_smoothed_frame = cv2.addWeighted(target_frame, 1.0 - alpha, current_smoothed_frame, alpha, 0)
        display_frame = current_smoothed_frame.astype(np.uint8)

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

        if vcam is not None:
            hd = cv2.resize(display_frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
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
                    zmq_socket.send(buffer.tobytes())
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
                f"(composite {'on' if args.composite else 'off'}, "
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
    parser.add_argument("--normalize_lighting", action="store_true")
    parser.add_argument("--temporal_denoise", type=float, default=0.4,
                        help="0 disables. Lower = smoother but laggier input.")
    parser.add_argument("--mock_camera", action="store_true")
    parser.add_argument("--cuda_graph", action="store_true",
                        help="~10%% faster but has been observed to emit stale / "
                             "corrupted frames on some driver versions.")

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
                    except Exception as e:
                        log(f"[Engine] Bad command: {e}")
                except zmq.Again:
                    break
            time.sleep(0.05)
    finally:
        try:
            socket.close(linger=0)
            context.term()
        except Exception:
            pass



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

    engine_dir = os.path.join(SCRIPT_DIR, f"engines_tinyvae_fb{args.frame_buffer}_steps{args.steps}")
    if lora_name and lora_name.lower() != "none":
        lora_path = os.path.join(SCRIPT_DIR, "loras", lora_name)
        if os.path.exists(lora_path):
            log(f"[Engine] Fusing character LoRA: {lora_name}")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora()
            safe = "".join(c for c in lora_name if c.isalnum() or c in ("-", "_"))
            safe = safe.replace("safetensors", "")
            engine_dir = os.path.join(SCRIPT_DIR, f"engines_tinyvae_{safe}_fb{args.frame_buffer}_steps{args.steps}")
        else:
            log(f"[Engine] LoRA not found at {lora_path} — continuing without it.")

    if args.steps == 4:
        t_list = [max(0, args.t_index - 30), max(0, args.t_index - 20), max(0, args.t_index - 10), args.t_index]
    else:
        step1 = min(args.t_index - 2, max(10, args.t_index - 15))
        step1 = max(0, step1)
        t_list = [step1, args.t_index]
    
    stream = StreamDiffusion(
        pipe,
        t_index_list=t_list,
        torch_dtype=torch.float16,
        cfg_type="full",
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
    stream = accelerate_with_tensorrt(
        stream,
        engine_dir,
        max_batch_size=stream.trt_unet_batch_size,
        use_cuda_graph=args.cuda_graph,
        engine_build_options={"opt_batch_size": stream.trt_unet_batch_size},
    )
    return pipe, stream

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

    pipe, stream = load_model_and_engine(args, args.lora)

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
                  "prompt_dirty": True}
    
    if args.cmd_port > 0:
        cmd_t = threading.Thread(target=cmd_listener_thread, args=(args.cmd_port, state_dict), daemon=True)
        cmd_t.start()
        log(f"[Engine] Listening for commands on port {args.cmd_port}")

    log(f"[Engine] Preparing (t_index={args.t_index}, cfg={args.cfg_type}, "
        f"guidance={args.guidance_scale}, delta={args.delta})")
    stream.prepare(
        prompt=state_dict["base_prompt"] + (", closed mouth" if args.audio_sync else ""),
        negative_prompt=args.negative_prompt,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
    )

    # 1.0 means "never skip"; anything lower freezes the output while you sit
    # still. The slider had been wired up but the call was commented out.
    if args.freeze_threshold < 0.999:
        stream.enable_similar_image_filter(threshold=args.freeze_threshold, max_skip_frame=10)
        log(f"[Engine] Similar-image filter on at {args.freeze_threshold}")

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

    cam_t = threading.Thread(target=camera_thread, args=(cap, args, state_dict),
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

    try:
        while not STOP.is_set():
            if "freeze_threshold_dirty" in state_dict:
                val = state_dict.pop("freeze_threshold_dirty")
                if hasattr(stream, "similar_filter") and stream.similar_filter is not None:
                    stream.similar_filter.set_threshold(val)
                elif val < 0.999:
                    stream.enable_similar_image_filter(threshold=val, max_skip_frame=10)

            t_wait = time.perf_counter()
            try:
                frame_rgb, soft_mask, original_frame_rgb, emotions = Q_IN.get(timeout=1.0)
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
                    log("[FATAL] No camera frames for 30s — the device was most "
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
                
                # We must encode BOTH positive and negative prompts for cfg_type="full",
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
            output_image = stream(Image.fromarray(frame_rgb))
            
            t_done = time.perf_counter()

            if "lora" in state_dict and state_dict["lora"] != current_lora:
                new_lora = state_dict.pop("lora")
                log(f"[Engine] LoRA hot-swap requested: {current_lora} -> {new_lora}")
                current_lora = new_lora
                
                # Cleanup old engine
                import gc
                del stream
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                
                # Rebuild
                pipe, stream = load_model_and_engine(args, current_lora)
                state_dict["prompt_dirty"] = True
                
                # Re-warmup
                log("[Engine] Re-preparing TensorRT embeddings after hot-swap...")
                stream.prepare(
                    prompt=state_dict["base_prompt"] + (", closed mouth" if args.audio_sync else ""),
                    negative_prompt=state_dict["negative_prompt"],
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    delta=args.delta,
                )
                
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

            if isinstance(output_image, (list, tuple)):
                output_image = output_image[0] if output_image else None

            # The similar-image filter returns prev_image_result, which is None
            # on the very first skipped frame — previously this raised
            # UnboundLocalError on out_frame and killed the engine.
            if output_image is None:
                continue

            if isinstance(output_image, np.ndarray):
                out_np = output_image
            else:
                out_np = postprocess_image(output_image, output_type="np")[0]

            if out_np.dtype != np.uint8:
                # NaN/inf survives np.clip and casts to garbage bytes, which is
                # what "the AI output is glitching" actually looks like.
                if not np.isfinite(out_np).all():
                    nan_frames += 1
                    if nan_frames in (1, 10, 100, 1000):
                        finite = np.isfinite(out_np)
                        log(f"[Engine] WARNING: non-finite pixels from the model "
                            f"({(~finite).sum()}/{out_np.size} on frame #{nan_frames}). "
                            f"finite range [{out_np[finite].min() if finite.any() else float('nan'):.3f}, "
                            f"{out_np[finite].max() if finite.any() else float('nan'):.3f}]. "
                            f"This is an fp16 overflow in the UNet — try lowering CFG, "
                            f"or set --cfg_type none to rule out RCFG.")
                    out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
                out_np = (np.clip(out_np, 0.0, 1.0) * 255).astype(np.uint8)
            out_frame = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

            if Q_OUT.full():
                try:
                    Q_OUT.get_nowait()
                except queue.Empty:
                    pass
            try:
                Q_OUT.put((out_frame, soft_mask, original_frame_rgb), timeout=1.0)
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

    if FAILED.is_set():
        log("[Engine] Exiting non-zero: a worker thread failed.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
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
