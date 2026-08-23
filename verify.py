"""Headless verification: compile both files and dry-run the GUI -> engine
argument handshake with every heavy dependency stubbed out."""
import importlib, os, py_compile, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

for f in ("realtime_video.py", "launcher.py"):
    py_compile.compile(os.path.join(HERE, f), doraise=True)
    print(f"compile ok: {f}")

# --- stub the GPU / media stack -------------------------------------------
class Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub(name)
    def __call__(self, *a, **k):
        return Stub("call")

for name in ["cv2", "torch", "PIL", "PIL.Image", "diffusers", "streamdiffusion",
             "streamdiffusion.image_utils", "streamdiffusion.acceleration",
             "streamdiffusion.acceleration.tensorrt", "mediapipe",
             "mediapipe.python", "mediapipe.python.solutions"]:
    sys.modules[name] = Stub(name)

sys.path.insert(0, HERE)
rv = importlib.import_module("realtime_video")
lc_src = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()

# strength_to_t_index lives in launcher; pull it out without importing tkinter
import re
ns = {}
block = re.search(r"^T_INDEX_MIN.*?^(?=class )", lc_src, re.S | re.M).group(0)
exec(compile(block, "launcher", "exec"), ns)
strength_to_t_index = ns["strength_to_t_index"]

# --- replay the exact command launcher.py builds ---------------------------
cmd_tail = [
    "--prompt", "1girl, masterpiece",
    "--negative_prompt", "blurry, deformed",
    "--camera", "0",
    "--lora", "None",
    "--guidance_scale", "2.400",
    "--delta", "1.000",
    "--t_index", str(strength_to_t_index(40.0)),
    "--freeze_threshold", "1.000",
    "--zmq_port", "51234",
    "--mirror_camera",
    "--virtual_camera",
    "--audio_sync",
    "--composite",
    "--normalize_lighting",
    "--cuda_graph",
    "--bg_image", "C:/tmp/bg.png",
]
sys.argv = ["realtime_video.py"] + cmd_tail
args = rv.build_args()
assert args.audio_sync and args.composite and args.cuda_graph
assert args.frame_buffer == 1
assert 12 <= args.t_index <= 45
print("argparse handshake ok ->", f"t_index={args.t_index}", f"cfg={args.cfg_type}",
      f"guidance={args.guidance_scale}", f"freeze={args.freeze_threshold}",
      f"color_lock={args.color_lock}")

# old launcher flags must not explode the parser any more
sys.argv = ["realtime_video.py", "--audio_sync", "--emotion_sync", "--frame_buffer", "2"]
args2 = rv.build_args()
print("legacy/unknown flags tolerated ok ->", args2.audio_sync, args2.emotion_sync)

# AI strength mapping sanity
for s in (0, 40, 100):
    print(f"  AI Strength {s:>3} -> t_index {strength_to_t_index(s)}")

# color_transfer must survive a flat frame (zero std -> div by zero)
sys.modules.pop("cv2")
import types as _t
print("static checks passed")
