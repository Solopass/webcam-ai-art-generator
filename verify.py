"""GPU-free checks: compile both files, then prove the GUI and the engine still
agree on the command line.

The bug that started this whole effort was launcher.py sending `--audio_sync`
when realtime_video.py had never declared it — argparse exited with code 2 and
the engine died instantly. So rather than replaying a hand-written command that
drifts out of date, this reads every flag build_command() can actually emit and
checks the engine's parser accepts each one.
"""
import importlib
import os
import py_compile
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

for f in ("realtime_video.py", "launcher.py"):
    py_compile.compile(os.path.join(HERE, f), doraise=True)
    print(f"compile ok: {f}")


# --- stub the GPU / media stack so the engine module imports headlessly -----
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
launcher_src = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()

ns = {}
block = re.search(r"^T_INDEX_MIN.*?^(?=class )", launcher_src, re.S | re.M).group(0)
exec(compile(block, "launcher", "exec"), ns)
strength_to_t_index = ns["strength_to_t_index"]

# --- every flag the GUI can emit must be one the engine declares ------------
build_cmd = re.search(r"def build_command\(self.*?\n        return cmd", launcher_src, re.S).group(0)
gui_flags = sorted(set(re.findall(r'"(--[a-z_]+)"', build_cmd)))

parser_src = re.search(r"def build_args\(\).*?\n    args, unknown", rv.__file__ and
                       open(os.path.join(HERE, "realtime_video.py"), encoding="utf-8").read(),
                       re.S).group(0)
engine_flags = set(re.findall(r'add_argument\("(--[a-z_]+)"', parser_src))

missing = [f for f in gui_flags if f not in engine_flags]
assert not missing, f"launcher.py emits flags the engine does not declare: {missing}"
print(f"flag handshake ok -> {len(gui_flags)} GUI flags, all declared by the engine")
print(f"  {' '.join(gui_flags)}")

# --- the parser must survive the real command shape -------------------------
sys.argv = ["realtime_video.py",
            "--prompt", "1girl, masterpiece",
            "--negative_prompt", "blurry, deformed",
            "--camera", "0", "--lora", "None",
            "--guidance_scale", "1.600",
            "--t_index", str(strength_to_t_index(40.0)),
            "--freeze_threshold", "1.000",
            "--zmq_port", "51234", "--cmd_port", "51235",
            "--mirror_camera", "--virtual_camera", "--audio_sync",
            "--composite", "--normalize_lighting", "--cuda_graph",
            "--bg_image", "C:/tmp/does_not_exist.png"]
args = rv.build_args()
assert args.audio_sync and args.composite and args.cuda_graph
assert args.frame_buffer == 1, "frame_buffer must stay pinned at 1"
assert 2 <= args.t_index <= 49
assert args.bg_image in ("", None), "a missing bg_image must not be honoured"
print(f"parse ok -> t_index={args.t_index} cfg={args.cfg_type} "
      f"guidance={args.guidance_scale} freeze={args.freeze_threshold}")

# --- an unknown flag must be logged, never fatal ---------------------------
sys.argv = ["realtime_video.py", "--audio_sync", "--some_removed_flag", "--frame_buffer", "2"]
args2 = rv.build_args()
assert args2.audio_sync and args2.frame_buffer == 1
print("unknown flags tolerated ok (argparse must never exit 2 on a mismatch)")

# --- the CFG floor: at exactly 1.0 StreamDiffusion disables CFG entirely ----
floor = re.search(r'"--guidance_scale", f"\{max\(([0-9.]+),', launcher_src)
assert floor and float(floor.group(1)) > 1.0, "launcher must clamp guidance above 1.0"
slider = re.search(r'_slider_row\(1, "Prompt Strictness \(CFG\)", self\.guidance_var, ([0-9.]+)', launcher_src)
assert slider and float(slider.group(1)) > 1.0, "the CFG slider must not reach 1.0"
print(f"CFG floor ok -> clamp {floor.group(1)}, slider min {slider.group(1)}")

# --- AI strength mapping ----------------------------------------------------
for s in (0, 40, 100):
    t = strength_to_t_index(s)
    assert 2 <= t <= 49
    print(f"  AI Strength {s:>3} -> t_index {t}")

print("static checks passed")
