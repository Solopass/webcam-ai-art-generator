"""Headless checks on the engine-side logic changes. No GPU, no imports of torch."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "realtime_video.py"), encoding="utf-8").read()

# --- 1. the 4-step schedule -------------------------------------------------
block = re.search(r"^    if args\.steps == 4:\n(.*?)^    else:", SRC, re.S | re.M).group(1)
body = "\n".join(l[8:] if l.startswith(" " * 8) else l.strip()
                 for l in block.splitlines())


def new_t_list(t_index):
    ns = {"args": type("A", (), {"t_index": t_index, "steps": 4})()}
    exec(body, ns)
    return ns["t_list"]


def old_t_list(t):
    return [max(0, t - 30), max(0, t - 20), max(0, t - 10), t]


# strictly ascending everywhere the engine can be asked for
for t in range(2, 50):
    lst = new_t_list(t)
    assert len(lst) == 4, (t, lst)
    assert all(lst[i] < lst[i + 1] for i in range(3)), f"t_index={t} -> {lst} not ascending"
    assert lst[0] >= 0
print("PASS  4-step schedule strictly ascending for every t_index 2-49")

# unchanged wherever the old formula wasn't already clamping
same = [t for t in range(32, 50) if new_t_list(t) == old_t_list(t)]
assert len(same) == 18, [(t, old_t_list(t), new_t_list(t)) for t in range(32, 50)
                         if new_t_list(t) != old_t_list(t)]
print(f"PASS  identical to the old schedule for all t_index 32-49 "
      f"(e.g. 45 -> {new_t_list(45)})")

# and fixed where it was broken
assert old_t_list(14) == [0, 0, 4, 14] and new_t_list(14) != old_t_list(14)
print(f"PASS  the degenerate case is gone: t_index 14 was {old_t_list(14)}, now {new_t_list(14)}")

# --- 2. every command the launcher sends now has a handler ------------------
launcher_src = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()
sent = set(re.findall(r'send_command\(\{"([a-z_]+)"', launcher_src))
sent |= set(re.findall(r'dumps\(\{"([a-z_]+)"', launcher_src))
# Sliders send through a helper with the key as an argument, so the literal
# never appears next to send_command. Without this the test silently ignored
# every one of them on BOTH sides and reported full coverage.
sent |= set(re.findall(r'_live\("([a-z_]+)"', launcher_src))
sent |= set(re.findall(r'\{"([a-z_]+)": [a-z]+\(self\.[a-z_]+\.get\(\)\)\}', launcher_src))
sent.discard("cmd")

listener = re.search(r"def cmd_listener_thread.*?\n    finally:", SRC, re.S).group(0)
handled = set(re.findall(r'"([a-z_]+)" in cmd', listener))
# The engine also handles a batch of keys via a (key, label) tuple loop.
handled |= set(re.findall(r'\("([a-z_]+)", "[A-Za-z ]+"\)', listener))

missing = sorted(sent - handled)
assert not missing, f"launcher sends commands the engine drops: {missing}"
print(f"PASS  all {len(sent)} launcher commands have an engine handler")
print(f"      {' '.join(sorted(sent))}")

# --- 3. the command socket must bind before the model load ------------------
main_body = SRC[SRC.index("\ndef main("):]
i_listen = main_body.index("cmd_listener_thread, args=")
i_load = main_body.index("load_model_and_engine(args")
assert i_listen < i_load, ("the command socket still binds after the model load, so Stop "
                           "cannot work during a 5-15 minute engine build")
print("PASS  command socket binds before the model load (Stop works during a build)")

# --- 4. current_lora is initialised before it is read -----------------------
i_init = main_body.index("current_lora = args.lora")
i_use = main_body.index('if "lora" in state_dict and state_dict["lora"] != current_lora')
assert i_init < i_use, "current_lora is read before assignment -> NameError on first swap"
print("PASS  current_lora initialised before the hot-swap comparison")

# --- 5. segmentation is gateable, and off by default ------------------------
assert 'state_dict.get("no_segment", args.no_segment)' in SRC
assert '"--no_segment", action="store_true"' in SRC, "must default to off"
print("PASS  segmentation gate exists and defaults to the old behaviour")

# --- 6. cfg_type is honoured, and the cache key preserves existing engines ---
assert "cfg_type=args.cfg_type" in SRC, "--cfg_type is still ignored"
assert 'cfg_suffix = "" if args.cfg_type == "full"' in SRC, \
    "engine dir must keep its current name for cfg_type=full or every cached engine is orphaned"
print("PASS  cfg_type honoured; existing engine caches keep their names")

# --- 7. the new image sliders must be no-ops at their defaults -------------
import numpy as np, cv2
blk = re.search(r"                sharpness = float.*?COLOR_HSV2BGR\)", SRC, re.S).group(0)
body7 = "\n".join(l[16:] if l.startswith(" " * 16) else l for l in blk.splitlines())
raw = np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8)
g = cv2.GaussianBlur(raw, (0, 0), 1.5)
old = cv2.addWeighted(raw, 1.4, g, -0.4, 0)
h_, s_, v_ = cv2.split(cv2.cvtColor(old, cv2.COLOR_BGR2HSV))
old = cv2.cvtColor(cv2.merge((h_, cv2.add(s_, 20), cv2.add(v_, 10))), cv2.COLOR_HSV2BGR)
ns7 = {"cv2": cv2, "state_dict": {}, "raw_out_frame": raw,
       "args": type("A", (), {"sharpness": 1.0, "saturation": 20, "brightness": 10})()}
exec(body7, ns7)
assert np.array_equal(ns7["out_frame"], old), \
    "the image sliders changed the picture at their default values"
print("PASS  sharpness/saturation/brightness at defaults are byte-identical to the old path")

# --- 8. the live freeze value must survive to its consumer ----------------
assert 'val = state_dict.pop("freeze_threshold_dirty")' not in SRC, \
    "the freeze value is popped before its only consumer reads it - Freeze and Stillness Blend both die"
assert 'state_dict.get("freeze_threshold_dirty", args.freeze_threshold)' in SRC
print("PASS  live freeze threshold is not discarded before it is read")

# --- 9. idle postprocess iterations must not re-composite in place ---------
assert SRC.count("pristine_full_frame") >= 4, \
    "postprocess re-composites the same buffer on idle iterations; blend modes compound"
print("PASS  idle iterations re-composite from a pristine copy")

# --- 10. every blend mode the GUI offers must be implemented ---------------
gui = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()
offered = set(re.findall(r'"([A-Z][A-Za-z ()]+)"',
                         re.search(r'values=\["Normal".*?\]', gui, re.S).group(0)))
impl = set(re.findall(r'vfx_blend_mode == "([^"]+)"', SRC)) | {"Normal"}
assert offered <= impl, f"GUI offers blend modes the engine ignores: {sorted(offered - impl)}"
print(f"PASS  all {len(offered)} offered blend modes are implemented")

print("\nALL PASS")
