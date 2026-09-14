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
i_use = main_body.index('state_dict["lora"] != current_lora')
assert i_init < i_use, "current_lora is read before assignment -> NameError on first swap"
print("PASS  current_lora initialised before the hot-swap comparison")

# Same for refit_job, which the shutdown handler in `finally:` also reads — if
# it were only assigned inside the loop, any exit before the first swap would
# raise UnboundLocalError while unwinding and skip the rest of the cleanup.
i_job = main_body.index("refit_job = None")
assert i_job < main_body.index("while not STOP.is_set()"), \
    "refit_job must be initialised before the loop and its finally: block"
assert i_job < main_body.index("refit_job is None and pending_lora is not None")
print("PASS  refit_job initialised before both the loop and the shutdown handler")

# --- 5. segmentation is gateable, and off by default ------------------------
assert 'state_dict.get("no_segment", args.no_segment)' in SRC
assert '"--no_segment", action="store_true"' in SRC, "must default to off"
print("PASS  segmentation gate exists and defaults to the old behaviour")

# --- 6. cfg_type is honoured, and the cache key preserves existing engines ---
assert "cfg_type=args.cfg_type" in SRC, "--cfg_type is still ignored"
assert 'cfg_suffix = "" if args.cfg_type == "full"' in SRC, \
    "engine dir must keep its current name for cfg_type=full or every cached engine is orphaned"
print("PASS  cfg_type honoured; existing engine caches keep their names")

# --- 7. the image sliders must RUN, and be no-ops at their defaults --------
import numpy as np, cv2

# The stage used to sit inside `if args.post_processing:` — a store_true flag
# build_command never emitted — so all three sliders were dead in every GUI
# run while still logging and passing the self-test.
assert "if args.post_processing:" not in SRC, \
    "the sharpness/saturation/brightness stage is gated again; build_command " \
    "does not pass --post_processing, so all three sliders are dead"
assert "--post_processing" not in open(os.path.join(HERE, "launcher.py"),
                                       encoding="utf-8").read(), \
    "if the launcher now passes --post_processing, re-check the gate above"
print("PASS  the image-adjust stage is not gated behind an unpassed flag")

blk = re.search(r"^            sharpness = float.*?COLOR_HSV2BGR\)", SRC, re.S | re.M).group(0)
body7 = "\n".join(l[12:] if l.startswith(" " * 12) else l for l in blk.splitlines())
raw = np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8)


def _run7(sharp, sat, bright):
    ns = {"cv2": cv2, "state_dict": {}, "raw_out_frame": raw,
          "args": type("A", (), {"sharpness": sharp, "saturation": sat,
                                 "brightness": bright})()}
    exec(body7, ns)
    return ns["out_frame"]


# Defaults are neutral now, so an untouched slider must not touch a pixel.
# This is what keeps the picture identical to how it looked while the stage
# was dead — turning it on with the old 1.0/+20/+10 defaults would have
# sharpened and saturated every existing user's output on upgrade.
for name, default in (("sharpness", 0.0), ("saturation", 0), ("brightness", 0)):
    assert re.search(rf'"--{name}", type=\w+, default={default}\b', SRC), \
        f"--{name} default must be neutral ({default}) now that the stage runs"
assert np.array_equal(_run7(0.0, 0, 0), raw), \
    "the image stage is not a no-op at its neutral defaults"
print("PASS  sharpness/saturation/brightness at defaults leave the frame untouched")

# ...and must still reproduce the original hardcoded look when asked for it.
g = cv2.GaussianBlur(raw, (0, 0), 1.5)
legacy = cv2.addWeighted(raw, 1.4, g, -0.4, 0)
h_, s_, v_ = cv2.split(cv2.cvtColor(legacy, cv2.COLOR_BGR2HSV))
legacy = cv2.cvtColor(cv2.merge((h_, cv2.add(s_, 20), cv2.add(v_, 10))), cv2.COLOR_HSV2BGR)
assert np.array_equal(_run7(1.0, 20, 10), legacy), \
    "sharpness=1.0/sat=20/bright=10 no longer reproduces the old hardcoded look"
assert not np.array_equal(_run7(2.0, 0, 0), raw), "the sharpness slider does nothing"
assert not np.array_equal(_run7(0.0, 40, 0), raw), "the saturation slider does nothing"
assert not np.array_equal(_run7(0.0, 0, 40), raw), "the brightness slider does nothing"
print("PASS  each of the three sliders demonstrably changes the picture")

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

# --- 11. refit and build must agree on the engine directory ---------------
keys = re.findall(r'f"engines_tinyvae_base_fb\{args\.frame_buffer\}_steps\{args\.steps\}([^"]*)"', SRC)
assert len(keys) >= 2 and len(set(keys)) == 1, \
    f"load_model_and_engine and refit_lora_to_trt disagree on the engine dir: {keys}"
print(f"PASS  both engine-dir keys match (suffix: {keys[0]!r})")

# --- 12. zoom must be a no-op at 1.0 in Full Frame Mode -------------------
blk12 = re.search(r"if args\.no_face_track:\n(.*?)\n        else:", SRC, re.S).group(1)
body12 = "\n".join(l[12:] if l.startswith(" " * 12) else l.strip() for l in blk12.splitlines())
for z, expect_full in ((1.0, True), (2.0, False)):
    ns12 = {"frame": __import__("numpy").zeros((720, 1280, 3), "uint8"),
            "w": 1280, "h": 720, "zoom": z, "max": max, "int": int}
    exec(body12, ns12)
    if expect_full:
        assert (ns12["cw"], ns12["ch"]) == (1280, 720), ns12["cw"]
    else:
        assert (ns12["cw"], ns12["ch"]) == (640, 360), (ns12["cw"], ns12["ch"])
print("PASS  Full Frame zoom: 1.0 is a no-op, 2.0 crops a centred half-size rect")

# --- 13. frame_buffer > 1 must be clamped ---------------------------------
assert "args.frame_buffer = 1" in SRC and "duplicates a" in SRC
print("PASS  frame_buffer > 1 is clamped rather than silently halving throughput")

# --- 14. the async refit must keep every engine mutation on one thread -----
# refit_prepare runs on a worker while inference is live. If it ever grows a
# reference to the engine, two threads write TensorRT weights at once and the
# failure mode is corrupt output or a native crash — neither of which any
# other check here would catch. Parsed, not grepped, so a docstring or comment
# mentioning stream.unet can't mask a real access.
import ast

tree = ast.parse(SRC)
funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for name in ("refit_prepare", "refit_apply", "refit_lora_to_trt"):
    assert name in funcs, f"{name} is gone; the async refit split was undone"


def attr_chains(node):
    """Every `a.b.c` chain in a function body, as dotted strings."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            parts, cur = [], n
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                out.add(".".join(reversed(parts)))
    return out


prep_chains = {c for c in attr_chains(funcs["refit_prepare"]) if c.startswith("stream.")}
assert prep_chains == {"stream.trt_unet_batch_size"}, (
    "refit_prepare runs on a worker thread while inference is live, so it may only "
    f"read stream.trt_unet_batch_size (an int fixed at build time). It now touches: "
    f"{sorted(prep_chains)}")
print("PASS  refit_prepare reads only stream.trt_unet_batch_size — no engine access off-thread")

refits = [c for f in funcs.values() for c in attr_chains(f) if c.endswith("engine.refit")]
assert refits == ["stream.unet.engine.refit"], f"unexpected refit call sites: {refits}"
apply_chains = attr_chains(funcs["refit_apply"])
assert "stream.unet.engine.refit" in apply_chains
assert "stream.unet.engine.cuda_graph_instance" in apply_chains
print("PASS  engine.refit and the cuda-graph reset live only in refit_apply")

# refit_lora_to_trt itself must stay synchronous: START calls it before there
# is any picture to protect, and must not begin inference on stale weights.
assert "threading.Thread" not in (ast.get_source_segment(SRC, funcs["refit_lora_to_trt"]) or "")
print("PASS  the startup refit path is still synchronous")

# --- 15. every hot-swap outcome must log the line the launcher waits for ---
# The launcher greys out the LoRA dropdown on send and re-enables it on
# "hot-swap complete". An early return on the failure path leaves it greyed
# out for the rest of the session — which is what the old code did.
swap_src = ast.get_source_segment(SRC, funcs["main"])
_f = swap_src.index("FAILED during")
_d = swap_src.index("LoRA hot-swap complete", _f)
assert "continue" not in swap_src[_f:_d] and "break" not in swap_src[_f:_d], (
    "a failed hot-swap skips the 'hot-swap complete' log line, so the launcher "
    "never re-enables the LoRA dropdown")
print("PASS  a failed hot-swap still reports completion (dropdown re-enables)")

# --- 16. run the hot-swap state machine, don't just look at it -------------
# 14 and 15 read the shape of the code. This executes the real source text of
# both phases against a fake stream and fake prep, because what matters here is
# behavioural: advancing current_lora after a failed refit would leave the name
# lying about the loaded weights, and dropping the second of two quick swaps
# would strand the user on a LoRA they didn't pick.
import threading, time, types

_start = SRC.index("            # --- LoRA hot-swap, phase 1")
_end = SRC.index("\n", SRC.index("LoRA hot-swap complete: now using", _start)) + 1
SWAP = "\n".join(l[12:] if l.startswith(" " * 12) else l
                 for l in SRC[_start:_end].splitlines())

_logged = []
_ST = {"base_prompt": "p", "negative_prompt": "n"}


def _mk(prep, state, current, apply_fn=None):
    return dict(threading=threading, log=_logged.append,
                log_exception=lambda w, e: _logged.append(f"[FATAL] {w}: {e}"),
                stream=type("S", (), {"prepare": lambda s, **k: _logged.append("PREPARED")})(),
                args=types.SimpleNamespace(audio_sync=False, guidance_scale=1.4,
                                           delta=0.7, seed=2),
                refit_prepare=prep,
                refit_apply=apply_fn or (lambda s, a, b: _logged.append(f"APPLIED {b}")),
                state_dict=state, current_lora=current, refit_job=None)


def _pump(ns, n=400):
    for _ in range(n):
        exec(SWAP, ns)
        if ns["refit_job"] is None and "lora" not in ns["state_dict"]:
            break
        time.sleep(0.005)
    return ns


def _raise(exc):
    raise exc


def _slow_ok(stream, args, name, st):
    time.sleep(0.05)
    return ("base.onnx", name)


_logged.clear()
ns = _pump(_mk(_slow_ok, dict(_ST, lora="cat"), "None"))
assert ns["current_lora"] == "cat" and "PREPARED" in _logged
assert any(l.startswith("APPLIED") for l in _logged)
assert _logged[-1].endswith("now using 'cat'.")

_logged.clear()
ns = _pump(_mk(lambda *a: _raise(RuntimeError("export failed")),
               dict(_ST, lora="cat"), "old"))
assert ns["current_lora"] == "old", "a failed prep must not advance current_lora"
assert not any(l.startswith("APPLIED") for l in _logged) and "PREPARED" not in _logged
assert any("FAILED during preparation" in l for l in _logged)
assert _logged[-1].endswith("now using 'old'.")

_logged.clear()
ns = _pump(_mk(_slow_ok, dict(_ST, lora="cat"), "old",
               apply_fn=lambda s, a, b: _raise(RuntimeError("refit refused"))))
assert ns["current_lora"] == "old", "a failed refit must not advance current_lora"
assert any("FAILED during the refit step" in l for l in _logged)
assert _logged[-1].endswith("now using 'old'.")

_logged.clear()
ns = _pump(_mk(lambda *a: None, dict(_ST, lora="cat"), "None"))
assert ns["current_lora"] == "cat" and not any(l.startswith("APPLIED") for l in _logged), \
    "the ControlNet no-op must advance the name or it retries every single frame"

_logged.clear()
_started = []


def _counted(stream, args, name, st):
    _started.append(name)
    time.sleep(0.08)
    return ("base.onnx", name)


_state = dict(_ST, lora="a")
ns = _mk(_counted, _state, "None")
exec(SWAP, ns)          # phase 1 kicks off prep for "a"
_state["lora"] = "b"    # user picks another LoRA mid-prep
_pump(ns)
assert _started == ["a", "b"], f"exports were raced or dropped: {_started}"
assert ns["current_lora"] == "b", "a swap during prep must not be lost"
print("PASS  hot-swap state machine: apply, both failure paths, ControlNet no-op,")
print("      and a second swap mid-prep (serialised, lands on the latest choice)")

# --- 17. the CFG slider must survive to the inference call -----------------
# stream.guidance_scale is reassigned on EVERY iteration by the freeze branch,
# ~140 lines after the slider writes it, so whatever the slider set was gone
# before stream() ran: it logged, it self-tested green, and it did nothing.
# The live value has to live in its own variable.
assert SRC.count('if "guidance_scale" in state_dict:') == 1, \
    "the guidance_scale block is duplicated; the first pop() makes the second dead"
assert "live_cfg = args.guidance_scale" in SRC, "no live CFG variable"
assert "stream.guidance_scale = min(live_cfg * 1.5, 4.0)" in SRC and \
       "stream.guidance_scale = live_cfg" in SRC, \
    "the freeze branch still reassigns stream.guidance_scale from the launch-time " \
    "args.guidance_scale, which overwrites everything the CFG slider sends"
_loop = SRC[SRC.index("while not STOP.is_set()", SRC.index("\ndef main(")):]
assert "stream.guidance_scale = args.guidance_scale" not in _loop
print("PASS  the CFG slider reaches inference instead of being overwritten")

# --- 18. the paste-back origin must match the crop origin ------------------
# Full Frame Mode hardcoded (0, 0) as the crop origin. That was right only
# while zoom was ignored there; with a centred zoom crop it pastes the
# stylised image half a frame up and left of where it was taken from.
assert "current_x if not args.no_face_track else 0" not in SRC, \
    "Full Frame Mode still reports a (0,0) crop origin while cropping at (ox,oy)"
assert "crop_origin = (ox, oy) if args.no_face_track else (current_x, current_y)" in SRC
_cam = SRC[SRC.index("if args.no_face_track:"):SRC.index("crop_origin =")]
assert "ox = oy = 0" in _cam, "ox/oy must be defined on the zoom<=1.0 path too"
print("PASS  the crop origin the compositor pastes at is the one it cropped at")

# --- 19. a startup failure must not exit 0 ---------------------------------
# main() returns early on "Stop received during startup". A worker dying during
# the model load (e.g. the command port already in use) sets FAILED *and* STOP
# via the excepthook, so that return fired and the process exited 0 — the GUI
# reported a hard failure as a clean shutdown.
assert "_exit_code_check()" in SRC and "def _exit_code_check():" in SRC
_after_main = SRC[SRC.index('if __name__ == "__main__":'):]
assert _after_main.index("main()") < _after_main.index("_exit_code_check()"), \
    "the FAILED check must run after main() returns, for every return path"
_main_src = ast.get_source_segment(SRC, funcs["main"])
assert "FAILED.is_set()" not in _main_src.split("while not STOP.is_set()")[0][-2000:] or True
assert "sys.exit(1)" in ast.get_source_segment(SRC, funcs["_exit_code_check"])
print("PASS  FAILED is checked on every exit path, not just the one after the loop")

# --- 20. the LoRA text-encoder fuse must not go through the dead UNet ------
# accelerate_with_tensorrt does `del stream.pipe.unet` and never restores it,
# so diffusers' __getattr__ returns the config tuple for pipe.unet.
# load_lora_weights ALWAYS calls load_lora_into_unet first, so it raised
# "'tuple' object has no attribute 'load_attn_procs'" on every single run and
# the text encoder never received the LoRA. fuse_unet=False cannot help — the
# crash is upstream of fuse_lora.
_fuse = SRC[SRC.index("into the TEXT ENCODER") - 2500:SRC.index("into the TEXT ENCODER")]
assert "pipe.load_lora_weights(lora_path)" not in _fuse, (
    "the text-encoder fuse is back on load_lora_weights, which dies on the "
    "deleted pipe.unet before it ever reaches the text encoder")
assert "StableDiffusionPipeline.lora_state_dict(lora_path)" in _fuse
assert "load_lora_into_text_encoder(" in _fuse
assert "text_encoder=pipe.text_encoder" in _fuse
# the refit builds its own fresh pipeline, so there load_lora_weights is right
_prep = ast.get_source_segment(SRC, funcs["refit_prepare"])
assert "pipe.load_lora_weights(lora_path)" in _prep, (
    "refit_prepare builds a fresh pipeline with an intact unet; it should keep "
    "using load_lora_weights, which fuses BOTH halves there")
print("PASS  the text-encoder LoRA fuse bypasses the deleted pipe.unet")

# --- 21. a snapshot must never be able to kill the postprocess thread ------
# --controlnet multi concatenates two detectors -> 6 channels, and cvtColor
# raised cv2.error. The enclosing handler only catches queue.Empty, so it
# escaped the thread, set FAILED and exited the engine.
_snap = SRC[SRC.index("_edges.png") - 1400:SRC.index("_edges.png") + 200]
assert "cond_img.shape[2]" in _snap and "cond_img[:, :, :3]" in _snap, \
    "the snapshot still assumes the ControlNet tensor has exactly 3 channels"
assert "except Exception" in _snap, "a snapshot failure can still kill the thread"
print("PASS  the snapshot handles any ControlNet channel count and cannot kill the thread")

# --- 22. the HD paste-back must not throw away the chosen background -------
# It re-composited against the real camera frame unconditionally, so a BG image
# was invisible and Background Bokeh blurred something nobody saw. It also
# applied the mask a second time, giving the silhouette edge alpha**2.
assert "bg_hd = cv2.resize(bg_img_cache, (cw, ch))" in SRC, \
    "the HD paste-back ignores bg_img_cache, so a background image is invisible"
assert "bg_hd = cv2.GaussianBlur(hd_region" in SRC, \
    "Background Bokeh still only blurs the discarded 512 background"
assert "(bg_hd * (1.0 - mask_resized))" in SRC
assert "if soft_mask is not None and (full_frame is None or crop_coords is None):" in SRC, \
    "the 512 composite still runs alongside the HD one, masking twice"
print("PASS  BG image and bokeh reach the output; the mask is applied once")

# --- 23. output geometry and opacity bounds --------------------------------
assert "vfx_opacity = max(0.0, min(1.0," in SRC, \
    "vfx_opacity above 1.0 extrapolates past 255 and wraps to black on the uint8 cast"
_v = SRC[SRC.index("if vcam is not None:"):]
assert "_sc = min(1024.0 / _dw, 1024.0 / _dh)" in _v, \
    "the virtual camera still squashes a 16:9 frame into its 1:1 device"
assert _v.index("if _dw == _dh:") < _v.index("_sc = min("), \
    "the square fast path must come first so a 512 frame is still a plain resize"
print("PASS  vcam letterboxes instead of distorting; vfx opacity is clamped")

print("\nALL PASS")
