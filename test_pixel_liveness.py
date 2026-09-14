"""Pixel-level liveness checks: does each control actually change the picture?

Every other check in this repo verifies *shape* — that a command has a handler,
that a value reaches state_dict, that the engine echoes it back. Eight controls
have shipped dead anyway (CFG, sharpness, saturation, brightness, freeze
threshold, stillness blend, background image, bokeh). Each one logged its new
value, each one passed the GUI self-test, and none of them touched a pixel.

The lesson from HANDOFF §3 is that an echo proves arrival, not effect. So this
file proves effect: it lifts the engine's real image-processing source out of
realtime_video.py, runs it on a synthetic frame at two different values of each
control, and asserts the output bytes differ. A control that cannot change a
pixel fails here.

No GPU and no torch — these stages are pure cv2/numpy, which is exactly why
they are worth testing this way.

Two distinct failure messages matter when this file goes red:
  * "EXTRACTION FAILED" — the source moved. Fix the extractor below; the
    control may well be fine.
  * "is DEAD" — the block was found and executed, and the control changed
    nothing. That is a real defect.

Known limit, worth saying out loud: this runs each stage in ISOLATION, so it
proves the code does something, not that the engine reaches it. A control
killed by an enclosing `if` that is never true — which is exactly how
Sharpness, Saturation and Brightness died inside `if args.post_processing:` —
shows up here as EXTRACTION FAILED, not as "is DEAD". The reachability half
lives in test_engine_logic.py; the two files are complements, and run_checks.py
runs both. Mutation-tested: reintroducing each of the bugs this was built for
turns the suite red.
"""
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "realtime_video.py"), encoding="utf-8").read()

FAILURES = []


def extract(name, pattern, dedent):
    """Pull one contiguous block of engine source out and dedent it to run."""
    m = re.search(pattern, SRC, re.S | re.M)
    if not m:
        raise SystemExit(
            f"EXTRACTION FAILED for {name!r}: the source block no longer matches\n"
            f"  {pattern}\n"
            f"This is a test-maintenance problem, not necessarily a bug in the\n"
            f"engine. Re-anchor the pattern on the current source and re-run.")
    block = m.group(0)
    return "\n".join(l[dedent:] if l.startswith(" " * dedent) else l
                     for l in block.splitlines())


def frame(seed=7, h=64, w=64):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def check(control, a, b, what=""):
    """a and b are renders at two different values of `control`."""
    if np.array_equal(a, b):
        FAILURES.append(f"{control} is DEAD — the picture is byte-identical "
                        f"at both values{(' (' + what + ')') if what else ''}")
        print(f"FAIL  {control} changed nothing")
        return False
    print(f"PASS  {control} changes the picture")
    return True


def check_noop(control, out, base):
    if not np.array_equal(out, base):
        FAILURES.append(f"{control} is not a no-op at its default value — "
                        f"upgrading changes everyone's picture")
        print(f"FAIL  {control} alters the frame at its default")
        return False
    print(f"PASS  {control} at its default leaves the frame untouched")
    return True


# =====================================================================
# Stage 1 — the image adjust stage (sharpness / saturation / brightness)
# =====================================================================
ADJUST = extract("image adjust",
                 r"^            sharpness = float.*?COLOR_HSV2BGR\)", 12)


def render_adjust(sharpness=0.0, saturation=0, brightness=0, src=None):
    ns = {"cv2": cv2, "state_dict": {},
          "raw_out_frame": frame() if src is None else src,
          "args": type("A", (), {"sharpness": sharpness,
                                 "saturation": saturation,
                                 "brightness": brightness})()}
    exec(ADJUST, ns)
    return ns["out_frame"]


print("--- image adjust ---")
base = frame()
check_noop("Sharpness/Saturation/Brightness", render_adjust(), base)
check("Sharpness", render_adjust(sharpness=0.0), render_adjust(sharpness=2.0))
check("Saturation", render_adjust(saturation=0), render_adjust(saturation=40))
check("Brightness", render_adjust(brightness=0), render_adjust(brightness=40))
# and negative directions, which take a different cv2 branch (subtract, not add)
check("Saturation (negative)", render_adjust(saturation=0), render_adjust(saturation=-40))
check("Brightness (negative)", render_adjust(brightness=0), render_adjust(brightness=-40))
check("Sharpness (blur direction)", render_adjust(sharpness=0.0), render_adjust(sharpness=-2.0))

# =====================================================================
# Stage 2 — the HD paste-back: blend mode, VFX opacity, mask, bokeh, BG image
# =====================================================================
PASTE = extract("HD paste-back",
                r"^                ai_resized = cv2\.resize.*?"
                r"full_frame\[cy:cy\+ch, cx:cx\+cw\] = final_ai\.astype\(np\.uint8\)", 16)

CW = CH = 64


def render_paste(blend="Normal", opacity=1.0, bokeh=0.0, bg_image=False,
                 mask=True, mask_value=0.5):
    """Run the engine's real paste-back over a synthetic AI frame + HD frame."""
    # Textured, not flat: a Gaussian blur of a uniform frame is the identity,
    # which would make a dead Bokeh slider look alive (or vice versa).
    full = frame(seed=23, h=CH, w=CW)
    ai = frame(seed=11, h=CH, w=CW)
    soft = None
    if mask:
        soft = np.full((CH, CW), np.float32(mask_value), dtype=np.float32)
        # a gradient, so feathering and double-masking are both visible
        soft[:, : CW // 2] = np.float32(mask_value * 0.5)
    ns = {"cv2": cv2, "np": np,
          "display_frame": ai, "full_frame": full,
          "cx": 0, "cy": 0, "cw": CW, "ch": CH,
          "soft_mask": soft,
          "vfx_blend_mode": blend,
          "vfx_opacity": max(0.0, min(1.0, opacity)),
          "bg_img_cache": frame(seed=99, h=512, w=512) if bg_image else None,
          "state_dict": {"bokeh_blur": bokeh},
          "args": type("A", (), {"bokeh_blur": 0.0})()}
    exec(PASTE, ns)
    return ns["full_frame"].copy()


# The clamp lives outside the paste-back block, so drive it separately —
# passing an already-clamped value in (as this harness used to) means the
# engine's own clamp is never exercised and could be deleted unnoticed.
CLAMP = extract("vfx opacity clamp", r"^        vfx_opacity = [^\n]*$", 8)


def clamped_opacity(v):
    ns = {"state_dict": {"vfx_opacity": v}, "max": max, "min": min, "float": float}
    exec(CLAMP, ns)
    return ns["vfx_opacity"]


print("\n--- vfx opacity bounds ---")
for raw_v, want in ((1.6, 1.0), (-0.5, 0.0), (0.4, 0.4)):
    got = clamped_opacity(raw_v)
    if abs(got - want) > 1e-6:
        FAILURES.append(f"vfx_opacity {raw_v} was not clamped to {want} (got {got}); "
                        f"above 1.0 the blend extrapolates past 255 and the uint8 "
                        f"cast wraps highlights to black")
        print(f"FAIL  vfx_opacity {raw_v} -> {got}, expected {want}")
    else:
        print(f"PASS  vfx_opacity {raw_v} -> {got}")

print("\n--- HD paste-back ---")
check("VFX opacity", render_paste(opacity=1.0), render_paste(opacity=0.2))
check("Background Bokeh", render_paste(bokeh=0.0), render_paste(bokeh=1.0),
      "it blurred the 512 background, which the HD paste-back then discarded")
check("Background image", render_paste(bg_image=False), render_paste(bg_image=True),
      "the HD paste-back re-composited the real camera frame over it")
check("Mask Feather / segmentation", render_paste(mask_value=0.2),
      render_paste(mask_value=0.9))

# Every blend mode the GUI offers must produce a DISTINCT picture. Eight of
# them once fell through to Normal while the listener logged their name, so
# "it is implemented" is not the same as "it does something different".
gui = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()
modes = re.findall(r'"([A-Z][A-Za-z ()]+)"',
                   re.search(r'values=\["Normal".*?\]', gui, re.S).group(0))
normal = render_paste(blend="Normal", opacity=0.8)
seen = {}
for mode in modes:
    out = render_paste(blend=mode, opacity=0.8)
    if mode != "Normal" and np.array_equal(out, normal):
        FAILURES.append(f"blend mode {mode!r} is DEAD — identical to Normal, "
                        f"so it is falling through the if/elif chain")
        print(f"FAIL  blend mode {mode} is indistinguishable from Normal")
        continue
    dup = next((m for m, o in seen.items() if np.array_equal(o, out)), None)
    if dup:
        FAILURES.append(f"blend modes {mode!r} and {dup!r} produce identical "
                        f"output — one of them is wired to the wrong formula")
        print(f"FAIL  blend mode {mode} is identical to {dup}")
        continue
    seen[mode] = out
print(f"PASS  all {len(modes)} blend modes produce distinct pictures")

# No blend mode may wrap through the uint8 cast. Checking "bright in, bright
# out" would be wrong — Difference and Exclusion are legitimately dark on
# white-on-white — so assert the real invariant on the pre-cast float: every
# value must already be inside [0, 255] before .astype(np.uint8) truncates it.
for mode in modes:
    for a_val, b_val in ((250, 250), (250, 5), (5, 250), (5, 5), (128, 200)):
        ns = {"cv2": cv2, "np": np,
              "display_frame": np.full((CH, CW, 3), a_val, dtype=np.uint8),
              "full_frame": np.full((CH, CW, 3), b_val, dtype=np.uint8),
              "cx": 0, "cy": 0, "cw": CW, "ch": CH, "soft_mask": None,
              "vfx_blend_mode": mode, "vfx_opacity": 1.0, "bg_img_cache": None,
              "state_dict": {}, "args": type("A", (), {"bokeh_blur": 0.0})()}
        exec(PASTE, ns)
        f_ = ns["final_ai"]
        if f_.min() < -0.5 or f_.max() > 255.5:
            FAILURES.append(
                f"blend mode {mode!r} produced {f_.min():.1f}..{f_.max():.1f} "
                f"for AI={a_val} over base={b_val}; the uint8 cast wraps it")
            print(f"FAIL  blend mode {mode} leaves [0,255] "
                  f"({f_.min():.1f}..{f_.max():.1f}) at AI={a_val} base={b_val}")
            break
else:
    print(f"PASS  no blend mode leaves [0,255] before the uint8 cast")

# The mask must be applied exactly once. Applying it twice (which is what the
# 512 composite running alongside the HD one did) squares alpha, so a 0.5 mask
# contributes 0.25 of the AI frame and the feathered edge reads thin.
m = 0.5
ai = frame(seed=11, h=CH, w=CW).astype(np.float32)
bgv = frame(seed=23, h=CH, w=CW).astype(np.float32)
once = ai * m + bgv * (1 - m)
got = render_paste(mask=True, mask_value=m)[:, CW // 2:].astype(np.float32)
if not np.allclose(got, once[:, CW // 2:], atol=1.5):
    FAILURES.append("the segmentation mask is not applied exactly once — the "
                    "feathered edge will not match the Mask Feather setting")
    print("FAIL  mask is applied more than once (alpha is squared at the edge)")
else:
    print("PASS  the segmentation mask is applied exactly once")

# =====================================================================
# Stage 3 — camera crop geometry
# =====================================================================
CROP = extract("full-frame crop",
               r"^            if zoom > 1\.001:.*?^                cropped = frame$", 12)


def render_crop(zoom):
    ns = {"frame": frame(seed=5, h=720, w=1280), "w": 1280, "h": 720,
          "zoom": zoom, "max": max, "int": int}
    exec(CROP, ns)
    return ns["cropped"], (ns["ox"], ns["oy"]), (ns["cw"], ns["ch"])


print("\n--- camera crop ---")
c1, o1, s1 = render_crop(1.0)
c2, o2, s2 = render_crop(2.0)
if s1 != (1280, 720) or o1 != (0, 0):
    FAILURES.append(f"Full Frame at zoom 1.0 is no longer a no-op: {o1} {s1}")
    print("FAIL  zoom 1.0 is not a plain full frame")
else:
    print("PASS  Camera Zoom at 1.0 is a byte-identical no-op")
if s2 == s1:
    FAILURES.append("Camera Zoom is DEAD in Full Frame Mode — the crop is the "
                    "same size at 1.0 and 2.0")
    print("FAIL  Camera Zoom changes nothing in Full Frame Mode")
else:
    print("PASS  Camera Zoom crops in Full Frame Mode")
# the origin must be the centre of the crop, or the paste-back lands offset
if o2 != ((1280 - s2[0]) // 2, (720 - s2[1]) // 2):
    FAILURES.append(f"the zoom crop origin {o2} is not centred, so the "
                    f"paste-back will be offset from where it was captured")
    print("FAIL  the zoomed crop origin is not centred")
else:
    print("PASS  the zoomed crop origin is centred (paste-back lines up)")

# =====================================================================
print()
if FAILURES:
    print("=" * 70)
    for f in FAILURES:
        print("  " + f)
    raise SystemExit(f"\n{len(FAILURES)} control(s) failed a pixel-level check.")
print("ALL PASS")
