"""Headless checks for the preset + history layer.

customtkinter/tkinter can't run without a display here, so this drives the new
pure-logic methods on a stand-in object holding the same widget attributes.
"""
import importlib.util
import json
import os
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))

# --- stub the GUI toolkit so launcher.py imports ---------------------------
class _Any:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, n):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


class _Stub(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        # a real class, so `class VTuberStudioApp(ctk.CTk)` works
        return type(n, (_Any,), {})


for name in ("customtkinter", "tkinter", "tkinter.filedialog", "zmq", "cv2",
             "numpy", "PIL", "PIL.Image", "PIL.ImageTk", "imageio"):
    sys.modules.setdefault(name, _Stub(name))

spec = importlib.util.spec_from_file_location("launcher", os.path.join(HERE, "launcher.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
App = launcher.VTuberStudioApp
print("launcher imported ok")

tmp = tempfile.mkdtemp()
launcher.PRESETS_FILE = os.path.join(tmp, "presets.json")
launcher.HISTORY_FILE = os.path.join(tmp, "history.json")
launcher.HISTORY_LIMIT = 5


# --- a fake app exposing the same widget surface ---------------------------
class Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class Entry:
    def __init__(self, v=""):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v

    def delete(self, *a):
        self.v = ""

    def insert(self, _i, v):
        self.v = v


class Menu:
    def __init__(self):
        self.values = []

    def configure(self, values=None, **k):
        if values is not None:
            self.values = values


class FakeApp(App):
    def __init__(self):
        self.logs = []
        self.process = None
        self._slider_rows = []
        self._history_entries = []
        self.prompt_entry = Entry("cat girl")
        self.neg_prompt_entry = Entry("blurry")
        self.camera_entry = Entry("0")
        for attr, val in [
            ("preview_var", True), ("mirror_var", True), ("vcam_var", False),
            ("no_face_track_var", False), ("audio_var", False),
            ("bg_keep_var", False), ("clahe_var", True), ("cudagraph_var", False),
            ("lora_var", "None (Original Default)"), ("controlnet_var", "None"),
            ("perf_var", "Maximum Speed (Low Quality)"), ("easyneg_var", True),
            ("no_segment_var", False), ("sharp_var", 1.0), ("sat_var", 20.0),
            ("bright_var", 10.0), ("feather_var", 7.0), ("denoise_var", 0.4),
            ("stillness_var", 0.3), ("vfx_op_var", 1.0), ("vfx_blend_var", "Normal"),
            ("seed_var", 2), ("lora_strength_var", 1.0),
            ("bg_var", ""), ("guidance_var", 2.3175), ("strength_var", 25.0),
            ("freeze_var", 0.99), ("motion_var", 0.67), ("bokeh_var", 0.0),
            ("zoom_var", 1.9),
        ]:
            setattr(self, attr, Var(val))
        self.expr_vars = {"smiling": Var("smiling"), "open mouth": Var("open mouth")}
        self.sens_vars = {"smile": Var(0.001), "mouth": Var(0.02)}
        self.lora_dropdown = Menu()
        self.perf_dropdown = Menu()
        self.controlnet_dropdown = Menu()
        self._slider_widgets = {}
        self.preset_var = Var("")
        self.preset_menu = Menu()
        self.history_var = Var("")
        self.history_menu = Menu()

    def log(self, m):
        self.logs.append(m)

    def toggle_preview(self):
        pass

    def apply_prompt(self, quiet=False):
        self.logs.append("apply_prompt")

    def send_command(self, payload):
        self.logs.append(("cmd", payload))
        return True


app = FakeApp()

# --- 1. every collected key must be applicable ------------------------------
collected = app._collect_settings()
unhandled = []
for key in collected:
    if key in ("prompt", "negative_prompt", "camera"):
        continue
    if key.startswith("expr_") or key.startswith("sens_"):
        continue
    if app._settings_var(key) is None:
        unhandled.append(key)
assert not unhandled, f"_collect_settings emits keys _apply_settings_dict cannot set: {unhandled}"
print(f"PASS  round-trip coverage: all {len(collected)} saved keys are applicable")

# --- 2. built-in preset keys must all be real settings ----------------------
for pname, values in launcher.BUILTIN_PRESETS.items():
    bad = [k for k in values
           if k not in ("prompt", "negative_prompt", "camera")
           and not k.startswith(("expr_", "sens_"))
           and app._settings_var(k) is None]
    assert not bad, f"built-in preset {pname!r} references unknown keys: {bad}"
print(f"PASS  all {len(launcher.BUILTIN_PRESETS)} built-in presets reference real settings")

# --- 3. built-in perf_mode strings must match the dropdown ------------------
src = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()
perf_block = re.search(r"perf_modes = \[(.*?)\]", src, re.S).group(1)
valid_perf = set(re.findall(r'"([^"]+)"', perf_block))
for pname, values in launcher.BUILTIN_PRESETS.items():
    pm = values.get("perf_mode")
    assert pm in valid_perf, f"{pname!r} perf_mode {pm!r} not in dropdown {valid_perf}"
print("PASS  built-in perf_mode values all exist in the dropdown")

# --- 4. Flat 2D preset actually undoes the diagnosed causes -----------------
flat = launcher.BUILTIN_PRESETS["★ Flat 2D Anime"]
t_idx = launcher.strength_to_t_index(flat["ai_strength"])
assert t_idx <= 20, f"Flat 2D t_index {t_idx} is still too high to repaint"
assert flat["normalize_lighting"] is False
assert flat["freeze"] >= 0.999, "freeze must be off so the webcam refinement blend can't run"
assert "3d" in flat["negative_prompt"]
print(f"PASS  Flat 2D preset -> t_index {t_idx}, CLAHE off, freeze off, anti-3d negatives")
print(f"      (current settings map to t_index {launcher.strength_to_t_index(25.0)})")

# --- 5. preset save / load / delete ----------------------------------------
app.prompt_entry.set("my custom look")
app.strength_var.set(77.0)
saved = app._collect_settings()
launcher.VTuberStudioApp._write_json(app, launcher.PRESETS_FILE, {"Mine": saved}, "T")
assert "Mine" in app.preset_names()
app.prompt_entry.set("something else")
app.strength_var.set(10.0)
app.preset_var.set("Mine")
app.on_preset_load()
assert app.prompt_entry.get() == "my custom look"
assert app.strength_var.get() == 77.0
print("PASS  preset save -> mutate -> load restores values")

# --- 6. loading a partial preset leaves unrelated settings alone ------------
app.camera_entry.set("2")
app.vcam_var.set(True)
app.preset_var.set("★ Flat 2D Anime")
app.on_preset_load()
assert app.camera_entry.get() == "2", "a style preset must not touch the camera"
assert app.vcam_var.get() is True, "a style preset must not touch the OBS toggle"
assert app.clahe_var.get() is False, "the preset should have turned CLAHE off"
print("PASS  partial preset changed style only, left hardware settings alone")

# --- 7. restart-only reporting ---------------------------------------------
app.process = object()          # pretend the engine is running
app.logs.clear()
app.strength_var.set(20.0)
app.preset_var.set("★ Flat 2D Anime")
needs = app._apply_settings_dict(launcher.BUILTIN_PRESETS["★ Flat 2D Anime"])
assert "ai_strength" in needs, needs
assert any(isinstance(x, tuple) and "guidance_scale" in x[1] for x in app.logs), \
    "live settings should have been pushed over ZMQ"
print(f"PASS  restart-only keys reported: {sorted(needs)}")
app.process = None

# --- 8. history: append, dedupe, cap ---------------------------------------
for i in range(3):
    app.prompt_entry.set(f"prompt {i}")
    app.append_history()
app.append_history()                      # identical to the last -> must dedupe
hist = app.load_history()
assert len(hist) == 3, f"dedupe failed: {len(hist)} entries"
for i in range(10):
    app.prompt_entry.set(f"overflow {i}")
    app.append_history()
hist = app.load_history()
assert len(hist) == launcher.HISTORY_LIMIT, f"cap failed: {len(hist)}"
assert hist[-1]["settings"]["prompt"] == "overflow 9", "newest entry should be last"
print(f"PASS  history dedupes and caps at {launcher.HISTORY_LIMIT}")

# --- 9. history restore -----------------------------------------------------
app._refresh_history_menu()
assert app._history_entries[0]["settings"]["prompt"] == "overflow 9", "newest must be first"
app.history_var.set(app.history_menu.values[2])
app.prompt_entry.set("scratch")
app.on_history_restore()
assert app.prompt_entry.get() == "overflow 7", app.prompt_entry.get()
print("PASS  history restore applies the selected snapshot")

# --- 10. corrupt files must not crash the app -------------------------------
open(launcher.PRESETS_FILE, "w").write("{ not json")
open(launcher.HISTORY_FILE, "w").write("[[[")
assert app.preset_names() == list(launcher.BUILTIN_PRESETS)
assert app.load_history() == []
print("PASS  corrupt presets/history files degrade to empty instead of crashing")

# --- 11. save_settings must not be able to clobber presets ------------------
assert "saved" not in launcher.PRESETS_FILE.replace(tmp, "")
assert launcher.PRESETS_FILE != launcher.CONFIG_FILE
assert launcher.HISTORY_FILE != launcher.CONFIG_FILE
print("PASS  presets and history are stored outside vtuber_settings.json")

# --- 12. the Full Frame preset must set ALL THREE confinement flags --------
ff = launcher.BUILTIN_PRESETS["★ Full Frame Anime"]
assert ff["no_segment"] is True, "no_segment off -> silhouette mask still cuts you out"
assert ff["no_face_track"] is True, "face tracking on -> AI confined to a square"
assert ff["zoom"] == 1.0, "zoom > 1 -> the square shrinks further"
assert launcher.strength_to_t_index(ff["ai_strength"]) <= 20
print("PASS  Full Frame preset sets no_segment + no_face_track + zoom 1.0 together")

# --- 13. every builtin preset now pins zoom --------------------------------
for name, vals in launcher.BUILTIN_PRESETS.items():
    assert "zoom" in vals, f"{name} leaves the previous zoom in place"
print("PASS  every builtin preset pins zoom")

# --- 14. newly-live settings must not be listed as restart-only ------------
for k in ("lora", "keep_background"):
    assert k not in launcher.RESTART_ONLY_KEYS, f"{k} is live now, must not be restart-only"
assert "ai_strength" in launcher.RESTART_ONLY_KEYS
print("PASS  RESTART_ONLY_KEYS matches what the engine can actually apply live")

# --- 15. retired perf-mode names still resolve -----------------------------
src_l = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()
legacy = re.search(r"_LEGACY_PERF = \{(.*?)\}", src_l, re.S).group(1)
for old_name in ("Maximum Speed (Low Quality)", "Balanced (Recommended for ControlNet)",
                 "Maximum Quality (Low FPS)"):
    assert old_name in legacy, f"old preset/history entries using {old_name!r} would break"
print("PASS  retired performance-mode names are mapped, old saves still load")

# --- 16. no_segment round-trips through a user preset ----------------------
app.no_segment_var.set(True)
snap = app._collect_settings()
assert snap["no_segment"] is True
app.no_segment_var.set(False)
app._apply_settings_dict(snap, push_live=False)
assert app.no_segment_var.get() is True
print("PASS  no_segment round-trips through collect/apply")

# --- 17. every self-test probe targets a real command handler -------------
eng = open(os.path.join(HERE, "realtime_video.py"), encoding="utf-8").read()
listener = re.search(r"def cmd_listener_thread.*?\n    finally:", eng, re.S).group(0)
probes = app._selftest_probes()
for payload, marker in probes:
    key = list(payload)[0]
    assert f'"{key}" in cmd' in listener or f'("{key}", "' in listener, \
        f"self-test probes {key!r} but no engine handler exists"
print(f"PASS  all {len(probes)} self-test probes map to a real engine handler")

# --- 18. and each probe's marker matches what that handler actually logs ---
for payload, marker in probes:
    assert marker.rstrip(":") in eng, f"self-test waits for {marker!r} which the engine never logs"
print("PASS  every self-test marker appears in the engine's log strings")

# --- 19. image-slider defaults must match the engine's, and be neutral -----
# These three used to default to the values the engine once hardcoded (1.0 /
# +20 / +10). That was correct while the stage they feed was gated off and
# dead. Now that the stage actually runs, the same defaults would sharpen and
# saturate every existing user's picture on upgrade, so they are neutral and
# an untouched slider changes nothing.
for key, want in (("sharpness", 0.0), ("saturation", 0.0), ("brightness", 0.0),
                  ("mask_feather", 7.0), ("temporal_denoise", 0.4),
                  ("stillness_blend", 0.3)):
    assert launcher.DEFAULTS[key] == want, (key, launcher.DEFAULTS[key])
# GUI and engine defaults must agree, or a run started from the CLI looks
# different from the same settings started from the GUI.
import re as _re
for key in ("sharpness", "saturation", "brightness"):
    m = _re.search(rf'"--{key}", type=(\w+), default=([\d.]+)\)', eng)
    assert m, f"--{key} argument not found in the engine"
    assert float(m.group(2)) == launcher.DEFAULTS[key], (
        f"--{key} defaults to {m.group(2)} in the engine but "
        f"{launcher.DEFAULTS[key]} in the GUI")
print("PASS  image sliders default to a true no-op, and GUI matches engine")

# --- 20. the randomizer's positive must not fight the preset's negative ---
neg = "3d, render, realistic, photo, blurry, deformed"
kept, dropped = app._reconcile_negative("pixar 3D animation style, unreal engine", neg)
assert "3d" in [d.lower() for d in dropped], dropped
assert "blurry" in kept and "deformed" in kept, kept
kept2, dropped2 = app._reconcile_negative("flat color anime, cel shading", neg)
assert dropped2 == [] and kept2 == neg, (kept2, dropped2)
print(f"PASS  negative reconciler drops {dropped} for a 3D style, keeps everything for a 2D one")

# --- 21. LoRA strength must reach the text encoder AND the refit ----------
eng21 = open(os.path.join(HERE, "realtime_video.py"), encoding="utf-8").read()
assert "fuse_unet=False, fuse_text_encoder=True" in eng21, \
    "the user LoRA still never reaches stream.pipe.text_encoder"
assert 'lora_scale=float(state.get("lora_strength", 1.0))' in eng21, \
    "lora_strength does not reach the refit fuse"
assert "lora_strength" in launcher.DEFAULTS
print("PASS  LoRA text-encoder fuse present and strength reaches both fuse sites")

# --- 22. A/B compare is wired end to end ----------------------------------
assert '"ab_raw" in cmd' in eng21 and 'state_dict.get("ab_raw", False)' in eng21
gui22 = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read()
assert '{"ab_raw": True}' in gui22 and '{"ab_raw": False}' in gui22
print("PASS  A/B raw-camera toggle wired in both directions")

# --- 23. the Seed box must survive whatever the user types in it ----------
# It was an IntVar behind a free-text CTkEntry, so IntVar.get() raised TclError
# the moment the box was empty. save_settings() is called from start_script()
# outside any try/except, so clearing Seed silently prevented START with
# nothing in the GUI log.
import inspect
src_l = inspect.getsource(launcher)
assert "ctk.IntVar(value=int(self.settings.get(\"seed\"" not in src_l, \
    "the Seed box is an IntVar again; clearing it raises TclError and blocks START"
assert "def _seed(self)" in src_l, "no tolerant reader for the Seed box"
for bad in ("", "   ", "abc", "12.5", None):
    app.seed_var = type("V", (), {"get": staticmethod(lambda b=bad: b)})()
    got = App._seed(app)
    assert isinstance(got, int), f"_seed({bad!r}) returned {got!r}, not an int"
app.seed_var = type("V", (), {"get": staticmethod(lambda: " 4242 ")})()
assert App._seed(app) == 4242, "the Seed box no longer reads a normal value"
print("PASS  the Seed box tolerates empty/garbage input and still starts")

# --- 24. a preset that carries a LoRA must update the dropdown's guard ----
# on_lora_changed skips sending when the value equals _last_sent_lora. A preset
# load sent the LoRA directly without updating it, so picking that same LoRA
# afterwards looked like a change: the dropdown greyed, the engine correctly
# did nothing, and no completion line ever came back -> greyed out for the
# full 120s fallback.
blk = src_l[src_l.index('if "lora" in values:'):]
blk = blk[:blk.index("return changed_restart_only")]
assert "_last_sent_lora" in blk, (
    "the preset/history load sends a LoRA without updating _last_sent_lora, so "
    "the dropdown's no-op guard goes stale and re-arms the 120s grey-out")
print("PASS  a preset load keeps the LoRA dropdown's no-op guard in step")

# --- 25. per-run state must not leak across a stop -------------------------
import inspect as _i
_src = _i.getsource(launcher)
_exit = _src[_src.index("def on_process_exit"):]
_exit = _exit[:_exit.index("\n    def ", 10)]
for needle, why in (
        ("_finish_recording", "a recording left open after the engine exits "
                              "splices the next session onto the same file"),
        ("frame_buffer.clear()", "the replay deque splices two sessions"),
        ("_selftest_aborted", "a self-test running when the engine dies reports "
                              "an unconditional PASS, because on_process_exit "
                              "clears the waiting set it checks"),
        ("manual_freeze = False", "a stale freeze flag inverts the button"),
        ("_end_lora_swap", "the LoRA dropdown stays greyed after a crash")):
    assert needle in _exit, f"on_process_exit no longer resets: {why}"
print("PASS  on_process_exit resets recording, replay, self-test, freeze and LoRA state")

# a self-test cut short must say so rather than claiming everything answered
_rep = _src[_src.index("def _selftest_report"):]
_rep = _rep[:_rep.index("\n    def ", 10)]
assert "_selftest_aborted" in _rep and "ABORTED" in _rep, \
    "the self-test still reports PASS when the engine died mid-test"
assert _rep.index("_selftest_aborted") < _rep.index("if not missing:"), \
    "the abort check must come before the pass/fail branch"
print("PASS  a self-test interrupted by an engine exit reports ABORTED, not PASS")

# --- 26. controls that need a running engine must check for one -----------
_fr = _src[_src.index("def toggle_freeze"):]
_fr = _fr[:_fr.index("\n    def ", 10)]
assert _fr.index("cmd_socket") < _fr.index("self.manual_freeze ="), \
    "toggle_freeze still flips manual_freeze before checking the engine is up, " \
    "so a press while stopped inverts the button for the next run"
_rec = _src[_src.index("def toggle_recording"):]
_rec = _rec[:_rec.index("def _finish_recording")]
assert "self.process is None" in _rec and "if False" not in _rec, \
    "recording can still be started with no engine, producing an empty MP4 " \
    "that reports 'saved successfully'"
print("PASS  freeze and record both require a running engine")

# --- 27. Tab must still traverse focus while typing -----------------------
# Every shortcut is bound on the toplevel, which fires regardless of focus, and
# returning "break" there also aborts Tk's `all` bindtag where Tab traversal
# lives — so Tab stopped moving between the prompt boxes anywhere in the app.
_ab = _src[_src.index("def _ab_press"):]
_ab = _ab[:_ab.index("\n    def ", 10)]
assert "_typing" in _ab and _ab.index("_typing") < _ab.index('return "break"'), \
    "the A/B Tab shortcut still swallows Tab while the user is typing"
assert "def _typing" in _src
for cls in ("Entry", "CTkEntry", "CTkTextbox"):
    assert cls in _src[_src.index("def _typing"):_src.index("def _ab_press")], cls
print("PASS  Tab traverses focus in text fields and only triggers A/B elsewhere")

# --- 28. the LoRA fallback timer must be cancelled, not stacked -----------
assert "after_cancel(self._lora_timer)" in _src, \
    "each swap schedules another 120s timer without cancelling the last, so an " \
    "old timer re-enables the dropdown in the middle of a newer swap"
assert "self._lora_timer = self.after(120000" in _src
print("PASS  the LoRA fallback timer is cancelled before a new one is scheduled")

# --- 29. a killed engine must not be reported as a crash ------------------
assert _src.count("_force_killed") >= 3, \
    "a STOP we forced still logs 'see the error above' with no error above; " \
    "the flag must be SET in the kill path, and read+cleared in on_process_exit"
_gs = _src[_src.index("proc.terminate()") - 1500:_src.index("proc.terminate()")]
assert "self._force_killed = True" in _gs, \
    "nothing sets _force_killed on the path that actually kills the process"
assert "proc.wait(timeout=60)" in _src, \
    "a stop landing inside the uninterruptible TensorRT build still gets only " \
    "6 seconds before being killed mid-write"
print("PASS  a forced stop is reported honestly, and gets time to finish first")

# --- 30. a corrupt settings file must be preserved, not overwritten -------
_rj = _src[_src.index("def _read_json"):_src.index("def _write_json")]
assert ".corrupt" in _rj, \
    "an unreadable presets.json still degrades silently to empty, and the next " \
    "Save As overwrites every preset that was still recoverable"
print("PASS  an unreadable settings file is kept aside instead of overwritten")

# --- 31. STOP must be honoured around the uninterruptible build ----------
# main() only checked STOP *after* the whole model load returned, so a stop
# during a 5-15 minute TensorRT build was ignored entirely: the launcher's
# grace period expired and it fell through to TerminateProcess, with a phantom
# "see the error above" and a risk of a truncated unet.engine.
eng_src = open(os.path.join(HERE, "realtime_video.py"), encoding="utf-8").read()
assert "def _abort_if_stopped" in eng_src and "class EngineStopped" in eng_src
_load = eng_src[eng_src.index("def load_model_and_engine"):eng_src.index("def refit_prepare")]
assert _load.count("_abort_if_stopped") >= 2, (
    "load_model_and_engine has no STOP checkpoints, so a stop during startup "
    "is ignored until the whole build finishes")
assert "_abort_if_stopped(\"before the TensorRT build\")" in _load
assert "except EngineStopped:" in eng_src, \
    "EngineStopped escapes main() and is reported as a crash"
print("PASS  STOP is honoured at each startup checkpoint around the build")

print("\nALL PASS")
