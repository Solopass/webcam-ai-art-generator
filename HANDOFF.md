# HANDOFF — engineering notes

Last updated 2026-08-29. `README.md` is the user-facing doc; this is the part
that is not guessable from reading the code. Read §1 and §2 before changing
anything.

---

## 1. Invariants

**Engine / model**

- **`cfg_type` and step count are an engine contract, not knobs.**
  `trt_unet_batch_size` is `steps × frame_buffer` for `none`/`self`,
  `steps + 1` for `initialize`, `2 × steps × frame_buffer` for `full`. The build
  runs `full` with 2 steps → **batch 4**. Changing either needs a rebuild.
  Engine dirs are keyed `engines_tinyvae_base_fb{N}_steps{M}{cfg_suffix}/`; the
  suffix is omitted for `full` so existing caches stay valid.
- **`guidance_scale` must stay > 1.0.** At exactly 1.0 StreamDiffusion disables
  classifier-free guidance, which halves `prompt_embeds` and changes the UNet
  batch. Slider floors at 1.05 and `build_command` clamps again.
- **Prompt embeds: `.copy_()` in place, sized from `stream.prompt_embeds.shape`,**
  never from an assumed `[uncond, cond]` layout.
- **`stream.fuse_lora()` after `load_lcm_lora()`, before acceleration.**
- **The user LoRA's text-encoder half is fused AFTER acceleration**
  (`fuse_unet=False, fuse_text_encoder=True`). Doing it before would bake the
  LoRA into a freshly built engine, and `engine_dir` is a shared `base` across
  every LoRA — one build would poison all of them.
- **`t_index_list` must be strictly ascending.** It is the dominant quality dial.
- **No `torch.cuda.empty_cache()` in the frame loop** — device-wide sync.
  `torch.no_grad()` around `encode_prompt` is what fixes the VRAM leak.

**Threading / lifecycle**

- **The producer must stay ahead of the GPU.** Drop stale frames from `Q_IN`;
  never skip *producing* them because the queue is full.
- **`postprocess_thread` must not mutate `full_frame` in place.** It free-runs at
  ~30 Hz while inference delivers ~10 fps, so any in-place composite is applied
  ~3× per frame: blend modes compound and saturate, and VFX opacity renders far
  stronger than its value. It re-composites from `pristine_full_frame`.
- **Join worker threads before closing `vcam` / sockets / log handles.**
  `postprocess_thread` owns `vcam` and may reopen it, so main's reference can be
  stale; main only closes as a backstop, wrapped.
- **Never `zmq.Context.term()` with a socket still open** — it blocks forever.
  Use `destroy(linger=0)`.
- **Console control events cannot stop the engine.** It is spawned
  `CREATE_NO_WINDOW`, so `CTRL_BREAK_EVENT` can never arrive. Shutdown goes over
  the ZMQ command channel; the PULL socket binds *before* the model load so STOP
  works during a 5–15 minute build.
- **Never block the Tk main thread.**

**GUI**

- **Never name an attribute `self.config` on a `ctk.CTk` subclass** — it shadows
  Tk's `Misc.config()`.
- **`imageio` writers have `close()`, not `release()`.**
- **`save_settings()` rebuilds `vtuber_settings.json` from the widgets** on every
  START and close. Anything persisted there without a widget behind it is
  deleted — which is why presets and history have their own files. If you add a
  setting, add it to `_collect_settings` or it will silently vanish.
- **A control wired to something with no effect is a defect.** See §3.

---

## 2. The masking / crop pipeline

The least obvious part of the codebase, and the source of "why doesn't the AI
fill the frame".

`camera_thread` takes a `min(h, w) / zoom` square that tracks your face, unless
`--no_face_track` (Full Frame Mode), in which case `cw, ch = w, h`.

Segmentation is then applied **twice**:

1. **Before inference** the background is replaced with flat grey 64, so the
   model never sees the room and cannot generate one.
2. **During the HD paste-back** in `postprocess_thread`:
   `final_ai = final_ai * mask + base_region * (1 - mask)`, putting the real room
   back around your silhouette.

Neither is gated on `--composite`. `--no_segment` ("Paint Whole Frame") gates the
segmenter itself and is re-read from `state_dict` every frame, so it toggles
live; because both uses read the same `soft_mask`, one gate covers both. Default
**off**. ★ Full Frame Anime sets it with `no_face_track` and `zoom: 1.0` — all
three are required.

Trade-off it restores as a *choice*: with segmentation off the model paints your
real room, which is the point, but it can also hallucinate room detail. That was
the original reason it was made unconditional.

---

## 3. Verification, and two ways it has lied

`run_checks.py` runs `test_engine_logic.py` and `test_presets.py` with no GPU.
`.github/workflows/checks.yml` runs them on push.

The **🧪 Self-Test** button fires every live command and waits for the engine's
log echo. Both tools have produced false confidence, and the lessons are worth
keeping:

- **An echo proves arrival, not effect.** `cmd_listener_thread` logged
  `Freeze Threshold: 0.95` and the main loop then `pop`ped the value and threw it
  away before its only consumer read it. The self-test passed; the slider was
  dead, and took Stillness Blend with it. **Anything visual needs eyes on the
  picture.**
- **A coverage test only sees the forms it matches.** `test_engine_logic.py`
  recognised handlers written `"key" in cmd` and sends written as a literal next
  to `send_command`. The image sliders use a `_live(key)` helper and a
  `(key, label)` tuple loop, so they were invisible on *both* sides and the test
  reported full coverage while checking nothing about them. Both regexes are
  widened; it now verifies 23 commands. **If you add a new send or handler
  syntax, widen it again.**

Discipline that keeps this from regrowing: every new control ships with its
engine handler, a self-test probe, and a test assertion, in the same change.

---

## 4. Presets and history

`presets.json` and `history.json`, deliberately outside `vtuber_settings.json`
(§1). Atomic writes via `os.replace`; corrupt files degrade to empty.

- `_collect_settings()` is the single source of truth for what is persisted —
  `save_settings()`, preset saving and history snapshots all use it.
- `_apply_settings_dict()` writes a possibly-partial dict back and returns the
  restart-only keys that changed. `RESTART_ONLY_KEYS` must stay in sync with
  `cmd_listener_thread`'s handled keys.
- `var.set()` on a CTkOptionMenu updates the text **without firing its command**,
  so a preset load pushes `lora` explicitly.
- Sliders set programmatically don't fire their callback either; `_slider_row`
  registers `(var, label, fmt)` and `_refresh_slider_labels()` updates them.
- Built-in presets are partial by design, prefixed `★`, and undeletable.

---

## 5. Ranked open work

1. **`refit_lora_to_trt` blocks the inference thread** for 30–60s. It warns and
   fails safe, but a background refit against a double-buffered engine would
   remove the freeze.
2. **`refit_lora_to_trt` builds `engine_dir` without the `cfg_suffix`** that
   `load_model_and_engine` adds. Harmless today because the launcher never sends
   `--cfg_type`, but any non-`full` run would refit against the wrong directory.
3. **`frame_buffer > 1` is broken as a concept.** `x_in.repeat(fb, 1, 1, 1)` then
   `output_image[-1]` runs the same frame twice and discards a result — 2× the
   work for an identical picture. Either batch distinct frames or remove the
   option; `engines_tinyvae_base_fb2_steps2/` is dead weight either way.
4. **`refit_lora_to_trt` needs `engine_dir/onnx/unet.opt.onnx`**, which only
   exists if this machine built that engine. A copied cache cannot refit.
5. **Camera Zoom is silently ignored in Full Frame Mode** while its slider
   description still describes zooming.
6. **The replay deque isn't cleared on start/stop**, so a Ctrl+S right after a
   restart splices two sessions.
