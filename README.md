# AI VTuber Studio

Real-time AI restyling of a webcam or screen feed, powered by **StreamDiffusion**,
**TensorRT** and **CustomTkinter**. Point a camera at yourself, type a prompt, and
get a live stylised video stream you can send to OBS or Discord.

~10 FPS inference on an RTX 3080 Ti, interpolated to ~30 FPS output.

```
Start_GUI.bat            # or: venv\Scripts\python.exe launcher.py
python run_checks.py     # all GPU-free checks, ~10s
```

First-time setup is `setup.ps1`. **Read the Environment section** — there is a
patched dependency, and it is the reason a fresh clone can build at all.

---

## The dial that matters: AI Strength

The setting people get wrong, including twice in this project's history.

StreamDiffusion starts denoising at a chosen step of a 50-step schedule.
Timesteps run **most-noisy → least-noisy**, so a *high* step index hands the
model an almost-finished image and asks for a touch-up. It never gets enough
noise to discard your real face, so your actual photographic lighting and skin
texture survive — which is what "it looks 3D, not anime" means.

| AI Strength | Step index | What you get |
| --- | --- | --- |
| 20–30 | 37–35 | Your real face with a filter over it |
| 40–55 | 32–27 | Recognisably you, clearly stylised |
| 70–90 | 22–15 | Flat, committed to the prompt |
| 95–100 | 14–12 | Barely anchored to the webcam |

If the output looks like a filtered photo, **raise AI Strength** — CFG cannot
make the model repaint something it was never given room to repaint.

Three other things fight a flat look, and ★ Flat 2D Anime turns them all off:
**Normalize Lighting (CLAHE)** amplifies real shading on the input; **Sharpness**
re-adds photographic micro-detail on the output; and a **negative prompt**
without `3d, render, realistic, photo, octane, blender` isn't pushing back.

---

## Making the AI fill the whole frame

By default the AI does **not** cover the picture. Three things confine it:

1. **Zoom** — the engine restyles a `min(h, w) / zoom` square. At Zoom 1.9 on a
   1280×720 camera that's ~379×379, about 8% of the frame.
2. **Face tracking** — that square follows your face. Turn on **Full Frame Mode**.
3. **The cutout mask** — segmentation greys out your background before inference
   *and* masks the result to your silhouette during the paste-back. Turn on
   **Paint Whole Frame (Disable Cutout Mask)**.

The **★ Full Frame Anime** preset sets all three at once. Paint Whole Frame is
off by default, so nothing changes until you ask for it.

---

## Controls

**Live** — apply instantly over ZMQ while the engine runs:

| Control | What it does |
| --- | --- |
| Prompt / Negative | Enter or **Apply ✨**. Ctrl+R rolls a random style. |
| Prompt Strictness (CFG) | How hard the model chases the prompt. Floor 1.05. |
| Freeze Filter | Below 1.00, holds the image while you sit still. |
| Stillness Blend | How much raw webcam is re-fed while frozen. 0 = off. |
| Motion Blur | Output smoothing between inference frames. |
| Sharpness | 0 = off. Higher re-adds photographic detail. |
| Saturation / Brightness | Post-generation colour. |
| Mask Feather | Softness of the cutout edge. No effect with Paint Whole Frame. |
| Input Denoise | Smooths webcam grain before the model sees it. |
| Background Bokeh | Blurs the real background when compositing. |
| Camera Zoom | Ignored in Full Frame Mode. |
| LoRA Strength | Applies on the next swap or START — baked in at fuse time. |
| Character (LoRA) | Hot-swaps live. **Freezes output 30–60s** while TensorRT refits. |
| Composite Real Background · Paint Whole Frame · VFX opacity & blend mode | |

**Launch-only** — grey out while running, applied on next START: AI Strength,
Performance Mode, ControlNet, Camera, Seed, and the hardware toggles.

### Keyboard

| Key | Action |
| --- | --- |
| Ctrl+R | Randomize style |
| Ctrl+S | Save 5s WebP replay |
| F12 / Ctrl+Space | Snapshot |
| F8 / Ctrl+F | Manual freeze |
| F5 / F6 | Previous / next preset |
| **Hold Tab** | Show the raw camera (A/B compare) |

---

## Presets, history and the self-test

**Style Preset** (Settings tab): Load / Save As… / Delete. The ★ built-ins are
*partial* — they carry style keys only, so loading one never touches your camera,
OBS toggle or preview options.

| Preset | For |
| --- | --- |
| ★ Flat 2D Anime | Strength 82, CLAHE off, freeze off, anti-3D negatives |
| ★ Full Frame Anime | Flat 2D **and** paints the entire frame — no crop, no cutout |
| ★ Painterly | Strength 65, brushwork |
| ★ Subtle Filter | Strength 30, keeps your real face |

**Session History** (System tab): every START snapshots all settings, newest
first, labelled by time and prompt. One button restores. Deduped, last 50 kept.

Both live in `presets.json` and `history.json`, **not** `vtuber_settings.json` —
that file is rebuilt from the widgets on every START and close, so anything in
it without a widget behind it gets erased.

**🧪 Run Self-Test** (System tab) fires all 23 live commands with their current
values — nothing changes — and confirms the engine echoes each back, naming any
that don't answer. Caveat worth knowing: an echo proves the command *arrived*,
not that it has an effect. A control can answer and still be dead (this happened
to Freeze), so anything visual still wants eyes on the picture.

---

## Environment

python 3.11.9, torch 2.5.1+cu124, tensorrt 9.0.1, diffusers 0.24.0, mediapipe
0.10.14, pyzmq 27.2.0, numpy 1.26.4.

- **`StreamDiffusion/` is patched, and the patches live in `patches/`.** That
  directory is the only copy in version control, because `StreamDiffusion/` is
  gitignored. `setup.ps1` checks out base `b623251` and applies them. Without
  them an engine build fails with
  `TypeError: compile_unet() got an unexpected keyword argument 'timing_cache'`.
- **numpy must stay below 2** — numpy 2.x breaks this onnxruntime build, which is
  harmless with cached engines and fatal the moment you rebuild one. The engine
  logs numpy and onnxruntime versions at startup so you find out first.
- Two OpenCV distributions are installed and share the `cv2` package; last
  install wins (currently contrib 5.0.0). First thing to check if cv2 acts odd.
- Engines cache per `(frame_buffer, steps)` in
  `engines_tinyvae_base_fb{N}_steps{M}/`. First build is 5–15 minutes.

---

## Troubleshooting

- **Looks 3D / like a filtered photo** → AI Strength too low. See above.
- **AI only covers my upper body** → zoom, face tracking and the cutout mask.
  Load ★ Full Frame Anime.
- **Output froze for ~30–60s** → expected after a LoRA swap; it's the refit.
- **Engine exits with code 1** → a worker thread died or the camera stopped for
  30s. Traceback in `logs/engine-latest.log`.
- **Camera won't open** → the engine scans indices 0–5 and prints which ones
  responded. Discord, Zoom, Teams and OBS hold the device exclusively.

Everything writes to `logs/`: `engine-latest.log` (environment block, every
engine line, tracebacks), `launcher-latest.log`, `launcher-crash.log`.

---

## Directory

- `launcher.py` — the GUI; spawns and supervises the engine.
- `realtime_video.py` — the headless TensorRT inference engine.
- `audio_sync.py` — mic-driven mouth state.
- `prompts.md` — prompt-builder dropdown contents.
- `patches/` — required StreamDiffusion patches. See `patches/README.md`.
- `run_checks.py`, `test_engine_logic.py`, `test_presets.py` — GPU-free checks.
- `compile_base_engine.py`, `compile_controlnet_fused.py`, `*.bat` — engine builds.
- `process_video.py` — offline VFX render of a video file.
- `presets.json` / `history.json` — style presets and session snapshots.
- `HANDOFF.md` — engineering notes and invariants. Read before changing code.
