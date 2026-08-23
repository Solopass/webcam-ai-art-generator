# TensorRT AI VTuber Studio

A real-time AI VTuber engine powered by **StreamDiffusion**, **TensorRT**, and
**CustomTkinter**. Turns a webcam into a customizable anime avatar for OBS or
Discord.

> **Status: one open defect.** The model intermittently emits non-finite
> (NaN/inf) pixels, which reach the screen as garbage. A guard replaces them so
> the picture degrades instead of exploding, but it is a symptom mask.
> **[HANDOFF.md](HANDOFF.md) has the full state, the diagnosis procedure, and
> the decision tree.** Read it before changing anything.

## Running it

```
Start_GUI.bat                             # or: venv\Scripts\python.exe launcher.py
venv\Scripts\python.exe smoke_test.py     # headless, synthetic camera
venv\Scripts\python.exe diagnose.py       # the NaN hunt — see HANDOFF.md §2
python verify.py                          # static checks, no GPU needed
```

Everything writes to `logs/` — `engine-latest.log` (with an environment block
and full tracebacks), `launcher-latest.log`, `diagnose.log`.

## Architecture

Three threads, decoupled by single-slot queues, so throughput is set by the
slowest node (the UNet) rather than the sum of every stage:

| Thread | Work |
| --- | --- |
| `camera_thread` | capture → face-tracked square crop → CLAHE → temporal denoise → selfie mask → `Q_IN` |
| main thread | TinyVAE encode → TensorRT UNet → TinyVAE decode → `Q_OUT` |
| `postprocess_thread` | unsharp → Reinhard colour lock → saturation → composite → ZMQ / virtual camera |

Both queues hold one frame and the producer **replaces** a stale entry rather
than skipping the work. That matters: gating the preprocessing on "is the queue
full" serialises the camera wait and the CPU work with the GPU instead of
overlapping them, and costs more than it saves. See HANDOFF.md §4.

The main loop reports its frame time split into `wait` (starved by the producer)
versus `infer` (GPU busy), after a 5-second warmup — TensorRT's first calls are
not representative.

## Controls

| Control | What it actually does |
| --- | --- |
| **AI Strength** | `--t_index` (0 → step 45, 100 → step 12). The single most important dial. A high `t_index` starts denoising from an almost clean latent, so the output looks like the raw webcam with a light filter. If the avatar "isn't applying", raise this. |
| **CFG** | `--guidance_scale`. Only has an effect when `cfg_type` is not `none`; the engine runs `self` (RCFG Self-Negative), which honours CFG **and** keeps the UNet batch at 1, so no engine rebuild is needed. Under suspicion for the NaN — see HANDOFF.md §2. |
| **Delta** | RCFG virtual-residual weight. Higher = the model invents more (hair, outfits); lower = sticks to the webcam. Also inert under `cfg_type=none`. |
| **Freeze** | Similar-image filter threshold. **1.00 = off.** Lower halts regeneration while you sit still — cheap, but it looks frozen if set too aggressively. |
| **Keep Real Background** | Composites the AI avatar over your real room using the MediaPipe selfie mask. Off = the full AI frame. A background image forces it on (only if the file exists). |
| **Normalize Lighting** | CLAHE on the input. Helps in dim or unevenly lit rooms. |
| **CUDA Graph** | ~10% faster, but has emitted stale/corrupted frames on some drivers. Off by default; leave it off until the NaN is resolved. |

CLI-only flags the GUI does not emit: `--cfg_type`, `--no_face_track`,
`--no_color_lock`, `--temporal_denoise`, `--mock_camera`.

## Engine cache

TensorRT engines are cached per LoRA in `engines_tinyvae_<lora>_fb1/`. The first
build for a new LoRA takes 5–15 minutes; the log says so when no cached engine
is found. `frame_buffer` is pinned to 1 — any other value makes TensorRT expect
a different batch and fail with a shape error.

**Before any rebuild:** `numpy 2.x` has broken `onnxruntime` in this venv, so
the ONNX export will fail. `pip install "numpy<2"` first.

## Troubleshooting

- **Output is garbage / speckled** — non-finite pixels from the model. Check the
  log for `non-finite pixels from the model`, then run `diagnose.py`.
- **Output looks like a lightly filtered webcam** — AI Strength too low
  (`t_index` too high), or *Keep Real Background* is on and only your silhouette
  is being stylised.
- **Colours strobe frame to frame** — the Reinhard colour lock is on by default
  (`--no_color_lock` disables it).
- **Engine exits with code 2** — an argument mismatch between the launcher and
  the engine. The engine uses `parse_known_args` and logs unknown flags rather
  than dying, so this should no longer happen.
- **Exit code 1** — a worker thread died or the camera stopped delivering
  frames. The traceback is in `logs/engine-latest.log`.
- **Camera won't open** — the engine scans indices 0–5 and prints the ones that
  responded. Discord, Zoom, Teams and OBS all hold the device exclusively.
- **New engine build produces noise** — the LCM-LoRA must be fused
  (`stream.fuse_lora()`) before the ONNX export, or the exported UNet has no LCM
  weights and one step cannot converge.

## Directory

- `HANDOFF.md` — current state, changelog, open defect, invariants. **Start here.**
- `launcher.py` — CustomTkinter GUI, spawns and supervises the engine.
- `realtime_video.py` — headless TensorRT inference engine.
- `audio_sync.py` — mic-driven open/closed-mouth prompt switching.
- `diagnose.py` — per-stage finiteness and timing across four configurations.
- `smoke_test.py` — runs the engine against a synthetic camera.
- `verify.py` — GPU-free compile and argument-handshake checks.
- `loras/` — drop `.safetensors` character models here.
- `logs/` — engine, launcher and diagnostic output.
- `vtuber_settings.json` — saved UI state.
- `realtime_video_backup.py`, `*_prev.py` — reference snapshots, see HANDOFF.md §6.
