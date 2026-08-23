# HANDOFF — TensorRT AI VTuber Studio

Written 2026-08-22 for whoever picks this up next. Read this before changing code.

---

## 1. State of the world

The engine **boots, reaches READY, and renders**. It is not crashing and the
GUI is wired up correctly. Evidence, from `logs/engine-20260822-201417.log`:

```
20:14:19 gpu: NVIDIA GeForce RTX 3080 Ti  vram free 10.8/12.0 GiB
20:14:34 [Engine] READY — streaming frames.
20:14:39 [Engine] Rendering at 8.7 FPS
```

**One defect is still open**, and it is the reason the output looks wrong:

```
realtime_video.py: RuntimeWarning: invalid value encountered in cast
  out_np = (np.clip(out_np, 0.0, 1.0) * 255).astype(np.uint8)
```

That warning fires only on **NaN or inf**. `np.clip` passes NaN through
untouched, and casting NaN to `uint8` yields undefined bytes. The model is
emitting non-finite pixels and they reach the screen as garbage. This predates
all of the changes below — the old code did `(out_np * 255).astype(np.uint8)`
and corrupted just as silently.

A guard now replaces non-finite values and counts them, so the picture degrades
instead of exploding, **but the guard is a symptom mask, not the fix.** Do not
call this done until the NaN count is zero.

Everything else in §3 is fixed and verified as far as it can be without a GPU.

---

## 2. Start here: find the NaN

Do not guess. `diagnose.py` isolates it. It runs four configurations, each in
its own subprocess, feeds a fixed synthetic portrait through the real TensorRT
engine 30 times, and traces finiteness at each stage inside the denoise step by
wrapping `stream.unet_step`.

```
venv\Scripts\python.exe diagnose.py      # ~2 min, writes logs/diagnose.log
```

Configurations tested: `cfg_type=none` (the known-good baseline),
`cfg_type=self` at guidance 1.6 and 1.2, and `none` again with CUDA graphs on
for a timing comparison.

### Reading the result

Each line reports `latent_in` (the latent handed to the UNet, i.e. the VAE
*encoder* output plus noise), `model_pred` (the UNet's raw output), `x0` (after
the scheduler step), and the final decoded array.

| What you see | What it means | Where to go next |
| --- | --- | --- |
| `none` clean, `self` dirty | RCFG is the cause — `stock_noise` accumulates and overflows fp16 | See "If it is RCFG" below |
| `self` dirty at 1.6 but clean at 1.2 | Magnitude-dependent overflow, same cause | Clamp, or cap the CFG slider |
| Both `none` and `self` dirty | Not RCFG. The engine or the VAE is at fault | Rebuild engines (see §5 first), and A/B TinyVAE against the stock SD VAE |
| `latent_in=False` | The **VAE encoder** engine is producing NaN | Input side; rebuild `vae_encoder.engine` |
| `latent_in=True`, `model_pred=False` | The **UNet** engine | Rebuild `unet.engine`; suspect a bad or mismatched cached engine |
| `model_pred=True`, `x0=False` | Scheduler math — `c_out`/`c_skip`/`alpha_prod_t_sqrt` in fp16 | Check `t_index`; very low indices push these into fp16 range trouble |
| All three `True`, final array dirty | The **VAE decoder** (TinyVAE) | Try the stock VAE, or run decode in fp32 |

### If it is RCFG

`cfg_type="self"` was chosen deliberately: with `cfg_type="none"`,
StreamDiffusion hard-pins `guidance_scale` to 1.0 inside `prepare()` and never
reads `delta`, which made the CFG and Delta sliders completely inert. `"self"`
honours both **and** keeps `trt_unet_batch_size` at 1, so the cached engine
stays valid. Three ways out, best first:

1. Clamp `stream.stock_noise` each frame (e.g. to ±4) in the main loop. Cheapest,
   keeps the sliders working. `stock_noise` magnitude is already reported per
   iteration in `diagnose.log` — check whether it grows monotonically.
2. Keep the RCFG residual in fp32. More invasive; touches StreamDiffusion.
3. Revert to `cfg_type="none"` and **remove the CFG and Delta sliders from the
   GUI**. Honest fallback: `t_index` is the dominant quality dial anyway. If you
   do this, do not leave dead sliders in the UI — that is the trap the previous
   version fell into.

`cfg_type="initialize"` or `"full"` are *not* drop-in: they change
`trt_unet_batch_size` to 2, which needs a fresh engine build.

---

## 3. What was wrong, and what changed

### A. Why the AI appeared not to apply at all

| Root cause | Fix |
| --- | --- |
| `cfg_type="none"` pins `guidance_scale` to 1.0 and ignores `delta`. The CFG and Delta sliders did **nothing**. | `cfg_type="self"` (RCFG Self-Negative), batch stays 1, cached engine still valid. **See §2 — this may be the NaN source.** |
| `t_index_list=[38]` of 50 steps. Timesteps descend, so 38 starts denoising from a nearly clean latent — output is the webcam with a mild filter. The known-good backup used `[26]`. | Exposed as the GUI "AI Strength" slider (`--t_index`, 0→45, 100→12). |
| Compositing was unconditional: the AI covered only the MediaPipe silhouette, the rest was the real room. | Now opt-in via "Keep Real Background" (`--composite`). Default is the full AI frame. |
| `stream.fuse_lora()` was missing after `load_lcm_lora()`. Unfused weights are not baked into the ONNX export. | Added. Cached engines hid this; it would have bitten on the first LoRA build. |
| Non-finite pixels cast to garbage bytes, silently. | Detected, counted, logged once at frames 1/10/100/1000, and `nan_to_num`'d. **Symptom mask — see §1.** |

### B. Crashes, hangs, and lies

| Root cause | Fix |
| --- | --- |
| `launcher.py` sent `--audio_sync`; `realtime_video.py` never declared it. argparse exited **code 2** the instant that switch was on. | Declared, plus `parse_known_args` so a flag mismatch logs instead of killing the engine. |
| `out_frame` referenced before assignment when the model returned `None` → `UnboundLocalError`. This is why the similar-image filter had been commented out. | Guarded; the Freeze slider works again (1.00 = off). |
| A dead worker thread set `STOP`, the main loop exited normally, and the process returned **0** — the GUI reported a hard crash as a clean shutdown. | `FAILED` event; exits 1. |
| `launcher` called `terminate()` = `TerminateProcess` on Windows, which is uncatchable. The engine's entire cleanup path was dead code: the webcam was never released, the virtual camera never closed. | Engine installs SIGTERM/SIGINT/SIGBREAK handlers; launcher spawns with `CREATE_NEW_PROCESS_GROUP` and sends `CTRL_BREAK_EVENT`, falling back to terminate then kill. |
| Shutdown closed the ZMQ socket and virtual camera while the postprocess thread could still be mid-`send()`. | Threads joined first; if one will not stop, resources are deliberately leaked rather than freed underneath it. |
| `cap.release()` could race a stalled `read()` — `cv2.VideoCapture` is not thread-safe; a native use-after-free on an unplugged webcam. | Join with timeout; skip the release if the thread is still inside `read()`. |
| A stolen or unplugged camera hung the pipeline at 0 FPS forever with no error. | Bails out after 30s with a real failure exit. |
| Camera-open failure silently fell back to a **random-noise** mock camera; the model dutifully hallucinated garbage over it. | Scans indices 0–5, reports which opened, exits 1. The mock is now opt-in (`--mock_camera`) and a structured gradient, not noise. |
| `self.config = {...}` on a `ctk.CTk` subclass shadows Tk's own `Misc.config()`. | Renamed to `self.settings`. |
| Hardcoded `venv\Scripts\python.exe` relative to the *current working directory*; no error handling around `Popen`. A bad path produced a GUI that did nothing at all. | Anchored to the script directory, falls back to `sys.executable`, `Popen` failures are logged. |

### C. Throughput

| Root cause | Fix |
| --- | --- |
| **A regression I introduced and then reverted.** "Skip the frame before MediaPipe if `Q_IN` is full" looked like an optimisation but served the camera wait and CPU preprocessing *in series* with the GPU — the producer could never work ahead. | Reverted to always-produce / replace-on-full. **The producer must stay ahead of the consumer. Do not re-introduce this.** |
| FPS was averaged over the first 5 seconds, which included TensorRT's warmup calls. The reported 8.7 FPS is not a steady-state number. | 5s warmup skipped; the frame time is split into `wait` (starved by the producer) vs `infer` (GPU busy). |
| The Reinhard colour anchor could never drift: `color_transfer` forces the frame to the anchor's mean and std, and that already-corrected frame was fed back into the EMA — a mathematical no-op. The palette locked to the first post-warmup frame for the whole session. | Accumulates the pre-transfer frame. |
| `use_cuda_graph=True` was set while the output was corrupt. | Now `--cuda_graph`, **off by default**. Worth ~10% once the NaN is fixed. |

### D. Diagnosability

Everything now writes to `logs/`:

- `engine-latest.log` + a timestamped copy — environment block (torch / diffusers
  / tensorrt / mediapipe versions, GPU, free VRAM, resolved settings), every
  engine line, and the full traceback on death.
- `launcher-latest.log`, `launcher-crash.log`.
- `diagnose.log`.

`threading.excepthook` logs a dead worker's traceback instead of leaving a
silent hang. The main loop reports producer starvation.

---

## 4. Invariants — do not break these

- **`frame_buffer` must be 1.** Any other value makes TensorRT expect a
  different batch and fail with a shape error. It is pinned in `build_args`.
- **`cfg_type` changes the engine contract.** `none` and `self` keep
  `trt_unet_batch_size` at 1. `initialize` makes it 2, `full` makes it 2×.
  Changing to either requires a rebuild.
- **`fuse_lora()` before `accelerate_with_tensorrt()`**, always.
- **`prompt_embeds`: `copy_()` in place, never rebind** after acceleration.
  `StreamDiffusion.update_prompt()` rebinds, which is why the lip-sync embeds
  are pre-computed and re-pinned *before* acceleration.
- **The producer must stay ahead of the GPU.** Drop stale frames from the
  queue, never skip producing them.
- **Join worker threads before closing `vcam` / the ZMQ socket / log handles.**

---

## 5. Environment landmines

- **`numpy 2.4.6` has broken `onnxruntime`** (`_ARRAY_API not found`,
  `numpy.core.multiarray failed to import`). Harmless right now because the
  engines are already built — but **any fresh ONNX export will fail**, which
  includes adding a character LoRA. Fix before rebuilding:
  `pip install "numpy<2"` in the venv.
- `diffusers 0.24.0` uses the pre-PEFT LoRA backend (hence the
  `fuse_text_encoder_lora is deprecated` warning). Upgrading changes LoRA
  semantics — do not do it casually.
- `opencv 5.0.0` is an unusually new major version. Not implicated in anything
  observed, but worth remembering if something behaves oddly at the cv2 layer.
- Stack: torch 2.5.1+cu124, tensorrt 9.0.1, mediapipe 0.10.14, python 3.11.9.
- Engines are cached per LoRA in `engines_tinyvae_<lora>_fb1/`. First build is
  5–15 minutes.

### Known-good reference

`realtime_video_backup.py` is the last configuration known to produce output:
`t_index=26`, `cfg_type="none"`, `use_cuda_graph=False`, `fuse_lora()` called,
synchronous single-threaded loop. `engines_tinyvae_fb1/` was built from it. When
in doubt, bisect against that. `engines_tinyvae_fb2/` is stale and unused.

---

## 6. Not bugs — do not "fix" these

- `--emotion_sync` is accepted and unused. Kept so older saved configs and
  command lines do not break.
- `--cfg_type`, `--no_face_track`, `--no_color_lock`, `--temporal_denoise`,
  `--mock_camera`, `--cuda_graph` are CLI-only by design; the GUI does not emit
  all of them.
- `realtime_video_prev.py` and `launcher_prev.py` are the pre-fix snapshots.
  `realtime_video_backup.py` is the older synchronous engine. All three are
  references, not dead code to delete.
- The `test_*.py` files at the repo root are ad-hoc scratch from an earlier
  session. `test_speed5.py` references a `realtime_video_threaded.py` that does
  not exist. `smoke_test.py` and `diagnose.py` supersede them.

---

## 7. Verification available without a GPU

`verify.py` compiles both files, stubs the GPU stack, and replays the exact
command `launcher.py` builds through `realtime_video.py`'s parser — that
handshake is what broke before. It also checks the AI-Strength → `t_index`
mapping and that `color_transfer` survives degenerate frames.

```
python verify.py          # no GPU needed
python smoke_test.py      # GPU, synthetic camera, exits 0 on READY + FPS
python diagnose.py        # GPU, the NaN hunt
```

---

## 8. Definition of done

1. `diagnose.log` shows `0/30 frames non-finite` for the shipped configuration.
2. `smoke_test.py` exits 0.
3. A real run logs a steady-state FPS with `nan_frames` absent from the line.
4. No slider in the GUI is inert. If CFG and Delta cannot be made to work
   safely, remove them rather than leaving them connected to nothing.
