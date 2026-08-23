# TensorRT AI VTuber Studio

A real-time AI VTuber engine powered by **StreamDiffusion**, **TensorRT**, and **CustomTkinter**. Turns your webcam—or desktop screen recording—into a customizable, live-animated anime avatar for OBS or Discord.

> **Status: Highly Optimized & Production Ready.** ~10 FPS on an RTX 3080 Ti (2-step, full CFG), smoothed to a buttery 30 FPS output with real-time zero-lag UI tuning. 

## Features

- **Live UI Tuning:** Tune your prompt, CFG, Motion Blur, and trigger sensitivities live via a ZeroMQ event channel without restarting the TensorRT engine.
- **MediaPipe Face Tracking:** Dynamically tracks your face, panning and cropping the camera automatically. Maps your real-world facial expressions (smile, closed eyes) directly into the AI prompt!
- **Audio Lip-Sync:** Speaks when you speak! An FFT audio threshold analyzes your microphone volume to trigger "open mouth" AI generations instantly.
- **Screen Recording Mode:** Target your desktop monitor to stylize your screen-share instead of your webcam!
- **AI Green Screen:** Uses Selfie Segmentation to cut you out of your background, perfectly preventing the AI from hallucinating room details into your character.
- **Background Compositing & Bokeh:** Composite your anime avatar over a custom image, or over your raw room background with an adjustable Gaussian Bokeh blur.
- **Buttery Motion Blur:** An adjustable temporal lerp filter smooths the AI's 10 FPS output into a flawless 30 FPS display stream.
- **OBS Virtual Camera Integration:** Sends the generated output directly to OBS Studio as a virtual webcam.

## Running it

``bash
Start_GUI.bat                             # Boot the visual Launcher
venv\Scripts\python.exe smoke_test.py     # Headless test with a synthetic camera
python run_checks.py                      # Verify syntax and API arguments
``

First-time setup on a fresh machine is setup.ps1 — it creates the venv, installs torch and the NVIDIA TensorRT wheel, clones and installs StreamDiffusion, and pins 
umpy<2. equirements-lock.txt records the exact verified versions and why each one matters.

Everything writes to logs/ — ngine-latest.log (with an environment block and full tracebacks) and launcher-latest.log.

## Architecture

The engine is highly decoupled into 4 separate threads, guaranteeing that the GPU TensorRT inference is never starved by CPU processing tasks:

| Thread | Work |
| --- | --- |
| camera_thread | Web/Screen capture → Face Track crop → CLAHE → Selfie Segmentation → Q_IN |
| cmd_listener | A ZeroMQ PULL socket that asynchronously ingests live slider/UI events. |
| main thread | TinyVAE encode → TensorRT UNet inference → TinyVAE decode → Q_OUT |
| postprocess_thread | Unsharp filter → Reinhard colour lock → Saturation → Alpha Compositing → Motion Lerp → ZMQ / OBS Virtual Camera |

Both queues (Q_IN and Q_OUT) hold one frame and the producer **replaces** a stale entry rather than queueing. This allows the camera and post-processing threads to completely overlap the GPU inference time, creating a perfectly parallelized pipeline with zero lag build-up.

## Live Editing

The prompt boxes and all sliders apply while the engine is running. Adjusting sliders sends tiny JSON payloads over the ZeroMQ socket, where the engine instantly hot-loads the new values. 

For the text prompt, the engine re-encodes the CLIP embeddings in-place and copies them into the existing tensor. The new prompt style lands on the very next frame with absolutely zero engine restart required.

## Controls

| Control | What it actually does |
| --- | --- |
| **AI Strength** | --t_index (0 → step 45, 100 → step 12). The single most important dial. A high 	_index starts denoising from an almost clean latent, so the output looks like the raw webcam with a light filter. If the avatar "isn't applying", raise this. |
| **CFG** | --guidance_scale. Adjusts how strictly the AI adheres to the prompt. |
| **Freeze Filter** | Halts generation while you sit perfectly still (measured via structural cosine similarity) to save GPU power and increase visual quality. |
| **Motion Blur** | Blends sequential frames together to create a smooth, cinematic 30 FPS video feed. |
| **Trigger Sensitivities** | Real-time hysteresis thresholds controlling exactly how aggressively the MediaPipe and Audio-FFT algorithms trigger custom prompt injections. |
| **Expression Overrides** | Replaces default expression prompt injections (e.g., turning "smiling" into "sinister grin") on the fly. |

## Engine Cache

TensorRT engines are cached per LoRA in ngines_tinyvae_<lora>_fb1/. The first build for a new LoRA takes 5–15 minutes; the log will say so when no cached engine is found. rame_buffer is pinned to 1 — any other value makes TensorRT expect a different batch and fail with a shape error.

## Troubleshooting

- **Output is garbage / speckled** — non-finite pixels from the model. This happens if PyTorch's mixed precision breaks or NaN tensors escape the UNet.
- **Output looks like a lightly filtered webcam** — AI Strength too low (	_index too high), or *Keep Real Background* is on and only your silhouette is being stylised.
- **Colours strobe frame to frame** — the Reinhard colour lock is on by default to prevent strobe flashing.
- **Exit code 1** — a worker thread died or the camera stopped delivering frames. The traceback is in logs/engine-latest.log.
- **Camera won't open** — the engine scans indices 0–5 and prints the ones that responded. Discord, Zoom, Teams and OBS all hold the device exclusively.

## Directory

- launcher.py — CustomTkinter GUI, spawns and supervises the engine.
- ealtime_video.py — The core headless TensorRT inference engine.
- udio_sync.py — Mic-driven FFT volume processing.
- smoke_test.py — Runs the engine against a synthetic camera.
- loras/ — Drop .safetensors character models here.
- logs/ — Engine, launcher and diagnostic output.
- tuber_settings.json — Saved UI state.
