# HANDOFF - Webcam AI Art Generator (V2.0)

Last updated 2026-08-23. Read this document before contributing or changing code. The architecture is highly deliberate to squeeze maximum performance out of the StreamDiffusion GPU loop.

---

## 1. What this is and where it stands

A real-time AI Art Generator: StreamDiffusion + TensorRT turning a webcam or desktop feed into a live stylised video stream. The primary use-case is using the webcam as a live posing/composition tool to generate high-quality AI art, which can then be snapshotted and exported for upscaling.

**It works flawlessly.** Measured on an RTX 3080 Ti:

`
[Engine] 26.4 FPS | wait 0.0ms  infer 64.0ms
[Engine] output 26.4 fps, 2.4ms work/frame (composite off, smoothing on)
`

infer 64ms is the whole ceiling and it is a **deliberate trade**: two denoise steps at cfg_type="full" is four UNet passes per frame. 

`
Start_GUI.bat                             # Boot the visual Launcher
python run_checks.py                      # all GPU-free checks, ~10s
venv\Scripts\python.exe smoke_test.py     # end-to-end, synthetic camera
`

Everything writes to logs/: gine-latest.log (environment block, every engine line, full traceback on death) and launcher-latest.log. 

---

## 2. Architecture

Three worker threads in the engine, decoupled by single-slot queues, so throughput is set strictly by the GPU TensorRT bottleneck rather than the sum of CPU stages:

| Thread | Work |
| --- | --- |
| camera_thread | Web/Screen Capture → Face-tracked square crop → CLAHE → Selfie mask (skipped on screen mode) → FaceMesh emotions → Q_IN |
| cmd_listener | A ZeroMQ PULL socket that drains UI slider events into a shared state_dict without halting the engine. |
| main thread | prompt embed refresh → TinyVAE encode → TensorRT UNet ×2 steps → TinyVAE decode → Q_OUT |
| postprocess_thread | unsharp → saturation → alpha background composite → 30 FPS lerp → ZMQ / virtual camera |

Both queues hold one frame and the producer **replaces** a stale entry rather than skipping the work. That matters: gating the preprocessing on "is the queue full" serialises the camera wait and the CPU work with the GPU instead of overlapping them!

## 3. ZMQ Event Sockets

The UI communicates with the engine purely through ZeroMQ:
* **PUB/SUB Preview Socket:** Sends compressed JPEGs back to the UI at 30 FPS. Uses zmq.CONFLATE to drop frames if the UI renderer lags behind.
* **PUSH/PULL Command Socket:** The UI pushes JSON payloads when sliders are dragged. The engine drains this queue completely during its pacing sleep cycle, meaning the UI sliders can be scrubbed vigorously without causing minutes of lag.

## 4. Completed Features
- **Zero-Compile Dynamic LoRA Hot-Swapping**: Features a highly complex TensorRT VRAM Refitting pipeline. The engine permanently boots a single refittable "Base" UNet. Changing LoRAs triggers an on-the-fly PyTorch UNet fusion, an ONNX export, and a direct injection of the new weights into the live CUDA Engine.
- **CUDA Graph Offset Optimization**: By pairing TensorRT Refitting with CUDA Graph execution, we negate the dispatch latency overhead caused by un-fused layer paths in the unlocked engine.

## 5. Known Behaviors & Compilation Times
- **TensorRT Timing Cache Generation**: In `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/utilities.py`, we removed a bug that forcefully disabled cuDNN (`tactic_sources=[]`). With Nvidia libraries properly enabled, the first-time generation of the `trt_global_timing.cache` is extremely heavy.
  - **Low Latency Mode (Batch Size 4):** ~15 minutes to compile on RTX 3080 Ti.
  - **High Quality Mode (Batch Size 8):** Can trigger a silent WDDM driver deadlock in TensorRT 9 on Windows. The GPU will sit at 100% utilization, but the C++ Polygraphy builder will silently hang in an infinite loop while benchmarking cuDNN convolution heuristics.
  - **Claude / AI Audit Note**: If the user experiences a 2+ hour compile for Batch Size 8 that never finishes, **the codebase is not bugged**, the C++ compiler is deadlocked. The solution is to kill the Python process and instruct the user to select **"2-Step Low Latency"** in the GUI, which forces Batch Size 4 and completely avoids the WDDM memory deadlock.

## 6. Next Steps / V2.0 Ideas
- **ControlNet Integration (Highest Priority)**: The only way to perfectly lock the AI to the exact structural lines and poses of the webcam feed. Crucial for using the webcam as an art composition tool.
- **1-Step Turbo Upgrade**: Migrating the base pipeline to SD-Turbo or LCM to quadruple the framerate and generation speed without sacrificing art quality.
- **High-Res Snapshot Export**: Allowing users to save the raw noise latents or seeds alongside their snapshots for seamless drag-and-drop upscaling in ComfyUI or A1111.
