# HANDOFF — TensorRT AI VTuber Studio

Last updated 2026-08-23. Read this document before contributing or changing code. The architecture is highly deliberate to squeeze maximum performance out of the StreamDiffusion GPU loop.

---

## 1. What this is and where it stands

A real-time AI VTuber Engine: StreamDiffusion + TensorRT turning a webcam or desktop feed into a live stylised video stream at ~10 FPS, with real-time UI tuning via ZMQ, Audio Lip-Sync, and Face Tracking.

**It works flawlessly.** Measured on an RTX 3080 Ti:

`
[Engine] 10.0 FPS | wait 25.2ms  infer 70.3ms
[Engine] output 30.0 fps, 8.4ms work/frame (composite off, smoothing on)
`

infer 70ms is the whole ceiling and it is a **deliberate trade**: two denoise steps at cfg_type="full" is four UNet passes per frame. The postprocess thread interpolates the 10 FPS inference stream up to a buttery 30 FPS output, which costs about two frames of latency. 

`
Start_GUI.bat                             # Boot the visual Launcher
python run_checks.py                      # all GPU-free checks, ~10s
venv\Scripts\python.exe smoke_test.py     # end-to-end, synthetic camera
`

Everything writes to logs/: ngine-latest.log (environment block, every engine line, full traceback on death) and launcher-latest.log. 

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
- **Face & Audio Tracking**: Integrated MediaPipe and PyAudio FFT to live-inject expressions like "open mouth" into the prompt.
- **Sensitivities & Overrides**: Allowed real-time tuning of trigger thresholds and replacement expressions via the UI.
- **Background Compositing**: Selfie segmenter cleanly drops the user onto custom backgrounds with adjustable Bokeh blurs.
- **Screen Sharing**: Safely skips Selfie Segmentation on desktop captures so the whole screen is stylized.

## 5. Next Steps / V2.0 Ideas
- **ControlNet Depth Integration**: The only way to perfectly lock the AI to the exact structural lines of a drawing or face. This requires entirely rewriting the TensorRT engine builder script, as streamdiffusion does not natively support ControlNet TRT engines.
