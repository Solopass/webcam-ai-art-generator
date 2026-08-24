---
title: "I Turned My Webcam Into a Real-Time AI Art Machine"
date: 2026-08-23
tags: [ai, stable-diffusion, tensorrt, python, gpu]
---

Point a webcam at yourself. Type "studio ghibli style, lush nature, watercolor."
Press Enter. Two hundred milliseconds later you're looking at a moving painting
of yourself, and it keeps up as you move.

That's the whole pitch. The code is on GitHub at
[Solopass/webcam-ai-art-generator](https://github.com/Solopass/webcam-ai-art-generator),
it runs on a single RTX 3080 Ti, and it's fast enough to use as a live camera in
OBS or Discord.

## What it actually does

The core loop takes a 512×512 crop of your webcam feed and runs it through
Stable Diffusion as an image-to-image transform, about ten times a second. On
top of that:

- **Live prompts.** Type in the box and hit Enter; the change lands on the next
  frame. No restart, no reload.
- **A style randomizer.** Ctrl+R rolls one of sixteen presets — cyberpunk neon,
  claymation, stained glass, pencil sketch, vaporwave. It's the single most fun
  key on the keyboard and the reason I stopped calling this a VTuber tool.
- **Face tracking.** The crop follows your face around the room with a smoothed
  pan, so you don't have to sit rigidly in frame.
- **A face-driven prompt.** MediaPipe's face mesh watches your mouth, eyes and
  smile, and quietly appends "open mouth", "smiling" or "closed eyes" to the
  prompt. The generated character blinks when you blink.
- **Recording.** One button records straight to H.264 MP4 that Discord will
  actually play. Ctrl+S dumps the last five seconds from a rolling buffer, F12
  grabs a still.
- **OBS output.** It publishes to a virtual camera at 1024×1024, so anything
  that takes a webcam takes this.

## How it's put together

Three pieces of the puzzle make real-time diffusion possible at all, and none of
them are mine:

[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) restructures
the denoising loop for streaming input. **LCM-LoRA** collapses fifty scheduler
steps down to one or two. **TAESD** ("TinyVAE") replaces the standard VAE with
something roughly a hundred times smaller, which matters enormously when you're
encoding and decoding thirty times a second. Everything then gets compiled to a
**TensorRT** engine, which is where the last big multiplier comes from.

What I built around that is the part that makes it usable: the GUI, the capture
and compositing pipeline, and the plumbing that lets you change things while
it's running.

The engine runs three threads connected by single-slot queues:

```
camera_thread     capture → face-tracked crop → denoise → selfie mask
                  → grey-screen composite → face mesh          → Q_IN
main thread       TinyVAE encode → TensorRT UNet ×2 → decode   → Q_OUT
postprocess       sharpen → saturate → composite → 30 FPS lerp → ZMQ / OBS
```

The queues hold exactly one frame, and the producer *replaces* a stale frame
rather than skipping work. That detail is load-bearing. I tried the obvious
optimisation once — skip the expensive MediaPipe pass when the queue is already
full — and throughput went *down*, because it serialised the camera wait and the
CPU preprocessing with the GPU instead of overlapping them. The producer has to
run ahead of the consumer even when most of what it produces gets thrown away.

The GUI is a separate process. Frames come back over ZMQ as JPEGs on a socket
with `CONFLATE` set, so the preview can never build up a backlog of stale
frames — it always gets the newest one or nothing. A second ZMQ socket goes the
other way, carrying prompt changes and the shutdown command as JSON. Separating
them means a slow or wedged GUI can't stall the engine.

## The one dial that matters

If you build something like this, here's the thing I'd want to have known
earlier.

Everyone reaches for CFG — the guidance scale — when the output doesn't look
stylised enough. In an image-to-image loop, CFG is not the important dial.
**Where you start denoising is.**

StreamDiffusion picks a starting point from the scheduler's timestep list.
Timesteps run from most-noisy to least-noisy, so a *high* index means you start
from an almost-clean latent — and the model, given something that already looks
finished, politely leaves it alone. You get your webcam with a light filter over
it and no amount of CFG will fix that, because the model isn't being asked to do
much in the first place.

Drop that index and the model has real noise to resolve, so it actually commits
to the prompt. In the UI I've exposed it as a slider called **AI Strength**,
which maps 0–100 onto the step index backwards, because "AI Strength" is what it
does and "t_index" is what it's called.

The second thing worth knowing: the number of steps and the CFG mode together
determine the UNet batch size, and the TensorRT engine is compiled for a
*specific* batch size. Two steps with standard classifier-free guidance means
four UNet passes per frame — so switching CFG mode isn't a runtime toggle, it's
a fifteen-minute engine rebuild. Worth knowing before you casually change a
constant.

## Speed, honestly

On a 3080 Ti it runs at about **10 FPS of actual inference**, which the
postprocess thread interpolates up to a smooth ~30 FPS output. Roughly 70ms of
each frame is the UNet.

That 10 is a choice, not a wall. Two steps at full CFG buys noticeably better
prompt adherence than one step, and I'd rather have that than the frame rate.
There's a middle setting (`cfg_type="initialize"`, three UNet passes instead of
four) that would claw back about a quarter of the time, and CUDA graphs are
another ten percent. Both are one rebuild away if I decide I want them.

The interpolation is worth a note too. Blending toward each new frame at 30 FPS
instead of showing it the instant it arrives costs about two frames of latency,
and it's absolutely worth it — the difference between "AI video" and "a
slideshow" is mostly just whether the motion is continuous.

## Running it

```
.\setup.ps1        # venv, torch, StreamDiffusion, TensorRT wheel
.\Start_GUI.bat
```

Two warnings. The first launch compiles TensorRT engines and takes **5–15
minutes** with no visible progress — this is normal and the log says so. And
numpy has to stay below 2.x, because numpy 2 breaks the onnxruntime build this
depends on for the ONNX export. That one is fatal only when you rebuild an
engine, which is the worst possible time to find out, so the engine now prints
its numpy and onnxruntime versions at startup.

## Where it's going

It works, it's stable, and it's had a proper hardening pass — everything logs to
disk with full tracebacks, there's a headless smoke test, and a small suite of
checks runs without a GPU at all. Genuinely more disciplined than most things I
build for fun.

Next up: wiring those checks into CI, and probably chasing that `initialize`
rebuild to see whether 13 FPS feels meaningfully different from 10.

If you try it, I'd like to see what you make with it.
