# AI Webcam Generator - Development Roadmap

This document outlines the planned features, optimizations, and UI improvements for future releases of the AI Webcam Generator.

## 🎛️ UI & Fine Control (Upcoming Sliders)
- **Face Tracking Sensitivity Slider**: Allow users to tune the exact thresholds for when "smiling", "open mouth", and "closed eyes" trigger, accommodating different lighting and face shapes.
- **Motion Smoothing / Ghosting Slider**: A slider to control the temporal frame blending (`cv2.addWeighted`). Users can dial it up for buttery smooth motion blur, or dial it down to 0 for instant, raw responsiveness.
- **Background Blur (Bokeh) Slider**: Allow users to artificially blur their real room (when `Composite Real Background` is ON) to simulate a DSLR depth-of-field effect behind the AI avatar.
- **Custom Emotion Prompts**: A dedicated UI panel where users can type exactly what text gets injected for specific facial expressions (e.g. changing "smiling" to "sinister grin" or "radiant joy").

## 🚀 Engine & Architecture
- **ControlNet Integration (Depth/Canny)**: Add an optional ControlNet pass to perfectly lock the AI generation to the exact pose and structure of the webcam feed, eliminating hallucinated extra arms or shifting backgrounds.
- **Dynamic Resolution Scaling**: Support compiling TensorRT engines for 768x768 and 1024x1024 resolutions for users with high-end GPUs (e.g., RTX 4090).
- **Audio-Driven Lip Sync**: Map microphone audio waveforms directly to AI mouth shapes (visemes) for incredibly accurate talking, even when the webcam face mesh misses subtle mouth movements.
- **Multi-Face Tracking**: Track multiple people in the webcam feed and independently apply different LoRAs or characters to each person.

## 🎨 Content & Creative Tools
- **LoRA Hot-Swapping**: Allow changing the loaded Character LoRA directly from the GUI without needing to restart the TensorRT engine.
- **GIF & WebP Export**: Add smaller, web-friendly export options for the Snapshot and Replay tools.
- **Twitch / OBS WebSocket Integration**: Allow Twitch chat commands (e.g., `!cyberpunk`) to automatically trigger the "Randomize Style" button in real-time.
