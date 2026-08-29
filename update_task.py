import os

path = r'C:\Users\cohen\.gemini\antigravity\brain\4349b2e6-5ad9-4617-897a-f060b1151803\task.md'
with open(path, 'w', encoding='utf-8') as f:
    f.write('''- [x] Update ealtime_video.py ThreadedCamera to loop .mp4 files automatically.
- [x] Update ealtime_video.py to pass the full HD frame and crop coordinates (x, y, size) through Q_IN and Q_OUT.
- [x] Update ealtime_video.py postprocess_thread to composite the 512x512 stylized AI face back into the full HD frame using Alpha, Opacity, and Blend Mode.
- [x] Update ealtime_video.py cmd_listener_thread to receive fx_opacity and fx_blend_mode state updates.
- [x] Update launcher.py UI to add "Load Video" button, "Export Offline VFX" button, and VFX Compositing sliders/dropdowns.
- [x] Create process_video.py as a standalone script that runs the AI synchronously, applying the exact same VFX compositing logic, and writing to an .mp4 using cv2.VideoWriter.''')
