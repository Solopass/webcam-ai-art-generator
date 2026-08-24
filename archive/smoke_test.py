"""Run the engine headless against a synthetic camera.

Usage:  venv\\Scripts\\python.exe smoke_test.py [seconds]

Exits 0 if the engine reached the READY line and reported an FPS number.
Useful for checking the pipeline without fighting Discord/OBS for the webcam.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 45

venv = os.path.join(HERE, "venv", "Scripts", "python.exe")
python = venv if os.path.exists(venv) else sys.executable

cmd = [
    python, "-u", os.path.join(HERE, "realtime_video.py"),
    "--mock_camera",
    "--prompt", "1girl, masterpiece, anime key visual",
    "--lora", "None",
    "--t_index", "32",
    "--cfg_type", "full",
    "--guidance_scale", "1.4",
    "--delta", "1.0",
    "--cuda_graph",
    "--freeze_threshold", "1.0",
]

print(f"$ {' '.join(cmd)}\n")
proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", bufsize=1)

ready = False
saw_fps = False
deadline = time.time() + SECONDS
try:
    for line in iter(proc.stdout.readline, ''):
        if not line:
            break
        print(line.rstrip().encode('ascii', 'replace').decode('ascii'))
        if "READY" in line:
            ready = True
        if "FPS |" in line:
            saw_fps = True
            break
        if time.time() > deadline:
            print(f"\n[smoke_test] {SECONDS}s budget exhausted.")
            break
finally:
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()

print(f"\n[smoke_test] reached READY: {ready} | reported FPS: {saw_fps}")
sys.exit(0 if (ready and saw_fps) else 1)
