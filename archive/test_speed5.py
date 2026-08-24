import subprocess, sys
res = subprocess.run([sys.executable, "realtime_video_threaded.py", "--camera", "99", "--prompt", "1girl", "--lora", "None"], timeout=15, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
