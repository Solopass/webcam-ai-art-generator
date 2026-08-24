import subprocess, sys
try:
    res = subprocess.run([sys.executable, "launcher.py"], timeout=3, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)
except subprocess.TimeoutExpired as e:
    print("UI ran for 3 seconds.")
    if hasattr(e, 'stdout'):
        print("STDOUT:", e.stdout)
    if hasattr(e, 'stderr'):
        print("STDERR:", e.stderr)
