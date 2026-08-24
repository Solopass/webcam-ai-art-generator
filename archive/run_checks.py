"""Run every check that does not need a GPU.

    python run_checks.py

Exits 0 only if all of them pass. Run this before committing; it is the closest
thing this project has to CI.

Not covered here (needs the RTX): smoke_test.py and diagnose.py.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CHECKS = [
    ("verify.py", "GUI/engine flag handshake, argument parsing, CFG floor"),
    ("test_shutdown.py", "ZMQ command channel, stop path, teardown"),
    ("test_logic.py", "denoise schedule, emotion hysteresis"),
]

results = []
for script, what in CHECKS:
    print(f"\n{'=' * 70}\n  {script} — {what}\n{'=' * 70}")
    proc = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
    results.append((script, proc.returncode == 0))

print(f"\n{'=' * 70}")
for script, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {script}")
failed = [s for s, ok in results if not ok]
if failed:
    print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
    sys.exit(1)
print("\nAll GPU-free checks passed.")
print("Still needs a GPU: smoke_test.py (end-to-end), diagnose.py (per-stage NaN).")
