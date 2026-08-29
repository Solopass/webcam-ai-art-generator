"""Run every check that does not need a GPU.

    python run_checks.py

Exits 0 only if all pass. Run before committing.
Not covered here (needs the RTX): an actual engine build or inference.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = [
    ("test_engine_logic.py", "denoise schedule, command coverage, startup order"),
    ("test_presets.py", "presets, history, settings round-trip"),
]

results = []
for script, what in CHECKS:
    print(f"\n{'=' * 70}\n  {script} — {what}\n{'=' * 70}")
    results.append((script, subprocess.run([sys.executable, os.path.join(HERE, script)],
                                           cwd=HERE).returncode == 0))

print(f"\n{'=' * 70}")
for script, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {script}")
failed = [s for s, ok in results if not ok]
if failed:
    print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
    sys.exit(1)
print("\nAll GPU-free checks passed.")
