"""Turn a test session into one small file.

    python session_report.py            # after you have run and stopped the app

Runs the GPU-free checks, reads the newest engine and launcher logs, and writes
TEST_REPORT.md — a compact summary of what actually happened. The point is that
you do not have to read logs, copy anything, or describe what you saw: run the
app, stop it, run this, and hand over the one file.

Everything here is read-only apart from writing TEST_REPORT.md.
"""
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
OUT = os.path.join(HERE, "TEST_REPORT.md")
MAX_SAMPLE = 6          # keep the report small enough to read in one go


def newest(*names):
    best = None
    for n in names:
        p = os.path.join(LOGS, n)
        if os.path.exists(p) and (best is None or os.path.getmtime(p) > os.path.getmtime(best)):
            best = p
    return best


def read(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def this_session(text, *markers):
    """Trim to the most recent run.

    launcher-latest.log accumulates across launches, so reporting on the whole
    file mixed sessions together: errors from an hour ago showed up beside a
    clean run, and the shutdown section listed two exits. Cut everything before
    the last start marker.
    """
    cut = -1
    for m in markers:
        i = text.rfind(m)
        if i > cut:
            cut = i
    return text[cut:] if cut >= 0 else text


def stamp(line):
    m = re.match(r"(\d{2}:\d{2}:\d{2})", line)
    return m.group(1) if m else ""


def secs(hms):
    try:
        h, m, s = (int(x) for x in hms.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def fmt_gap(a, b):
    sa, sb = secs(a), secs(b)
    if sa is None or sb is None:
        return "?"
    d = sb - sa
    if d < 0:
        d += 24 * 3600          # crossed midnight
    return f"{d}s"


def med(values):
    return statistics.median(values) if values else None


# ----------------------------------------------------------------- checks --
def run_checks():
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "run_checks.py")],
                           cwd=HERE, capture_output=True, text=True, timeout=300)
    except Exception as e:
        return f"could not run: {e}", []
    fails = [l.strip() for l in (r.stdout + r.stderr).splitlines()
             if l.strip().startswith("FAIL") or "AssertionError" in l
             or "EXTRACTION FAILED" in l]
    return ("all passed" if r.returncode == 0 else "FAILED"), fails[:12]


# -------------------------------------------------------------- log mining --
def camera_lines(log):
    """[Camera] 10.1 fps produced | 98.4ms/frame = read 2.1  face 3.0 ..."""
    out = []
    for line in log.splitlines():
        m = re.search(r"\[Camera\] ([\d.]+) fps produced \| ([\d.]+)ms/frame = (.*)", line)
        if m:
            stages = dict(re.findall(r"([a-z]+) ([\d.]+)", m.group(3)))
            out.append((float(m.group(1)), float(m.group(2)),
                        {k: float(v) for k, v in stages.items()}))
    return out


def fps_lines(log):
    out = []
    for line in log.splitlines():
        m = re.search(r"([\d.]+) FPS \| wait ([\d.]+)ms\s+infer ([\d.]+)ms", line)
        if m:
            out.append(tuple(float(g) for g in m.groups()))
    return out


def swaps(log):
    """Reconstruct each LoRA change from its log lines, with real timings."""
    events, cur = [], None
    for line in log.splitlines():
        t = stamp(line)
        if "LoRA hot-swap requested" in line or "LoRA Refresh" in line:
            cur = {"t0": t, "line": line.split("] ", 1)[-1].strip(),
                   "prep": None, "refit": None, "done": None, "failed": None}
            events.append(cur)
        elif cur is None:
            continue
        elif "Refitting TensorRT Engine" in line:
            cur["prep"] = t
        elif "Dynamic Refit Complete" in line:
            cur["refit"] = t
        elif "FAILED during" in line:
            cur["failed"] = line.split("] ", 1)[-1].strip()
        elif "hot-swap complete" in line:
            cur["done"] = t
            cur = None
    return events


NOISE = ("FPS |", "[Camera]", "Saved high-res snapshot")
# Third-party chatter that is always present and never actionable. Listing it
# every run trains you to ignore the section that matters.
BENIGN = ("absl::InitializeLog", "WARNING: All log messages before",
          "TensorFloat-32", "oneDNN custom operations")


def problems(log):
    pat = re.compile(r"FATAL|Traceback|Error|error|skipped|WARNING|failed|FAILED|"
                     r"did not stop|abandoning|wraps|Could not")
    c = Counter()
    for line in log.splitlines():
        if any(n in line for n in NOISE) or any(b in line for b in BENIGN):
            continue
        if pat.search(line):
            c[re.sub(r"^\d{2}:\d{2}:\d{2} ", "", line).strip()[:150]] += 1
    return c


# ------------------------------------------------------------------ report --
def main():
    eng_p = newest("engine-latest.log")
    lau_p = newest("launcher-latest.log")
    eng = this_session(read(eng_p), "[Engine] Engine starting", "--- environment ---")
    lau = this_session(read(lau_p), "=== Booting AI VTuber Engine ===")
    L = []
    a = L.append

    a("# Test report")
    a("")
    a(f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
      f"Engine log: `{os.path.basename(eng_p) if eng_p else 'none found'}`.")
    a("")

    status, fails = run_checks()
    a(f"## Offline checks: **{status}**")
    if fails:
        a("")
        for f in fails:
            a(f"- `{f}`")
    a("")

    # --- the producer breakdown, which is the number I am waiting on -------
    cams = camera_lines(eng)
    a("## Producer cost per frame")
    a("")
    if not cams:
        a("No `[Camera]` lines. Either the engine did not run for 5+ seconds, "
          "or this build predates the instrumentation.")
    else:
        keys = ("read", "face", "prep", "seg", "mesh", "cond")
        a(f"{len(cams)} samples. Median of each stage, in ms:")
        a("")
        a("| " + " | ".join(("produced fps", "total") + keys) + " |")
        a("|" + "---|" * (len(keys) + 2))
        row = [f"{med([c[0] for c in cams]):.1f}", f"{med([c[1] for c in cams]):.1f}"]
        stage_med = {}
        for k in keys:
            vals = [c[2].get(k, 0.0) for c in cams]
            stage_med[k] = med(vals) or 0.0
            row.append(f"{stage_med[k]:.1f}")
        a("| " + " | ".join(row) + " |")
        a("")
        worst = max(stage_med, key=stage_med.get)
        total = med([c[1] for c in cams]) or 0.0
        a(f"Dominant stage: **{worst}** at {stage_med[worst]:.1f}ms "
          f"({stage_med[worst] / total * 100:.0f}% of the producer's frame).")
    a("")

    # --- throughput --------------------------------------------------------
    f = fps_lines(eng)
    a("## Throughput")
    a("")
    if not f:
        a("No FPS lines — the engine did not reach steady state.")
    else:
        mf, mw, mi = med([x[0] for x in f]), med([x[1] for x in f]), med([x[2] for x in f])
        a(f"{len(f)} samples. Median **{mf:.1f} FPS** | wait {mw:.1f}ms | infer {mi:.1f}ms.")
        a("")
        if mw > 5:
            a(f"Starved {mw:.1f}ms per frame waiting on the producer. "
              f"If the producer got under {mi:.0f}ms this would be "
              f"~{1000 / mi:.1f} FPS.")
        else:
            a("Not starved — the loop is inference-bound, which is the goal.")
    a("")

    # --- LoRA swaps --------------------------------------------------------
    sw = swaps(eng)
    a("## LoRA swaps / refreshes")
    a("")
    if not sw:
        a("None in this session. (To test one: start, wait for FPS lines, then "
          "change the Character dropdown or press ↻ Apply Strength.)")
    for s in sw[:MAX_SAMPLE]:
        prep = fmt_gap(s["t0"], s["prep"]) if s["prep"] else "never reached"
        refit = fmt_gap(s["prep"], s["refit"]) if s["prep"] and s["refit"] else "?"
        whole = fmt_gap(s["t0"], s["done"]) if s["done"] else "NO COMPLETION LINE"
        a(f"- `{s['line'][:80]}`")
        a(f"  - prep (off-thread): {prep} · refit (freezes output): {refit} · total: {whole}")
        if s["failed"]:
            a(f"  - **failed:** {s['failed'][:120]}")
        if not s["done"]:
            a("  - **no completion line — the LoRA dropdown would stay greyed out**")
    a("")

    # --- anything that looked wrong ---------------------------------------
    a("## Warnings and errors")
    a("")
    pr = problems(eng) + problems(lau)
    if not pr:
        a("None.")
    for line, count in pr.most_common(14):
        a(f"- {'×%d ' % count if count > 1 else ''}`{line}`")
    a("")

    # --- how it ended ------------------------------------------------------
    a("## Shutdown")
    a("")
    tail = [l for l in lau.splitlines()
            if "Engine exited" in l or "Engine Shut Down" in l
            or "force-stopped" in l or "Still shutting down" in l]
    for l in tail[-4:]:
        a(f"- `{l.strip()[:150]}`")
    if not tail:
        a("No shutdown line — the app may still be running.")
    a("")

    a("## What I saw (fill this in — it is the half the logs cannot show)")
    a("")
    for q in ("Did the picture keep moving during a LoRA swap?",
              "Does Tab move between the prompt boxes now?",
              "Did any slider look like it did nothing?",
              "Anything visually wrong (offset, colour, edges)?",
              "Anything else"):
        a(f"- **{q}** ")
    a("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n... written to {OUT}")


if __name__ == "__main__":
    main()
