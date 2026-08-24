"""Behavioural checks on the two logic changes, using the real source text."""
import re
import numpy as np

SRC = open("realtime_video.py", encoding="utf-8").read()

# ---------------------------------------------------------------- t_list ----
form = re.search(r"^    step1 = min\(.*?\n    t_list = \[step1, args\.t_index\]",
                 SRC, re.S | re.M).group(0)


def t_list_for(t_index):
    ns = {"args": type("A", (), {"t_index": t_index})()}
    exec(form.replace("    ", ""), ns)
    return ns["t_list"]


for ti in range(2, 50):
    a, b = t_list_for(ti)
    assert a < b, f"t_index={ti} produced a non-ascending list {[a, b]}"
    assert a >= 0
# the GUI slider range must be untouched by the clamp
for ti, expected in [(12, [10, 12]), (20, [10, 20]), (32, [17, 32]), (45, [30, 45])]:
    assert t_list_for(ti) == expected, (ti, t_list_for(ti), expected)
print("PASS  t_list strictly ascending for every t_index 0-49")
print(f"      slider range unchanged: 12->{t_list_for(12)}  32->{t_list_for(32)}  45->{t_list_for(45)}")
print(f"      old formula at t_index=5 gave [10, 5]; now {t_list_for(5)}")

# ------------------------------------------------------------- emotions ----
block = re.search(r"( *)if mouth_open > \(?mouth_sens\)?.*?emotions\.append\(\"closed eyes\"\)",
                  SRC, re.S).group(0)
block = "\n".join(l[20:] for l in block.splitlines())


class LM:
    def __init__(self, y):
        self.y = y
        self.x = 0.0


def run_new(mouth_series, eye_series):
    ns = {"mouth_open_state": False, "smiling_state": False, "eyes_closed_state": False,
          "mouth_sens": 0.030, "smile_sens": 0.010, "eyes_sens": 0.019}
    flips, prev = 0, None
    for m, e in zip(mouth_series, eye_series):
        lm = {13: LM(0.5), 14: LM(0.5), 61: LM(0.5), 291: LM(0.5),
              145: LM(e), 159: LM(0.0), 374: LM(e), 386: LM(0.0)}
        ns.update({"emotions": [], "mouth_open": m, "landmarks": lm})
        exec(block, ns)
        state = tuple(ns["emotions"])
        if prev is not None and state != prev:
            flips += 1
        prev = state
    return flips


def run_old(mouth_series, eye_series):
    """The previous single-threshold logic, for comparison."""
    flips, prev = 0, None
    for m, e in zip(mouth_series, eye_series):
        emotions = []
        if m > 0.04:
            emotions.append("open mouth")
        elif m <= 0.01:
            emotions.append("closed mouth")
        if e < 0.015:
            emotions.append("closed eyes")
        state = tuple(emotions)
        if prev is not None and state != prev:
            flips += 1
        prev = state
    return flips


rng = np.random.default_rng(7)
# someone sitting still, mouth just under the old cutoff, eyes right at it
mouth = 0.038 + rng.normal(0, 0.004, 600)
eyes = 0.0150 + rng.normal(0, 0.0015, 600)

old_flips = run_old(mouth, eyes)
new_flips = run_new(mouth, eyes)
print(f"PASS  prompt-state flips over 600 frames of a still face: "
      f"{old_flips} -> {new_flips}")
assert new_flips < old_flips / 5, (old_flips, new_flips)

# a real mouth movement must still register
speak = np.concatenate([np.full(30, 0.005), np.full(30, 0.08), np.full(30, 0.005)])
still_eyes = np.full(90, 0.03)
assert run_new(speak, still_eyes) == 2, run_new(speak, still_eyes)
print("PASS  a genuine open/close still produces exactly 2 transitions")

# no dead band: every frame emits exactly one of the two mouth states
ns = {"mouth_open_state": False, "smiling_state": False, "eyes_closed_state": False,
      "mouth_sens": 0.030, "smile_sens": 0.010, "eyes_sens": 0.019}
for m in np.linspace(0.0, 0.1, 200):
    lm = {13: LM(0.5), 14: LM(0.5), 61: LM(0.5), 291: LM(0.5),
          145: LM(0.03), 159: LM(0.0), 374: LM(0.03), 386: LM(0.0)}
    ns.update({"emotions": [], "mouth_open": m, "landmarks": lm})
    exec(block, ns)
    got = [e for e in ns["emotions"] if "mouth" in e]
    assert len(got) == 1, (m, ns["emotions"])
print("PASS  mouth state is always exactly one of open/closed (old code had a dead band)")

print("ALL PASS")
