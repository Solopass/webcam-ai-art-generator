# How to test

Run the app, do the things below, stop it, then double-click **Test_Report.bat**.
It writes `TEST_REPORT.md` — that single file is the only thing to send back.
You don't need to read logs or copy anything.

## The run

1. **START.** Wait until the FPS lines appear in the log box (~20s with a cached
   engine).
2. **Let it sit for a minute.** This is the important one — it's what produces
   the `[Camera]` timing lines that say where the frame budget goes.
3. **Change the Character (LoRA) dropdown.** Watch whether the picture keeps
   moving. It should stay live for 30–60s, then pause about a second.
4. **Move a slider or two** — Sharpness, Saturation, CFG. Just enough to see
   whether they do anything.
5. **Click into the Master Prompt box and press Tab.** Focus should move to the
   next box instead of blanking the picture.
6. **STOP**, then run the report.

## What only you can answer

The logs can prove a command arrived and that code ran. They cannot see the
picture. So the bottom of `TEST_REPORT.md` has a short "What I saw" section —
a line or two there is worth more than the rest of the file:

- Did the stream keep moving during the LoRA swap, or did it freeze?
- Does Tab move between the prompt boxes?
- Did any control look like it did nothing?
- Anything visually off — offset, colour, edges, the OBS output?

## If something goes badly wrong

Stop the app and run the report anyway. Errors, tracebacks and the exit code
are all captured, and a failed run is usually more informative than a clean one.
