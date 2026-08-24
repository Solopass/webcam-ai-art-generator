"""Integration test for the new shutdown path and the command channel.

Runs the real cmd_listener_thread from realtime_video.py against a PUSH socket
configured exactly like launcher.py's, with no console and no signals involved.
"""
import json
import re
import threading
import time

import zmq

SRC = open("realtime_video.py", encoding="utf-8").read()

# Pull the real listener out of the engine without importing torch/cv2/etc.
fn = re.search(r"^def cmd_listener_thread.*?^(?=def main)", SRC, re.S | re.M).group(0)
STOP = threading.Event()
logged = []
ns = {"STOP": STOP, "log": logged.append}
exec(fn, ns)
cmd_listener_thread = ns["cmd_listener_thread"]

PORT = 45999
state = {"base_prompt": "original", "negative_prompt": "blurry", "prompt_dirty": False}
t = threading.Thread(target=cmd_listener_thread, args=(PORT, state), daemon=True)
t.start()
time.sleep(0.4)

ctx = zmq.Context()
push = ctx.socket(zmq.PUSH)
push.setsockopt(zmq.LINGER, 0)
push.setsockopt(zmq.SNDTIMEO, 0)
push.connect(f"tcp://127.0.0.1:{PORT}")
time.sleep(0.2)

# 1. the launcher's live-prompt payload: both boxes in one message
push.send_string(json.dumps({"prompt": "steampunk workshop",
                             "negative_prompt": "text, watermark"}), flags=zmq.NOBLOCK)
time.sleep(0.3)
assert state["base_prompt"] == "steampunk workshop", state
assert state["negative_prompt"] == "text, watermark", state
assert state["prompt_dirty"] is True
print("PASS  live prompt + negative prompt applied")

# 2. resending the identical text must NOT mark the prompt dirty, or every
#    Enter keypress would burn a CLIP re-encode in the inference loop
state["prompt_dirty"] = False
push.send_string(json.dumps({"prompt": "steampunk workshop",
                             "negative_prompt": "text, watermark"}), flags=zmq.NOBLOCK)
time.sleep(0.3)
assert state["prompt_dirty"] is False, "unchanged text still triggered a re-encode"
print("PASS  unchanged text is a no-op")

# 3. changing only the negative prompt still triggers a re-encode
push.send_string(json.dumps({"prompt": "steampunk workshop",
                             "negative_prompt": "blurry, deformed"}), flags=zmq.NOBLOCK)
time.sleep(0.3)
assert state["prompt_dirty"] is True and state["negative_prompt"] == "blurry, deformed"
print("PASS  negative-prompt-only change is picked up")

# 4. malformed JSON must not kill the listener
push.send_string("{not json", flags=zmq.NOBLOCK)
time.sleep(0.3)
assert t.is_alive(), "listener died on malformed input"
print("PASS  malformed command survived")

# 5. the stop command sets STOP and unwinds the thread
t0 = time.time()
push.send_string(json.dumps({"cmd": "stop"}), flags=zmq.NOBLOCK)
t.join(timeout=5)
took = time.time() - t0
assert STOP.is_set(), "STOP was never set"
assert not t.is_alive(), "listener thread did not exit"
print(f"PASS  stop command honoured in {took*1000:.0f}ms (old path: 8000ms + SIGKILL)")

# 6. the close-hang fix: destroy() returns with a socket still open.
#    ctx.term() here would block forever, which is exactly the old bug.
done = threading.Event()
threading.Thread(target=lambda: (ctx.destroy(linger=0), done.set()), daemon=True).start()
assert done.wait(timeout=5), "context teardown hung with an open socket"
print("PASS  zmq_context.destroy(linger=0) returns with cmd_socket still open")

# 7. a send with no peer must raise, never block the GUI thread
ctx2 = zmq.Context()
orphan = ctx2.socket(zmq.PUSH)
orphan.setsockopt(zmq.LINGER, 0)
orphan.setsockopt(zmq.SNDTIMEO, 0)
t0 = time.time()
blocked = False
try:
    for _ in range(2000):          # overflow the queue with nowhere to send
        orphan.send_string("x", flags=zmq.NOBLOCK)
except zmq.ZMQError:
    pass
if time.time() - t0 > 2.0:
    blocked = True
ctx2.destroy(linger=0)
assert not blocked, "send blocked on a peerless socket"
print(f"PASS  peerless send raises instead of blocking ({(time.time()-t0)*1000:.0f}ms)")

print(f"\nlistener log: {logged}")
print("ALL PASS")
