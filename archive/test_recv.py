import zmq, cv2, numpy as np, time
context = zmq.Context()
sock = context.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:12345")
sock.setsockopt_string(zmq.SUBSCRIBE, "")
print("Receiver connected.")
frames_received = 0
start = time.time()
while time.time() - start < 3:
    try:
        data = sock.recv(zmq.NOBLOCK)
        frames_received += 1
    except zmq.Again:
        time.sleep(0.01)
print(f"Received {frames_received} frames.")
