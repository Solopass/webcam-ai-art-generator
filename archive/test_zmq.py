import zmq, time, cv2, numpy as np
context = zmq.Context()
sock = context.socket(zmq.PUB)
sock.bind("tcp://127.0.0.1:12345")
print("Sender bound.")
while True:
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    sock.send(buffer.tobytes())
    time.sleep(0.03)
