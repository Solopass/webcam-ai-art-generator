import cv2, mediapipe as mp, numpy as np, time

def test_numpy_speed():
    frame_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    soft_mask = np.random.rand(512, 512).astype(np.float32)
    
    start = time.time()
    for _ in range(50):
        res = (frame_rgb * soft_mask[..., None]).astype(np.uint8)
    end = time.time()
    print(f"NumPy float64 multiply: {(end-start)/50 * 1000:.1f} ms")

    start = time.time()
    for _ in range(50):
        res = (frame_rgb.astype(np.float32) * soft_mask[..., None]).astype(np.uint8)
    end = time.time()
    print(f"NumPy float32 multiply: {(end-start)/50 * 1000:.1f} ms")

test_numpy_speed()
