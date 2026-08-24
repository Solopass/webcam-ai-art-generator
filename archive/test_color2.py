import cv2, numpy as np, time

def color_transfer_fast(source, target):
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = np.mean(src_lab, axis=(0, 1))
    tgt_mean = np.mean(tgt_lab, axis=(0, 1))
    tgt_lab += (src_mean - tgt_mean)
    np.clip(tgt_lab, 0, 255, out=tgt_lab)
    return cv2.cvtColor(tgt_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

start = time.time()
for _ in range(50):
    color_transfer_fast(img1, img2)
end = time.time()
print(f"Time per frame: {(end-start)/50 * 1000:.1f} ms")
