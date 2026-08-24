import cv2, numpy as np, time

def color_transfer(source, target):
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)
    tgt_std = np.where(tgt_std == 0, 1e-5, tgt_std)
    l = (tgt_lab[:, :, 0] - tgt_mean[0][0]) * (src_std[0][0] / tgt_std[0][0]) + src_mean[0][0]
    a = (tgt_lab[:, :, 1] - tgt_mean[1][0]) * (src_std[1][0] / tgt_std[1][0]) + src_mean[1][0]
    b = (tgt_lab[:, :, 2] - tgt_mean[2][0]) * (src_std[2][0] / tgt_std[2][0]) + src_mean[2][0]
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    transfer = cv2.merge([l, a, b]).astype("uint8")
    return cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)

img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

start = time.time()
for _ in range(50):
    color_transfer(img1, img2)
end = time.time()
print(f"Time per frame: {(end-start)/50 * 1000:.1f} ms")
