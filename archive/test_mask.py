import cv2, mediapipe as mp, numpy as np

cap = cv2.VideoCapture(0)
segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

ret, frame = cap.read()
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_rgb = cv2.resize(frame_rgb, (256, 256))
    results = segmenter.process(small_rgb)
    if results.segmentation_mask is not None:
        print("Mask max:", np.max(results.segmentation_mask))
        print("Mask min:", np.min(results.segmentation_mask))
    else:
        print("No mask!")
