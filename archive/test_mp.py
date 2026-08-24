import mediapipe as mp, cv2, numpy as np
segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
frame = np.zeros((512, 512, 3), dtype=np.uint8)
results = segmenter.process(frame)
print("Mask type:", type(results.segmentation_mask))
