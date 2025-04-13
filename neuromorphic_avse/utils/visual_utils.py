import cv2
import numpy as np

def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        features.append(resized.flatten())
        if len(features) >= 128:
            break
    cap.release()
    return np.array(features)
