# Pinned to mediapipe==0.10.30 — 1.0.1 crashes on macOS (GPU/Metal delegate bug)

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

base_options = BaseOptions(model_asset_path="hand_landmarker.task",
                           delegate=BaseOptions.Delegate.CPU)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        h, w, _ = frame.shape
        for lm in landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)


    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
