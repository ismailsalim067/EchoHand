# Pinned to mediapipe==0.10.30 — 1.0.1 crashes on macOS (GPU/Metal delegate bug)

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import math

def calculate_angle(p_before, p_joint, p_after):
    v1 = (p_before.x - p_joint.x, p_before.y - p_joint.y, p_before.z - p_joint.z)
    v2 = (p_after.x - p_joint.x, p_after.y - p_joint.y, p_after.z - p_joint.z)
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    len_v1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    len_v2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    cos_angle = dot / (len_v1 * len_v2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

base_options = BaseOptions(model_asset_path="hand_landmarker.task",
                           delegate=BaseOptions.Delegate.CPU)  # GPU delegate crashes on macOS — do not remove
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

fingers = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

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
        for finger_name, finger_points in fingers.items():
            for i in range(len(finger_points) - 2):
                before = finger_points[i]
                joint = finger_points[i + 1]
                after = finger_points[i + 2]
                angle = calculate_angle(landmarks[before], landmarks[joint], landmarks[after])
                print(f"{finger_name}: before={before}, joint={joint}, after={after}, angle={angle}")


    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()