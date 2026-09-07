import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened(): raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

