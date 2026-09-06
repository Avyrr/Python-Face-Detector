# Blu Oloquina

import cv2
from ultralytics import YOLO

#Face Detector model. :>
import cv2
from ultralytics import YOLO

# Loads the Face Detector
model = YOLO("yolov8n-face.pt")

# Opens the webcam
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not read camera.")
        break

    # Unmirrors the camera
    frame = cv2.flip(frame, 1)

    # Detects the faces
    results = model(
        frame,
        conf=0.60,
        imgsz=640,
        verbose=False
    )

    face_count = 0

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(box.conf[0])

            face_count += 1

            # face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            text = f"Face {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # this tells you how many faces are detected ( pretty cool tbh )
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Displays the camera.
    cv2.imshow("YOLO Face Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()