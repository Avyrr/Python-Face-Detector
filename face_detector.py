# Blu Oloquina
# YOLO Face Detector + Hand Sign / Number Detector (MediaPipe Tasks API)

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

FACE_MODEL_PATH = "yolov8n-face.pt"
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Download the hand model if it is not here yet
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand model (one time only)...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    print("Done.")

if not os.path.exists(FACE_MODEL_PATH):
    print(f"Error: {FACE_MODEL_PATH} not found. Put it in the same folder as this script.")
    exit()

# Face detector
face_model = YOLO(FACE_MODEL_PATH)

# Hand tracker
hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Lines that connect the 21 hand points
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---------------------------------------------------------
# Hand gesture helpers
# ---------------------------------------------------------

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_finger_states(pts):
    """
    Returns [thumb, index, middle, ring, pinky]
    1 = finger is extended, 0 = finger is curled.
    pts = list of 21 (x, y) pixel points.
    """
    hand_size = dist(pts[0], pts[9])

    thumb = int(
        dist(pts[4], pts[17]) > dist(pts[3], pts[17])
        and dist(pts[4], pts[5]) > 0.45 * hand_size
    )

    fingers = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(int(dist(pts[tip], pts[0]) > dist(pts[pip], pts[0])))

    return [thumb] + fingers


def classify_gesture(states, pts):
    hand_size = dist(pts[0], pts[9])
    thumb, index, middle, ring, pinky = states

    # OK sign
    if dist(pts[4], pts[8]) < 0.25 * hand_size and middle and ring and pinky:
        return "OK Sign"

    # Thumb only
    if states == [1, 0, 0, 0, 0]:
        dx = pts[4][0] - pts[2][0]
        dy = pts[4][1] - pts[2][1]
        if abs(dy) > abs(dx):
            return "Thumbs Up" if dy < 0 else "Thumbs Down"
        return "Thumb Out"

    gestures = {
        (0, 0, 0, 0, 0): "Fist (Number 0)",
        (0, 1, 0, 0, 0): "Number 1",
        (0, 1, 1, 0, 0): "Peace Sign (Number 2)",
        (1, 1, 1, 0, 0): "Number 3",
        (0, 1, 1, 1, 0): "Number 3 (W)",
        (0, 1, 1, 1, 1): "Number 4",
        (1, 1, 1, 1, 0): "Number 4 (thumb out)",
        (1, 1, 1, 1, 1): "Number 5 (Open Palm)",
        (0, 1, 0, 0, 1): "Rock On",
        (1, 1, 0, 0, 1): "I Love You",
        (1, 0, 0, 0, 1): "Call Me",
        (1, 1, 0, 0, 0): "L Sign",
        (0, 0, 0, 0, 1): "Pinky Up",
        (1, 0, 1, 1, 1): "OK Sign (loose)",
    }

    name = gestures.get(tuple(states))
    if name:
        return name

    return f"Unknown ({sum(states)} fingers)"


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

last_timestamp = 0

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not read camera.")
        break

    # Unmirrors the camera
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Clean copy for the hand tracker (before anything is drawn)
    clean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------
    # Hand detection (MediaPipe)
    # -----------------------------------------------------
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=clean_rgb)

    timestamp = int(time.time() * 1000)
    if timestamp <= last_timestamp:
        timestamp = last_timestamp + 1
    last_timestamp = timestamp

    hand_results = hand_landmarker.detect_for_video(mp_image, timestamp)

    # -----------------------------------------------------
    # Face detection (YOLO)
    # -----------------------------------------------------
    results = face_model(
        frame,
        conf=0.60,
        imgsz=640,
        verbose=False
    )

    face_count = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            face_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"Face {confidence * 100:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # -----------------------------------------------------
    # Draw hands + gestures
    # -----------------------------------------------------
    hand_count = 0
    total_fingers = 0

    for i, landmarks in enumerate(hand_results.hand_landmarks):
        hand_count += 1

        pts = [(lm.x * w, lm.y * h) for lm in landmarks]
        side = hand_results.handedness[i][0].category_name

        states = get_finger_states(pts)
        gesture = classify_gesture(states, pts)
        total_fingers += sum(states)

        # Skeleton lines
        for a, b in HAND_CONNECTIONS:
            pa = (int(pts[a][0]), int(pts[a][1]))
            pb = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(frame, pa, pb, (255, 255, 255), 2)

        # Joint points
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)

        # Box around the hand
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        hx1, hy1 = int(min(xs)) - 20, int(min(ys)) - 20
        hx2, hy2 = int(max(xs)) + 20, int(max(ys)) + 20
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 128, 0), 2)

        label = f"{side} Hand: {gesture}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = max(hy1 - 10, th + 10)

        cv2.rectangle(
            frame,
            (hx1, ty - th - 8),
            (hx1 + tw + 8, ty + 6),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            frame,
            label,
            (hx1 + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2
        )

    # -----------------------------------------------------
    # Info panel
    # -----------------------------------------------------
    cv2.putText(frame, f"Faces: {face_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Hands: {hand_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

    if hand_count == 2:
        cv2.putText(frame, f"Total fingers: {total_fingers}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("YOLO Face + Hand Sign Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
hand_landmarker.close()
cv2.destroyAllWindows()