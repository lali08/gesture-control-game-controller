import cv2
import mediapipe as mp
import pyautogui
import time
import urllib.request
import os

# ── Download the model if needed ──────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model downloaded.")

# ── New MediaPipe Tasks API setup ──────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

# Hand connections for drawing (hardcoded — no mp.solutions needed)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (5,9),(9,10),(10,11),(11,12),    # middle
    (9,13),(13,14),(14,15),(15,16),  # ring
    (13,17),(17,18),(18,19),(19,20), # pinky
    (0,17),                          # palm base
]

# ── Webcam ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── Shared state ───────────────────────────────────────────────────────────────
right_pressed    = False
left_pressed     = False
last_hover_time  = 0
last_action_time = 0
gesture_cooldown = 0.5   # seconds

# ── Finger-count helper (thumb excluded) ───────────────────────────────────────
def count_extended_fingers(landmarks):
    """Count extended fingers (index → pinky), ignoring thumb."""
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    return sum(
        1 for tip, pip in zip(finger_tips, finger_pips)
        if landmarks[tip].y < landmarks[pip].y
    )

# ── Draw landmarks on frame using the new NormalizedLandmark objects ───────────
def draw_landmarks_on_frame(frame, hand_landmarks_list):
    """
    hand_landmarks_list: list of mp.tasks.vision.HandLandmarkerResult.hand_landmarks
    Each element is a list of NormalizedLandmark.
    We convert to the proto format mp_drawing expects via mp.framework.formats.
    """
    h, w, _ = frame.shape
    for hand_lms in hand_landmarks_list:
        # Build a list of (x_px, y_px) points and draw connections manually
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start, end = connection
            cv2.line(frame, pts[start], pts[end], (0, 200, 0), 2)

        # Draw landmark dots
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 0, 200),     1)

# ── Gesture detection (IMAGE mode — synchronous) ───────────────────────────────
def detect_gesture(frame, landmarker):
    """Return (extended_finger_count, annotated_frame)."""
    # MediaPipe Tasks expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result   = landmarker.detect(mp_image)
    extended = 0

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]   # first hand
        extended  = count_extended_fingers(landmarks)
        draw_landmarks_on_frame(frame, result.hand_landmarks)

    return extended, frame

# ── Game modes ─────────────────────────────────────────────────────────────────
def hill_climb_mode(landmarker):
    global right_pressed, left_pressed
    print("[INFO] Gesture Mode: Hill Climb Racing")
    print("[INFO] Open hand = Accelerate (→), Fist = Brake (←), Q = Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame        = cv2.flip(frame, 1)
        t0           = time.time()
        extended, annotated = detect_gesture(frame, landmarker)

        fps = int(1 / max(time.time() - t0, 0.001))
        cv2.putText(annotated, f"FPS: {fps}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if extended >= 4:           # open hand → accelerate
            if not right_pressed:
                pyautogui.keyDown('right')
                right_pressed = True
                print("[ACTION] Accelerating →")
            if left_pressed:
                pyautogui.keyUp('left')
                left_pressed = False
            cv2.putText(annotated, "ACCELERATING", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif extended == 0:         # fist → brake
            if not left_pressed:
                pyautogui.keyDown('left')
                left_pressed = True
                print("[ACTION] Braking ←")
            if right_pressed:
                pyautogui.keyUp('right')
                right_pressed = False
            cv2.putText(annotated, "BRAKING", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:                       # transition
            if right_pressed:
                pyautogui.keyUp('right')
                right_pressed = False
            if left_pressed:
                pyautogui.keyUp('left')
                left_pressed = False
            cv2.putText(annotated, "NO GESTURE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        cv2.imshow("Hill Climb Racing", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def subway_surfers_mode(landmarker):
    global last_hover_time, last_action_time
    print("[INFO] Gesture Mode: Subway Surfers")
    print("[INFO] 1=Left, 2=Right, 3=Up/Hoverboard (double), 4+=Down, Q = Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame        = cv2.flip(frame, 1)
        t0           = time.time()
        extended, annotated = detect_gesture(frame, landmarker)

        fps = int(1 / max(time.time() - t0, 0.001))
        cv2.putText(annotated, f"FPS: {fps}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        current_time = time.time()
        if current_time - last_action_time > gesture_cooldown:
            if extended == 1:
                pyautogui.press('left')
                print("[ACTION] Move Left")
                last_action_time = current_time

            elif extended == 2:
                pyautogui.press('right')
                print("[ACTION] Move Right")
                last_action_time = current_time

            elif extended == 3:
                if current_time - last_hover_time < 1.2:
                    pyautogui.press('space')
                    pyautogui.press('space')
                    print("[ACTION] Hoverboard")
                    last_hover_time = 0
                else:
                    pyautogui.press('up')
                    print("[ACTION] Jump")
                    last_hover_time = current_time
                last_action_time = current_time

            elif extended >= 4:
                pyautogui.press('down')
                print("[ACTION] Roll Down")
                last_action_time = current_time

        cv2.imshow("Subway Surfers", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ── Main ───────────────────────────────────────────────────────────────────────
print("Choose Game Mode:")
print("1. Hill Climb Racing")
print("2. Subway Surfers")
choice = input("Enter choice (1 or 2): ")

# Create the HandLandmarker in IMAGE mode (synchronous, one frame at a time)
options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)

with HandLandmarker.create_from_options(options) as landmarker:
    if choice == '1':
        hill_climb_mode(landmarker)
    elif choice == '2':
        subway_surfers_mode(landmarker)
    else:
        print("Invalid choice. Exiting...")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pyautogui.keyUp('right')
pyautogui.keyUp('left')
print("[INFO] Exited cleanly.")