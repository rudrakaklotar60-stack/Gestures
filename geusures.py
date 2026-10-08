import mediapipe as mp
import cv2 as cv
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision import RunningMode

model = python.BaseOptions(model_asset_path = "hand_landmarker.task")
option = vision.HandLandmarkerOptions(
        base_options = model,
        running_mode = RunningMode.VIDEO,
        num_hands = 2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(option)


def draw(frame):

    timeswap = int(time.perf_counter() * 1000)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
    results = detector.detect_for_video(mp_img, timeswap)

    for hand in results.hand_landmarks:
        for lm in hand:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv.circle(frame, (x, y), 7, (0, 255, 0), -1)
        for con in HandLandmarksConnections.HAND_CONNECTIONS:
            lm1 = hand[con.start]
            lm2 = hand[con.end]
            x1 = int(lm1.x * frame.shape[1])
            y1 = int(lm1.y * frame.shape[0])
            x2 = int(lm2.x * frame.shape[1])
            y2 = int(lm2.y * frame.shape[0])
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame, results.hand_landmarks


