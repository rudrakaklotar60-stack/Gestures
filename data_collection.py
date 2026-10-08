import csv
import math
import os
import time

import cv2 as cv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections


OUTPUT_FILE = "hand_features.txt"

FEATURE_NAMES = [
    "step",
    "thumb_curl",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
    "thumb_dx",
    "thumb_dy",
    "index_dx",
    "index_dy",
    "middle_dx",
    "middle_dy",
    "ring_dx",
    "ring_dy",
    "pinky_dx",
    "pinky_dy",
    "thumb_index_spread",
    "index_middle_spread",
    "middle_ring_spread",
    "ring_pinky_spread",
    "palm_pitch",
    "palm_roll",
    "palm_width",
    "palm_height",
    "palm_diagonal",
    "thumb_index_distance",
    "thumb_middle_distance",
    "thumb_ring_distance",
    "thumb_pinky_distance",
    "index_middle_distance",
    "index_ring_distance",
    "index_pinky_distance",
    "middle_ring_distance",
    "middle_pinky_distance",
    "ring_pinky_distance",
    "palm_thumb_distance",
    "palm_index_distance",
    "palm_middle_distance",
    "palm_ring_distance",
    "palm_pinky_distance",
]

FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)


def landmark_point(hand, index):
    lm = hand[index]
    return (lm.x, lm.y, lm.z)


def vector(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def distance(a, b):
    dx, dy, dz = vector(a, b)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def angle(a, b, c):
    ba = vector(b, a)
    bc = vector(b, c)
    ba_len = math.sqrt(dot(ba, ba))
    bc_len = math.sqrt(dot(bc, bc))
    if ba_len == 0 or bc_len == 0:
        return 0.0

    cos_value = dot(ba, bc) / (ba_len * bc_len)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.acos(cos_value)


def palm_center(hand):
    palm_ids = (0, 5, 9, 13, 17)
    x = sum(hand[i].x for i in palm_ids) / len(palm_ids)
    y = sum(hand[i].y for i in palm_ids) / len(palm_ids)
    z = sum(hand[i].z for i in palm_ids) / len(palm_ids)
    return (x, y, z)


def palm_scale(hand):
    wrist = landmark_point(hand, 0)
    index_mcp = landmark_point(hand, 5)
    pinky_mcp = landmark_point(hand, 17)
    middle_mcp = landmark_point(hand, 9)

    width = distance(index_mcp, pinky_mcp)
    height = distance(wrist, middle_mcp)
    scale = max(width, height, 1e-6)
    return width, height, scale


def finger_curl(hand, finger):
    joints = FINGER_JOINTS[finger]
    points = [landmark_point(hand, index) for index in joints]

    if finger == "thumb":
        angles = (
            angle(landmark_point(hand, 0), points[0], points[1]),
            angle(points[0], points[1], points[2]),
            angle(points[1], points[2], points[3]),
        )
    else:
        angles = (
            angle(points[0], points[1], points[2]),
            angle(points[1], points[2], points[3]),
        )

    straight_angle = math.pi * len(angles)
    curl = (straight_angle - sum(angles)) / straight_angle
    return max(0.0, min(1.0, curl))


def palm_orientation(hand):
    wrist = landmark_point(hand, 0)
    index_mcp = landmark_point(hand, 5)
    pinky_mcp = landmark_point(hand, 17)
    middle_mcp = landmark_point(hand, 9)

    across_palm = vector(pinky_mcp, index_mcp)
    up_palm = vector(wrist, middle_mcp)
    normal = cross(across_palm, up_palm)
    normal_len = math.sqrt(dot(normal, normal))
    if normal_len == 0:
        return 0.0, 0.0

    nx = normal[0] / normal_len
    ny = normal[1] / normal_len
    nz = normal[2] / normal_len

    pitch = math.atan2(ny, max(abs(nz), 1e-6))
    roll = math.atan2(nx, max(abs(nz), 1e-6))
    return pitch, roll


def extract_features(hand):
    center = palm_center(hand)
    palm_width, palm_height, scale = palm_scale(hand)
    palm_diagonal = math.sqrt(palm_width * palm_width + palm_height * palm_height)
    pitch, roll = palm_orientation(hand)

    tip_points = {
        finger: landmark_point(hand, index)
        for finger, index in FINGER_TIPS.items()
    }

    curls = [finger_curl(hand, finger) for finger in FINGER_TIPS]

    tip_offsets = []
    for finger in FINGER_TIPS:
        tip = tip_points[finger]
        tip_offsets.extend(
            [
                (tip[0] - center[0]) / scale,
                (tip[1] - center[1]) / scale,
            ]
        )

    adjacent_spreads = [
        distance(tip_points["thumb"], tip_points["index"]) / scale,
        distance(tip_points["index"], tip_points["middle"]) / scale,
        distance(tip_points["middle"], tip_points["ring"]) / scale,
        distance(tip_points["ring"], tip_points["pinky"]) / scale,
    ]

    tip_distances = []
    finger_names = list(FINGER_TIPS)
    for i in range(len(finger_names)):
        for j in range(i + 1, len(finger_names)):
            tip_distances.append(
                distance(tip_points[finger_names[i]], tip_points[finger_names[j]]) / scale
            )

    palm_tip_distances = [
        distance(center, tip_points[finger]) / scale
        for finger in FINGER_TIPS
    ]

    features = (
        curls
        + tip_offsets
        + adjacent_spreads
        + [pitch, roll, palm_width, palm_height, palm_diagonal]
        + tip_distances
        + palm_tip_distances
    )

    if len(features) != 39:
        raise ValueError(f"Expected 39 features, got {len(features)}")

    return features


def draw_landmarks(frame, hands):
    for hand in hands:
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


def detect_hands(frame):
    timestamp_ms = int(time.perf_counter() * 1000)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect_for_video(mp_img, timestamp_ms)
    hands = results.hand_landmarks
    draw_landmarks(frame, hands)
    return frame, hands


def write_header_if_needed(file):
    if file.tell() == 0:
        writer = csv.writer(file)
        writer.writerow(FEATURE_NAMES)


def next_step_number(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0

    with open(path, "r", newline="") as file:
        return max(sum(1 for _ in file) - 1, 0)


def main():
    cap = cv.VideoCapture(0)
    step = next_step_number(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        write_header_if_needed(file)
        writer = csv.writer(file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, hands = detect_hands(frame)

            if hands:
                features = extract_features(hands[0])
                writer.writerow([step] + [f"{value:.8f}" for value in features])
                file.flush()
                cv.putText(
                    frame,
                    f"saved step {step}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                step += 1

            cv.imshow("Hand Landmarks", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
