import argparse
import math
import os
import subprocess
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import cv2 as cv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

from model import load_weights, predict
from overlay import OverlayStatus

MODEL_PATH = "hand_landmarker.task"
WEIGHTS_PATH = "Wights.npz"
STABLE_FRAME_TARGET = 10
DRIFT_TOLERANCE_FRAMES = 1
ACTION_COOLDOWN_FRAMES = 10
MOTION_HISTORY_FRAMES = 5
MOTION_THRESHOLD = 0.02
MIN_CONFIDENCE = 0.60
CURSOR_STABLE_FRAME_TARGET = 3
CURSOR_SENSITIVITY = 2000
SCROLL_SENSITIVITY = 50
SCROLL_AMOUNT = 0
SHORT_PINCH_MAX = 6
LONG_PINCH_MIN = 10
DOUBLE_PINCH_WINDOW = 15
PINCH_THRESHOLD = 0.20
PINCH_NEAR_THRESHOLD = 0.50
PINCH_OTHER_FINGER_MIN = 0.35
ONE_SHOT_GESTURES = {"hand_open", "hand_close"}

GESTURE_ACTIONS = {
    "hand_open": ("key", "space"),
    "hand_close": ("key", "f"),
    "volume_control": ("volume", 5),
}

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

def create_hand_landmarker(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

def landmark_point(hand, index):
    landmark = hand[index]
    return (landmark.x, landmark.y, landmark.z)

def vector(start, end):
    return (end[0] - start[0], end[1] - start[1], end[2] - start[2])

def distance(start, end):
    dx, dy, dz = vector(start, end)
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def dot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]

def cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )

def angle(start, middle, end):
    first = vector(middle, start)
    second = vector(middle, end)
    first_len = math.sqrt(dot(first, first))
    second_len = math.sqrt(dot(second, second))
    if first_len == 0 or second_len == 0:
        return 0.0

    cos_value = dot(first, second) / (first_len * second_len)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.acos(cos_value)

def palm_center(hand):
    palm_indices = (0, 5, 9, 13, 17)
    x = sum(hand[index].x for index in palm_indices) / len(palm_indices)
    y = sum(hand[index].y for index in palm_indices) / len(palm_indices)
    z = sum(hand[index].z for index in palm_indices) / len(palm_indices)
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

    normal_x = normal[0] / normal_len
    normal_y = normal[1] / normal_len
    normal_z = normal[2] / normal_len

    pitch = math.atan2(normal_y, max(abs(normal_z), 1e-6))
    roll = math.atan2(normal_x, max(abs(normal_z), 1e-6))
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
        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv.circle(frame, (x, y), 6, (0, 255, 0), -1)

        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start = hand[connection.start]
            end = hand[connection.end]
            x1 = int(start.x * frame.shape[1])
            y1 = int(start.y * frame.shape[0])
            x2 = int(end.x * frame.shape[1])
            y2 = int(end.y * frame.shape[0])
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def put_status(frame, text, position=(20, 40), color=(0, 255, 0)):
    cv.putText(
        frame,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv.LINE_AA,
    )

class GestureStabilizer:
    def __init__(self, stable_frames, drift_tolerance, cooldown_frames):
        self.stable_frames = stable_frames
        self.drift_tolerance = drift_tolerance
        self.cooldown_frames = cooldown_frames
        self.current_gesture = None
        self.current_count = 0
        self.drift_gesture = None
        self.drift_count = 0
        self.cooldown_remaining = 0
        self.locked_gesture = None

    def update(self, prediction):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if prediction is None:
            self.current_gesture = None
            self.current_count = 0
            self.drift_gesture = None
            self.drift_count = 0
            self.locked_gesture = None
            return None

        if prediction != self.locked_gesture:
            self.locked_gesture = None

        if prediction == self.current_gesture:
            self.current_count += 1
            self.drift_gesture = None
            self.drift_count = 0
        elif self.current_gesture is None:
            self.current_gesture = prediction
            self.current_count = 1
        elif prediction == self.drift_gesture:
            self.drift_count += 1
            if self.drift_count > self.drift_tolerance:
                self.current_gesture = prediction
                self.current_count = self.drift_count
                self.drift_gesture = None
                self.drift_count = 0
        else:
            self.drift_gesture = prediction
            self.drift_count = 1

        if (
            self.current_count >= self.stable_frames
            and self.cooldown_remaining == 0
            and self.current_gesture != self.locked_gesture
        ):
            return self.current_gesture

        return None

    def start_cooldown(self, lock_gesture=None):
        self.cooldown_remaining = self.cooldown_frames
        self.locked_gesture = lock_gesture
        self.current_count = 0
        self.drift_gesture = None
        self.drift_count = 0

class HandMotionTracker:
    def __init__(self, history_frames, threshold):
        self.history_frames = history_frames
        self.threshold = threshold
        self.y_history = []
        self.previous_center = None
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.cursor_alpha = 0.18
        self.scroll_alpha = 0.35
        self.filtered_delta_x = 0.0
        self.filtered_delta_y_cursor = 0.0
        self.filtered_delta_y_scroll = 0.0
        self.scroll_accumulator = 0.0

    def update(self, hand):
        center_x, center_y, _ = palm_center(hand)
        if self.previous_center is None:
            self.delta_x = 0.0
            self.delta_y = 0.0
        else:
            self.delta_x = center_x - self.previous_center[0]
            self.delta_y = center_y - self.previous_center[1]
        self.previous_center = (center_x, center_y)
        self.y_history.append(center_y)
        if len(self.y_history) > self.history_frames:
            self.y_history.pop(0)

        self.filtered_delta_x = (self.cursor_alpha * self.delta_x + (1.0 - self.cursor_alpha) * self.filtered_delta_x)
        self.filtered_delta_y_cursor = (self.cursor_alpha * self.delta_y + (1.0 - self.cursor_alpha) * self.filtered_delta_y_cursor)
        self.filtered_delta_y_scroll = (self.scroll_alpha * self.delta_y + (1.0 - self.scroll_alpha) * self.filtered_delta_y_scroll)


    def reset(self):
        self.y_history = []
        self.previous_center = None
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.filtered_delta_x = 0.0
        self.filtered_delta_y_cursor = 0.0
        self.reset_scroll()

    def reset_scroll(self):
        self.filtered_delta_y_scroll = 0.0
        self.scroll_accumulator = 0.0

    def direction(self):
        if len(self.y_history) < self.history_frames:
            return None
        delta_y = self.y_history[-1] - self.y_history[0]

        if delta_y < -self.threshold:
            return "up"

        if delta_y > self.threshold:
            return "down"

        return None

    def scroll_delta(self, sensitivity):
        if abs(self.filtered_delta_y_scroll) < 0.005:
            return 0

        self.scroll_accumulator += (self.filtered_delta_y_scroll * sensitivity)
        scroll = int(self.scroll_accumulator)
        self.scroll_accumulator -= scroll
        return scroll

    def cursor_delta(self, sensitivity):

        dead_zone = 0.0015

        dx = self.filtered_delta_x
        dy = self.filtered_delta_y_cursor

        if abs(dx) < dead_zone:
            dx = 0.0

        if abs(dy) < dead_zone:
            dy = 0.0

        return (-dx * sensitivity, dy * sensitivity)




class PinchController:
    def __init__(self, long_pinch_frames=10, short_pinch_max=6, double_pinch_window=15, release_tolerance=3):
        self.long_pinch_frames = long_pinch_frames
        self.short_pinch_max = short_pinch_max
        self.double_pinch_window = double_pinch_window
        self.release_tolerance = release_tolerance
        self.pinching = False
        self.pinch_frames = 0
        self.waiting_second_pinch = False
        self.wait_frames = 0
        self.cursor_mode = False
        self.release_frames = 0

    def update(self, pinch_detected, near_pinch):
        if pinch_detected:
            self.release_frames = 0
            if not self.pinching:
                self.pinching = True
                self.pinch_frames = 1
            else:
                self.pinch_frames += 1

            if (
                not self.cursor_mode
                and self.pinch_frames >= self.long_pinch_frames
            ):
                self.cursor_mode = True

            return None
        if self.pinching:
            if near_pinch:
                held = self.pinch_frames
                self.pinching = False
                self.pinch_frames = 0
                self.release_frames = 0
                if self.cursor_mode:
                    self.cursor_mode = False
                    return None
                if held <= self.short_pinch_max:
                    if self.waiting_second_pinch:
                        self.waiting_second_pinch = False
                        self.wait_frames = 0
                        return "right_click"
                    self.waiting_second_pinch = True
                    self.wait_frames = 0
                return None
            else:
                self.release_frames += 1
                if self.release_frames <= self.release_tolerance:
                    return None

            held = self.pinch_frames
            self.pinching = False
            self.pinch_frames = 0
            self.release_frames = 0
            if self.cursor_mode:
                self.cursor_mode = False

        if self.waiting_second_pinch:
            self.wait_frames += 1
            if self.wait_frames > self.double_pinch_window:
                self.waiting_second_pinch = False
                return "left_click"

        return None

    def reset(self):
        self.pinching = False
        self.pinch_frames = 0
        self.waiting_second_pinch = False
        self.wait_frames = 0
        self.cursor_mode = False
        self.release_frames = 0

def detect_pinch(hand, pinch_threshold, near_threshold, other_finger_min):
    _, _, scale = palm_scale(hand)
    thumb_tip = landmark_point(hand, FINGER_TIPS["thumb"])
    index_tip = landmark_point(hand, FINGER_TIPS["index"])
    thumb_index_distance = distance(thumb_tip, index_tip) / scale
    other_distances = [
        distance(thumb_tip, landmark_point(hand, FINGER_TIPS[finger])) / scale
        for finger in ("middle", "ring", "pinky")
    ]
    other_fingers_clear = min(other_distances) >= other_finger_min
    pinching = thumb_index_distance <= pinch_threshold and other_fingers_clear
    near_pinch = thumb_index_distance <= near_threshold and other_fingers_clear
    return pinching, near_pinch, thumb_index_distance

def gesture_motion_allowed(gesture, motion_direction):
    if gesture == "volume_control":
        return motion_direction in ("up", "down")
    return True

def normalize_gesture(gesture):
    if gesture in ("index_finger", "pinch"):
        return None
    if gesture in ("palm_up", "palm_down"):
        return "volume_control"
    return gesture

def execute_gesture(gesture, motion_direction=None):
    action = GESTURE_ACTIONS.get(gesture)
    if action is None:
        return False

    action_type, value = action
    if action_type == "volume" and not gesture_motion_allowed(gesture, motion_direction):
        return False
    if action_type == "volume" and motion_direction == "down":
        value = -value

    if action_type == "key":
        pyautogui = load_pyautogui()

        pyautogui.press(value, _pause=False)
    elif action_type == "click":
        pyautogui = load_pyautogui()

        pyautogui.click(button=value, _pause=False)
    elif action_type == "volume":
        subprocess.run(
            [
                "osascript",
                "-e",
                (
                    "set volume output volume "
                    f"((output volume of (get volume settings)) + {value})"
                ),
            ],
            check=False,
        )
    else:
        return False

    return True

def get_output_volume():
    result = subprocess.run(
        [
            "osascript",
            "-e",
            "output volume of (get volume settings)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    volume = result.stdout.strip()
    return volume if volume else "?"

def load_pyautogui():
    import pyautogui

    pyautogui.PAUSE = 0
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    return pyautogui

def move_cursor(delta_x, delta_y):
    pyautogui = load_pyautogui()

    if abs(delta_x) < 1 and abs(delta_y) < 1:
        return False

    pyautogui.moveRel(int(delta_x), int(delta_y), duration=0, _pause=False)
    return True

def detect_hands(detector, frame):
    timestamp_ms = int(time.perf_counter() * 1000)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    return result.hand_landmarks

def run(args):
    weights = load_weights(args.weights)
    detector = create_hand_landmarker(args.landmarker)
    video = cv.VideoCapture(args.camera)
    frame_count = 0
    stabilizer = GestureStabilizer(
        stable_frames=args.stable_frames,
        drift_tolerance=args.drift_tolerance,
        cooldown_frames=args.cooldown_frames,
    )
    motion_tracker = HandMotionTracker(
        history_frames=args.motion_history,
        threshold=args.motion_threshold,
    )
    pinch_controller = PinchController(long_pinch_frames=args.cursor_stable_frames)
    overlay = OverlayStatus(enabled=args.overlay)
    overlay.start()
    last_action = "none"
    
    if not video.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            hands = detect_hands(detector, frame)

            if hands:
                draw_landmarks(frame, hands)

                predictions = []
                for hand_index, hand in enumerate(hands):
                    features = extract_features(hand)
                    gesture, confidence = predict(features, weights=weights)
                    predictions.append((hand_index, gesture, confidence))

                    label = f"hand {hand_index}: {gesture} {confidence * 100:.1f}%"
                    put_status(frame, label, (20, 40 + hand_index * 35))

                best_prediction = max(predictions, key=lambda item: item[2])
                best_hand = hands[best_prediction[0]]
                motion_tracker.update(best_hand)
                motion_direction = motion_tracker.direction()
                pinching, near_pinch, pinch_distance = detect_pinch(
                    best_hand,
                    args.pinch_threshold,
                    args.pinch_near_threshold,
                    args.pinch_other_finger_min,
                )
                tracked_gesture = normalize_gesture(best_prediction[1])
                if best_prediction[2] < args.min_confidence:
                    tracked_gesture = None
                if near_pinch:
                    tracked_gesture = None
                pinch_event = pinch_controller.update(pinching, near_pinch)
                final_gesture = stabilizer.update(tracked_gesture)
                stable_gesture = (
                    stabilizer.current_gesture
                    if stabilizer.current_count >= args.stable_frames
                    else None
                )
                
                status_y = 40 + len(predictions) * 35
                put_status(
                    frame,
                    (
                        f"tracking: {stabilizer.current_gesture} "
                        f"{stabilizer.current_count}/{args.stable_frames} "
                        f"cooldown={stabilizer.cooldown_remaining} "
                        f"locked={stabilizer.locked_gesture} "
                        f"motion={motion_direction} "
                        f"pinch={pinch_distance:.2f}"
                    ),
                    (20, status_y),
                    (255, 255, 0),
                )

                if pinch_controller.cursor_mode:

                    cursor_delta_x, cursor_delta_y = motion_tracker.cursor_delta(args.cursor_sensitivity)
                    if move_cursor(cursor_delta_x, cursor_delta_y):
                        last_action = "cursor move"
                        put_status(frame, "cursor", (20, status_y + 35), (0, 255, 255))

                elif stable_gesture == "peace" and tracked_gesture == "peace":

                    scroll = motion_tracker.scroll_delta(args.scroll_sensitivity)
                    if scroll != 0:
                        pyautogui = load_pyautogui()
                        pyautogui.scroll(scroll, _pause=False)
                        last_action = f"scroll {scroll}"

                    put_status(frame, f"scroll {scroll}", (20, status_y + 35), (0, 255, 255))

                else:
                    if stable_gesture != "peace":
                        motion_tracker.reset_scroll()

                    if pinch_event == "left_click":
                        pyautogui = load_pyautogui()
                        pyautogui.click(button="left", _pause=False)
                        last_action = "left click"

                        put_status(frame, "Left Click", (20, status_y + 35), (0,255,255))

                    elif pinch_event == "right_click":

                        pyautogui = load_pyautogui()
                        pyautogui.click(button="right", _pause=False)
                        last_action = "right click"

                        put_status(frame, "Right Click", (20, status_y + 35), (0,255,255))

                    elif final_gesture is not None and final_gesture != "pinch":

                        did_execute = execute_gesture(
                            final_gesture,
                            motion_direction,
                        )

                        if did_execute:
                            lock_gesture = (
                                final_gesture
                                if final_gesture in ONE_SHOT_GESTURES
                                else None
                            )
                            stabilizer.start_cooldown(lock_gesture)

                        status = "executed" if did_execute else "final"

                        if (
                            final_gesture == "volume_control"
                            and not did_execute
                        ):
                            status = "waiting for up/down motion"
                        elif final_gesture == "volume_control" and did_execute:
                            volume = get_output_volume()
                            last_action = f"volume {volume}%"
                        elif did_execute:
                            last_action = final_gesture

                        put_status(frame, f"{status}: {final_gesture}", (20, status_y + 35), (0,255,255))

                mode = "cursor" if pinch_controller.cursor_mode else "idle"
                if stable_gesture == "peace" and tracked_gesture == "peace":
                    mode = "scroll"
                elif stable_gesture == "volume_control":
                    mode = "volume"
                overlay.show(
                    "\n".join(
                        [
                            f"Mode: {mode}",
                            f"Gesture: {stabilizer.current_gesture} ({stabilizer.current_count}/{args.stable_frames})",
                            f"Prediction: {best_prediction[1]} {best_prediction[2] * 100:.1f}%",
                            f"Motion: {motion_direction}  Pinch: {pinch_distance:.2f}",
                            f"Last: {last_action}",
                        ]
                    )
                )

                if frame_count % args.print_every == 0:
                    for hand_index, gesture, confidence in predictions:
                        print(
                            f"frame={frame_count} hand={hand_index} "
                            f"gesture={gesture} confidence={confidence:.4f}"
                        )
            else:
                stabilizer.update(None)
                motion_tracker.reset()
                pinch_controller.reset()
                overlay.show("Mode: idle\nGesture: none\nLast: no hand detected")
                put_status(frame, "No hand detected", color=(0, 0, 255))

            frame_count += 1
            cv.imshow("Hand Gesture Prediction", frame)

            if cv.waitKey(1) & 0xFF == ord(args.quit_key):
                break
    finally:
        video.release()
        detector.close()
        overlay.stop()
        cv.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Predict hand gestures from webcam video.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to read from.")
    parser.add_argument("--landmarker", default=MODEL_PATH, help="MediaPipe hand landmarker task file.")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Trained neural-network weights file.")
    parser.add_argument("--print-every", type=int, default=10, help="Print prediction every N frames.")
    parser.add_argument("--quit-key", default="q", help="Keyboard key that closes the webcam window.")
    parser.add_argument("--stable-frames", type=int, default=STABLE_FRAME_TARGET, help="Frames a gesture must last before it is final.")
    parser.add_argument("--drift-tolerance", type=int, default=DRIFT_TOLERANCE_FRAMES, help="Changed predictions to ignore before switching tracked gesture.")
    parser.add_argument("--cooldown-frames", type=int, default=ACTION_COOLDOWN_FRAMES, help="Frames to wait after executing a gesture.")
    parser.add_argument("--motion-history", type=int, default=MOTION_HISTORY_FRAMES, help="Frames used to detect hand motion direction.")
    parser.add_argument("--motion-threshold", type=float, default=MOTION_THRESHOLD, help="Minimum normalized Y movement needed for volume gestures.")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, help="Minimum NN confidence needed before static gestures execute.")
    parser.add_argument("--cursor-stable-frames", type=int, default=CURSOR_STABLE_FRAME_TARGET, help="Frames pinch must last before controlling the cursor.")
    parser.add_argument("--cursor-sensitivity", type=float, default=CURSOR_SENSITIVITY, help="Cursor pixels moved for each normalized hand movement unit.")
    parser.add_argument("--scroll-sensitivity",type=float,default=SCROLL_SENSITIVITY,help="Scroll sensitivity.")
    parser.add_argument("--pinch-threshold", type=float, default=PINCH_THRESHOLD, help="Max normalized thumb-index distance for pinch.")
    parser.add_argument("--pinch-near-threshold", type=float, default=PINCH_NEAR_THRESHOLD, help="Max normalized thumb-index distance after releasing pinch.")
    parser.add_argument("--pinch-other-finger-min", type=float, default=PINCH_OTHER_FINGER_MIN, help="Minimum normalized thumb distance from middle/ring/pinky for pinch.")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false", help="Disable the macOS click-through desktop HUD.")
    parser.set_defaults(overlay=True)
    args = parser.parse_args()

    if args.print_every < 1:
        parser.error("--print-every must be at least 1")
    if len(args.quit_key) != 1:
        parser.error("--quit-key must be a single character")
    if args.stable_frames < 1:
        parser.error("--stable-frames must be at least 1")
    if args.drift_tolerance < 0:
        parser.error("--drift-tolerance cannot be negative")
    if args.cooldown_frames < 0:
        parser.error("--cooldown-frames cannot be negative")
    if args.motion_history < 2:
        parser.error("--motion-history must be at least 2")
    if args.motion_threshold <= 0:
        parser.error("--motion-threshold must be greater than 0")
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    if args.cursor_stable_frames < 1:
        parser.error("--cursor-stable-frames must be at least 1")
    if args.cursor_sensitivity <= 0:
        parser.error("--cursor-sensitivity must be greater than 0")
    if args.scroll_sensitivity <= 0:
        parser.error("--scroll-sensitivity must be greater than 0")
    if args.pinch_threshold <= 0:
        parser.error("--pinch-threshold must be greater than 0")
    if args.pinch_near_threshold <= args.pinch_threshold:
        parser.error("--pinch-near-threshold must be greater than --pinch-threshold")
    if args.pinch_other_finger_min <= 0:
        parser.error("--pinch-other-finger-min must be greater than 0")
    return args

if __name__ == "__main__":
    run(parse_args())
