"""
Drowsiness & Head Pose Detection Module
Uses MediaPipe Face Mesh to detect:
- Eye closure via Eye Aspect Ratio (EAR)
- Yawning via Mouth Aspect Ratio (MAR)
- Blink rate tracking
- Head turn (sideways look) via face landmark geometry
"""

import time
import numpy as np
import mediapipe as mp


class DrowsinessDetector:
    """Detects driver drowsiness and head turns using facial landmarks."""

    # MediaPipe Face Mesh landmark indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH_TOP = [13]
    MOUTH_BOTTOM = [14]
    MOUTH_LEFT = [61]
    MOUTH_RIGHT = [291]
    MOUTH_INNER_TOP = [82, 312]
    MOUTH_INNER_BOTTOM = [87, 317]

    # Head pose landmarks
    NOSE_TIP = 1
    LEFT_CHEEK = 234    # left side of face
    RIGHT_CHEEK = 454   # right side of face
    CHIN = 152
    FOREHEAD = 10
    LEFT_EAR = 127
    RIGHT_EAR = 356

    def __init__(
        self,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.65,
        drowsy_frames_threshold: int = 20,
        blink_window_seconds: float = 60.0,
        head_turn_threshold: float = 0.34,
        head_turn_frames: int = 15,
    ):
        """
        Initialize the drowsiness detector.

        Args:
            ear_threshold: EAR below this value means eyes are closed
            mar_threshold: MAR above this value means yawning
            drowsy_frames_threshold: consecutive closed-eye frames to trigger drowsy alert
            blink_window_seconds: rolling window for blink rate calculation
            head_turn_threshold: ratio threshold for sideways head turn detection
            head_turn_frames: consecutive frames needed to trigger head turn alert
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_frames_threshold = drowsy_frames_threshold
        self.blink_window_seconds = blink_window_seconds
        self.head_turn_threshold = head_turn_threshold
        self.head_turn_frames = head_turn_frames

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State tracking
        self.closed_eye_counter = 0
        self.blink_timestamps = []
        self.was_eye_closed = False
        self.head_turn_counter = 0

    def _compute_ear(self, landmarks, eye_indices, img_w, img_h):
        """Compute Eye Aspect Ratio (EAR) for a single eye."""
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append((lm.x * img_w, lm.y * img_h))

        v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _compute_mar(self, landmarks, img_w, img_h):
        """Compute Mouth Aspect Ratio (MAR)."""
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        left = get_point(61)
        right = get_point(291)
        top = get_point(13)
        bottom = get_point(14)
        inner_top1 = get_point(82)
        inner_bottom1 = get_point(87)
        inner_top2 = get_point(312)
        inner_bottom2 = get_point(317)

        v1 = np.linalg.norm(top - bottom)
        v2 = np.linalg.norm(inner_top1 - inner_bottom1)
        v3 = np.linalg.norm(inner_top2 - inner_bottom2)
        h = np.linalg.norm(left - right)

        if h == 0:
            return 0.0
        return (v1 + v2 + v3) / (2.0 * h)

    def _compute_head_turn(self, landmarks, img_w, img_h):
        """
        Compute head yaw (left/right turn) using nose-to-cheek distances.

        Returns:
            direction: "center", "left", or "right"
            ratio: asymmetry ratio (0 = centered, 1 = fully turned)
            looking_sideways: True if turned beyond threshold
        """
        nose = np.array([landmarks[self.NOSE_TIP].x * img_w,
                         landmarks[self.NOSE_TIP].y * img_h])
        left_cheek = np.array([landmarks[self.LEFT_CHEEK].x * img_w,
                               landmarks[self.LEFT_CHEEK].y * img_h])
        right_cheek = np.array([landmarks[self.RIGHT_CHEEK].x * img_w,
                                landmarks[self.RIGHT_CHEEK].y * img_h])

        # Distance from nose to each cheek
        dist_left = np.linalg.norm(nose - left_cheek)
        dist_right = np.linalg.norm(nose - right_cheek)

        # Total face width for normalization
        total = dist_left + dist_right
        if total == 0:
            return "center", 0.0, False

        # Ratio: how far off-center the nose is
        # 0.5 = perfectly centered, 0 = fully left, 1 = fully right
        ratio = dist_left / total

        # Asymmetry from center (0 = centered, 0.5 = max)
        asymmetry = abs(ratio - 0.5)

        if asymmetry > self.head_turn_threshold:
            direction = "left" if ratio < 0.5 else "right"
            return direction, float(round(asymmetry, 3)), True
        else:
            return "center", float(round(asymmetry, 3)), False

    def _get_eye_landmarks(self, landmarks, eye_indices, img_w, img_h):
        """Get pixel coordinates for eye landmarks."""
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            points.append((int(lm.x * img_w), int(lm.y * img_h)))
        return points

    def process_frame(self, frame):
        """
        Process a single video frame for drowsiness and head pose detection.

        Returns:
            dict with detection results
        """
        import cv2

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = {
            "face_detected": False,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "ear_avg": 0.0,
            "mar": 0.0,
            "blink_count": int(len(self.blink_timestamps)),
            "drowsy": False,
            "yawning": False,
            "closed_eye_frames": int(self.closed_eye_counter),
            "left_eye_points": [],
            "right_eye_points": [],
            # Head turn fields
            "head_direction": "center",
            "head_turn_ratio": 0.0,
            "looking_sideways": False,
            "head_turn_alert": False,
            "head_turn_frames": 0,
        }

        if not results.multi_face_landmarks:
            return status

        face_landmarks = results.multi_face_landmarks[0].landmark
        status["face_detected"] = True

        # ── EAR ──
        ear_left = self._compute_ear(face_landmarks, self.LEFT_EYE, w, h)
        ear_right = self._compute_ear(face_landmarks, self.RIGHT_EYE, w, h)
        ear_avg = (ear_left + ear_right) / 2.0

        status["ear_left"] = float(round(ear_left, 3))
        status["ear_right"] = float(round(ear_right, 3))
        status["ear_avg"] = float(round(ear_avg, 3))

        # Eye landmark points for drawing
        status["left_eye_points"] = self._get_eye_landmarks(
            face_landmarks, self.LEFT_EYE, w, h
        )
        status["right_eye_points"] = self._get_eye_landmarks(
            face_landmarks, self.RIGHT_EYE, w, h
        )

        # ── Eye closure / blink tracking ──
        if ear_avg < self.ear_threshold:
            self.closed_eye_counter += 1
            if not self.was_eye_closed:
                self.was_eye_closed = True
        else:
            if self.was_eye_closed:
                self.blink_timestamps.append(time.time())
                self.was_eye_closed = False
            self.closed_eye_counter = 0

        current_time = time.time()
        self.blink_timestamps = [
            t for t in self.blink_timestamps
            if current_time - t <= self.blink_window_seconds
        ]

        status["drowsy"] = bool(self.closed_eye_counter >= self.drowsy_frames_threshold)
        status["closed_eye_frames"] = int(self.closed_eye_counter)
        status["blink_count"] = int(len(self.blink_timestamps))

        # ── MAR (yawning) ──
        mar = self._compute_mar(face_landmarks, w, h)
        status["mar"] = float(round(mar, 3))
        status["yawning"] = bool(mar > self.mar_threshold)

        # ── Head turn detection ──
        direction, ratio, looking_sideways = self._compute_head_turn(
            face_landmarks, w, h
        )
        status["head_direction"] = direction
        status["head_turn_ratio"] = ratio

        if looking_sideways:
            self.head_turn_counter += 1
        else:
            self.head_turn_counter = max(0, self.head_turn_counter - 2)

        status["looking_sideways"] = bool(looking_sideways)
        status["head_turn_alert"] = bool(self.head_turn_counter >= self.head_turn_frames)
        status["head_turn_frames"] = int(self.head_turn_counter)

        return status

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
