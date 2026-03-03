"""
Drowsiness Detection Module
Uses MediaPipe Face Mesh to detect:
- Eye closure via Eye Aspect Ratio (EAR)
- Yawning via Mouth Aspect Ratio (MAR)
- Blink rate tracking
"""

import time
import numpy as np
import mediapipe as mp


class DrowsinessDetector:
    """Detects driver drowsiness using facial landmarks."""

    # MediaPipe Face Mesh landmark indices for eyes and mouth
    # Left eye landmarks
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Upper lip landmarks
    UPPER_LIP = [61, 291, 13]
    # Lower lip landmarks
    LOWER_LIP = [61, 291, 14]
    # Mouth landmarks for MAR calculation
    MOUTH_TOP = [13]
    MOUTH_BOTTOM = [14]
    MOUTH_LEFT = [61]
    MOUTH_RIGHT = [291]
    MOUTH_INNER_TOP = [82, 312]
    MOUTH_INNER_BOTTOM = [87, 317]

    def __init__(
        self,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.65,
        drowsy_frames_threshold: int = 20,
        blink_window_seconds: float = 60.0,
    ):
        """
        Initialize the drowsiness detector.

        Args:
            ear_threshold: EAR below this value means eyes are closed
            mar_threshold: MAR above this value means yawning
            drowsy_frames_threshold: consecutive closed-eye frames to trigger drowsy alert
            blink_window_seconds: rolling window for blink rate calculation
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_frames_threshold = drowsy_frames_threshold
        self.blink_window_seconds = blink_window_seconds

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
        self.yawn_counter = 0
        self.is_yawning = False

    def _compute_ear(self, landmarks, eye_indices, img_w, img_h):
        """
        Compute Eye Aspect Ratio (EAR) for a single eye.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append((lm.x * img_w, lm.y * img_h))

        # Vertical distances
        v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        # Horizontal distance
        h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _compute_mar(self, landmarks, img_w, img_h):
        """
        Compute Mouth Aspect Ratio (MAR).
        MAR = (||top-bottom|| + ||inner_top1-inner_bottom1|| + ||inner_top2-inner_bottom2||) / (2 * ||left-right||)
        """
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        # Mouth corners
        left = get_point(61)
        right = get_point(291)
        top = get_point(13)
        bottom = get_point(14)
        inner_top1 = get_point(82)
        inner_bottom1 = get_point(87)
        inner_top2 = get_point(312)
        inner_bottom2 = get_point(317)

        # Vertical distances
        v1 = np.linalg.norm(top - bottom)
        v2 = np.linalg.norm(inner_top1 - inner_bottom1)
        v3 = np.linalg.norm(inner_top2 - inner_bottom2)

        # Horizontal distance
        h = np.linalg.norm(left - right)

        if h == 0:
            return 0.0
        mar = (v1 + v2 + v3) / (2.0 * h)
        return mar

    def _get_eye_landmarks(self, landmarks, eye_indices, img_w, img_h):
        """Get pixel coordinates for eye landmarks."""
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            points.append((int(lm.x * img_w), int(lm.y * img_h)))
        return points

    def process_frame(self, frame):
        """
        Process a single video frame for drowsiness detection.

        Args:
            frame: BGR image from OpenCV

        Returns:
            dict with detection results:
            {
                "face_detected": bool,
                "ear_left": float,
                "ear_right": float,
                "ear_avg": float,
                "mar": float,
                "blink_count": int,
                "drowsy": bool,
                "yawning": bool,
                "closed_eye_frames": int,
                "left_eye_points": list,
                "right_eye_points": list,
            }
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
            "blink_count": len(self.blink_timestamps),
            "drowsy": False,
            "yawning": False,
            "closed_eye_frames": self.closed_eye_counter,
            "left_eye_points": [],
            "right_eye_points": [],
        }

        if not results.multi_face_landmarks:
            return status

        face_landmarks = results.multi_face_landmarks[0].landmark
        status["face_detected"] = True

        # Compute EAR for both eyes
        ear_left = self._compute_ear(face_landmarks, self.LEFT_EYE, w, h)
        ear_right = self._compute_ear(face_landmarks, self.RIGHT_EYE, w, h)
        ear_avg = (ear_left + ear_right) / 2.0

        status["ear_left"] = round(ear_left, 3)
        status["ear_right"] = round(ear_right, 3)
        status["ear_avg"] = round(ear_avg, 3)

        # Get eye landmark points for drawing
        status["left_eye_points"] = self._get_eye_landmarks(
            face_landmarks, self.LEFT_EYE, w, h
        )
        status["right_eye_points"] = self._get_eye_landmarks(
            face_landmarks, self.RIGHT_EYE, w, h
        )

        # Check eye closure
        if ear_avg < self.ear_threshold:
            self.closed_eye_counter += 1
            if not self.was_eye_closed:
                self.was_eye_closed = True
        else:
            if self.was_eye_closed:
                # Eye just opened — register a blink
                self.blink_timestamps.append(time.time())
                self.was_eye_closed = False
            self.closed_eye_counter = 0

        # Clean old blinks outside the rolling window
        current_time = time.time()
        self.blink_timestamps = [
            t
            for t in self.blink_timestamps
            if current_time - t <= self.blink_window_seconds
        ]

        # Drowsy if eyes closed for too many consecutive frames
        status["drowsy"] = bool(self.closed_eye_counter >= self.drowsy_frames_threshold)
        status["closed_eye_frames"] = int(self.closed_eye_counter)
        status["blink_count"] = int(len(self.blink_timestamps))

        # Compute MAR for yawning
        mar = self._compute_mar(face_landmarks, w, h)
        status["mar"] = float(round(mar, 3))
        status["yawning"] = bool(mar > self.mar_threshold)

        return status

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
