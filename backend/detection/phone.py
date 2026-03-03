"""
Mobile Phone Detection Module
Uses MediaPipe EfficientDet-Lite0 object detector for accurate phone detection.
No heuristics — only detects when it actually sees a phone object.
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


class PhoneDetector:
    """Detects mobile phone using EfficientDet-Lite0 via MediaPipe Tasks."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.detection_counter = 0
        self.alert_frames_threshold = 5

        # Frame skipping
        self.frame_count = 0
        self.skip_frames = 1  # run every 2nd frame
        self.last_result = {
            "phone_detected": False,
            "confidence": 0.0,
            "bbox": None,
            "alert": False,
            "consecutive_frames": 0,
            "method": "none",
        }

        # Target categories (COCO labels)
        self.target_labels = {"cell phone", "remote"}

        # Load EfficientDet-Lite0
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "efficientdet_lite0.tflite"
        )

        self.detector = None
        if os.path.exists(model_path):
            try:
                base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
                options = vision.ObjectDetectorOptions(
                    base_options=base_options,
                    score_threshold=self.confidence_threshold,
                    max_results=5,
                )
                self.detector = vision.ObjectDetector.create_from_options(options)
                print("[PhoneDetector] EfficientDet-Lite0 loaded successfully")
            except Exception as e:
                print(f"[PhoneDetector] Error loading EfficientDet: {e}")
        else:
            print(f"[PhoneDetector] Model not found: {model_path}")

    def _detect_phone(self, frame):
        """Run EfficientDet on a frame to detect phones."""
        if self.detector is None:
            return False, 0.0, None

        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        result = self.detector.detect(mp_image)

        best_conf = 0.0
        best_bbox = None
        found = False

        for detection in result.detections:
            category = detection.categories[0]
            label = category.category_name.lower()
            score = float(category.score)

            if label in self.target_labels and score > best_conf:
                best_conf = score
                bb = detection.bounding_box
                best_bbox = [
                    int(bb.origin_x),
                    int(bb.origin_y),
                    int(bb.width),
                    int(bb.height),
                ]
                found = True

        return found, best_conf, best_bbox

    def process_frame(self, frame):
        """Process a frame for phone detection."""
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % (self.skip_frames + 1) != 0:
            # Decay counter on skip frames
            if not self.last_result["phone_detected"]:
                self.detection_counter = max(0, self.detection_counter - 1)
                self.last_result["consecutive_frames"] = int(self.detection_counter)
                self.last_result["alert"] = bool(
                    self.detection_counter >= self.alert_frames_threshold
                )
            return self.last_result

        phone_found, confidence, bbox = self._detect_phone(frame)
        method = "efficientdet" if phone_found else "none"

        # Update counter
        if phone_found:
            self.detection_counter = min(self.detection_counter + 1, 30)
        else:
            self.detection_counter = max(0, self.detection_counter - 2)

        alert = self.detection_counter >= self.alert_frames_threshold

        self.last_result = {
            "phone_detected": bool(phone_found),
            "confidence": float(round(confidence, 3)),
            "bbox": [int(x) for x in bbox] if bbox else None,
            "alert": bool(alert),
            "consecutive_frames": int(self.detection_counter),
            "method": method,
        }
        return self.last_result

    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.close()
            self.detector = None
