"""
Mobile Phone Detection Module
Uses MediaPipe EfficientDet-Lite0 for phone detection.
Also detects objects commonly misclassified as phones.
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


class PhoneDetector:
    """Detects mobile phone using EfficientDet-Lite0 via MediaPipe Tasks."""

    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold
        self.detection_counter = 0
        self.alert_frames_threshold = 4

        # Target categories — phones may be classified as any of these
        self.target_labels = {"cell phone", "remote"}

        # Load EfficientDet-Lite0
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "efficientdet_lite0.tflite"
        )

        self.detector = None
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                base_options = mp_tasks.BaseOptions(model_asset_buffer=model_bytes)
                options = vision.ObjectDetectorOptions(
                    base_options=base_options,
                    score_threshold=self.confidence_threshold,
                    max_results=10,  # get more results to debug
                )
                self.detector = vision.ObjectDetector.create_from_options(options)
                print("[PhoneDetector] EfficientDet-Lite0 loaded successfully")
            except Exception as e:
                print(f"[PhoneDetector] Error loading EfficientDet: {e}")
        else:
            print(f"[PhoneDetector] Model not found: {model_path}")

    def _detect_phone(self, frame):
        """Run EfficientDet on frame. Returns (found, confidence, bbox, all_labels)."""
        if self.detector is None:
            return False, 0.0, None, []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        best_conf = 0.0
        best_bbox = None
        found = False
        all_labels = []

        for detection in result.detections:
            cat = detection.categories[0]
            label = cat.category_name.lower()
            score = float(cat.score)
            all_labels.append(f"{label}:{score:.2f}")

            if label in self.target_labels and score > best_conf:
                best_conf = score
                bb = detection.bounding_box
                best_bbox = [
                    int(bb.origin_x), int(bb.origin_y),
                    int(bb.width), int(bb.height),
                ]
                found = True

        return found, best_conf, best_bbox, all_labels

    def process_frame(self, frame):
        """Process a frame for phone detection. Runs every frame."""
        phone_found, confidence, bbox, all_labels = self._detect_phone(frame)
        method = "efficientdet" if phone_found else "none"

        # Log detections periodically for debugging
        if all_labels and hasattr(self, '_log_counter'):
            self._log_counter += 1
            if self._log_counter % 30 == 0:  # log every ~2 seconds
                print(f"[PhoneDetector] Detected: {all_labels}")
        else:
            self._log_counter = 0

        # Update counter
        if phone_found:
            self.detection_counter = min(self.detection_counter + 2, 30)
        else:
            self.detection_counter = max(0, self.detection_counter - 1)

        alert = self.detection_counter >= self.alert_frames_threshold

        return {
            "phone_detected": bool(phone_found),
            "confidence": float(round(confidence, 3)),
            "bbox": [int(x) for x in bbox] if bbox else None,
            "alert": bool(alert),
            "consecutive_frames": int(self.detection_counter),
            "method": method,
        }

    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.close()
            self.detector = None
