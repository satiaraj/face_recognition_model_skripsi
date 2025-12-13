#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniFASNet Liveness Detection - ONNX Demo

This is a standalone demo that tests ONNX-converted MiniFASNet models
for face anti-spoofing (liveness detection).

Based on: Silent-Face-Anti-Spoofing by Minivision
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Dict, List


class MiniFASNetDemo:
    """
    MiniFASNet ONNX Demo for Face Liveness Detection

    Features:
    - Multi-model fusion for better accuracy
    - 3-class output: paper photo / real face / screen photo
    - Simple face detection using OpenCV
    """

    def __init__(self, model_dir: str = "onnx"):
        """
        Initialize the demo

        Args:
            model_dir: Directory containing ONNX models
        """
        self.model_dir = Path(model_dir)
        self.sessions = {}
        self.input_size = (80, 80)

        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load ONNX models
        self._load_models()

    def _load_models(self):
        """Load all ONNX models from the model directory"""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        onnx_files = list(self.model_dir.glob("*.onnx"))

        if not onnx_files:
            raise FileNotFoundError(f"No ONNX models found in {self.model_dir}")

        # print(f"\nLoading models from: {self.model_dir}")

        for onnx_file in onnx_files:
            try:
                session = ort.InferenceSession(
                    str(onnx_file),
                    providers=['CPUExecutionProvider']
                )
                self.sessions[onnx_file.name] = session
                # print(f"  [OK] {onnx_file.name}")
            except Exception as e:
                print(f"  [FAIL] Failed to load {onnx_file.name}: {e}")

        if not self.sessions:
            raise RuntimeError("No models loaded successfully")

        # print(f"Loaded {len(self.sessions)} model(s)\n")


    def _get_new_box(self, src_w: int, src_h: int, bbox: list, scale: float) -> Tuple[int, int, int, int]:
        """
        Calculate expanded bounding box (same logic as original project)

        Args:
            src_w: Image width
            src_h: Image height
            bbox: Face box [x, y, w, h]
            scale: Expansion ratio

        Returns:
            (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
        """
        x, y, box_w, box_h = bbox

        # Limit scale to image boundaries
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        # Boundary handling: adjust opposite side to keep size
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def _preprocess_face(self, img_bgr: np.ndarray, bbox: list, scale: float) -> np.ndarray:
        """
        Preprocess face image for model input

        Args:
            img_bgr: BGR image
            bbox: Face box [x, y, w, h]
            scale: Expansion ratio (2.7 or 4.0)

        Returns:
            Preprocessed image (1, 3, 80, 80) in [0, 255] range
        """
        src_h, src_w = img_bgr.shape[:2]

        # Get expanded crop box
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
            src_w, src_h, bbox, scale
        )

        # Crop (include right-bottom boundary +1)
        face_crop = img_bgr[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]

        # Resize to model input size
        face_resized = cv2.resize(face_crop, self.input_size)

        # IMPORTANT: Keep BGR format (do NOT convert to RGB)
        # Original model was trained with BGR input from cv2.imread
        face_float = face_resized.astype(np.float32)

        # Convert to CHW format
        face_chw = np.transpose(face_float, (2, 0, 1))

        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)

        return face_batch

    def _parse_model_name(self, model_name: str) -> Tuple[float, str]:
        """
        Parse model name to get scale parameter

        Examples:
            "2.7_80x80_MiniFASNetV2.onnx" -> (2.7, "MiniFASNetV2")
            "4_0_0_80x80_MiniFASNetV1SE.onnx" -> (4.0, "MiniFASNetV1SE")

        Returns:
            (scale, model_type)
        """
        parts = model_name.split('_')

        # Parse scale
        if parts[0] == '4':
            scale = 4.0
        else:
            scale = float(parts[0])

        # Parse model type
        model_type = parts[-1].replace('.onnx', '')

        return scale, model_type

    def predict(self, img_bgr: np.ndarray, bbox: list) -> Dict:
        """
        Perform liveness detection

        Args:
            img_bgr: BGR image
            bbox: Face box [x, y, w, h]

        Returns:
            Dictionary containing:
                - label: 0=paper, 1=real, 2=screen
                - label_text: Human-readable label
                - scores: [paper_score, real_score, screen_score]
                - is_real: True if real face
                - confidence: Prediction confidence
        """
        # Accumulate predictions from all models
        predictions = np.zeros((1, 3))  # [batch, classes]

        for model_name, session in self.sessions.items():
            # Parse model parameters
            scale, model_type = self._parse_model_name(model_name)

            # Preprocess
            input_data = self._preprocess_face(img_bgr, bbox, scale)

            # Inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            output = session.run([output_name], {input_name: input_data})[0]

            # Softmax
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)

            predictions += softmax_output

        # Average across models
        predictions = predictions / len(self.sessions)

        # Get label and scores
        label = int(np.argmax(predictions[0]))

        # Labels: 0=paper photo, 1=real face, 2=screen photo

        return (label == 1)

    def test_image(self, img_bgr, bbox, output_dir: str = "output") -> Dict:
        """
        Test a single image

        Args:
            image_path: Path to input image
            output_dir: Directory to save result image

        Returns:
            Prediction result dictionary
        """
        # Read image
        # img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image")

        # Detect face
        # bbox = self.detect_face(img_bgr)
        # print(f"  Face bbox: {bbox}")

        # Predict
        result = self.predict(img_bgr, bbox)

        return result


def check_liveness(image, bbox):

    try:
        detector = MiniFASNetDemo(model_dir="Models")
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return

    try:
        result = detector.test_image(image, bbox)

        print(f"RESULT: {result}")

    except Exception as e:
        print(f"  [ERROR] {e}\n")

    return result

