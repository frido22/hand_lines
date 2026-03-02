"""
Hand Lines — CV Pipeline

Hand detection and palm cropping using MediaPipe.
"""

import os
import cv2
import numpy as np
import mediapipe as mp


# MediaPipe task-based API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Landmark indices used for palm polygon
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
INDEX_MCP = 5
INDEX_PIP = 6
MIDDLE_MCP = 9
MIDDLE_PIP = 10
RING_MCP = 13
RING_PIP = 14
PINKY_MCP = 17
PINKY_PIP = 18


def detect_hand(image_rgb):
    """Detect hand and return list of 21 normalized landmarks, or None."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]


def landmarks_to_pixels(landmarks, w, h):
    """Convert normalized landmarks to pixel coordinates."""
    return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}


def _get_palm_polygon(pts):
    """Palm-only polygon (no fingers): wrist → thumb → MCP/PIP midpoints → pinky."""
    def mid(a, b):
        return ((pts[a][0] + pts[b][0]) // 2, (pts[a][1] + pts[b][1]) // 2)

    return np.array([
        pts[WRIST],
        pts[THUMB_CMC],
        pts[THUMB_MCP],
        pts[THUMB_IP],
        mid(INDEX_MCP, INDEX_PIP),
        mid(MIDDLE_MCP, MIDDLE_PIP),
        mid(RING_MCP, RING_PIP),
        mid(PINKY_MCP, PINKY_PIP),
        pts[PINKY_MCP],
    ], dtype=np.int32)


def get_palm_crop_box(pts, w, h, pad_frac=0.25):
    """Bounding box of palm polygon with generous padding."""
    poly = _get_palm_polygon(pts)
    xs, ys = poly[:, 0], poly[:, 1]
    pw, ph = max(xs) - min(xs), max(ys) - min(ys)
    pad_x = int(pad_frac * pw)
    pad_top = int(pad_frac * ph)
    pad_bot = int(pad_frac * ph * 1.5)  # extra padding at wrist
    return (
        max(0, min(xs) - pad_x),
        max(0, min(ys) - pad_top),
        min(w, max(xs) + pad_x),
        min(h, max(ys) + pad_bot),
    )


def create_palm_mask(pts, shape):
    """Binary mask of the palm area (slightly expanded, soft edges)."""
    h, w = shape[:2]
    poly = _get_palm_polygon(pts)
    center = poly.mean(axis=0)
    expanded = ((poly - center) * 1.6 + center).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [expanded], 255)
    mask = cv2.GaussianBlur(mask, (21, 21), 8)
    return mask
