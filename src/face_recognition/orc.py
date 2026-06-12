import cv2
import numpy as np
import dlib
from skimage.feature import hog


def _clamp_bbox_xyxy(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))

    if x2 <= x1:
        x2 = min(image_width, x1 + 1)

    if y2 <= y1:
        y2 = min(image_height, y1 + 1)

    return [x1, y1, x2, y2]


PATCH_SIZE = (64, 64)

HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
    "feature_vector": True,
}


class HaarDetector:
    def __init__(
        self, cascade_path=cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ):
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, img_gray):
        faces = self.detector.detectMultiScale(
            img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        x, y, w, h = faces[0]
        return _clamp_bbox_xyxy(
            (x, y, x + w, y + h), img_gray.shape[1], img_gray.shape[0]
        )


class HOGDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, img_gray):
        faces = self.detector(img_gray, 1)
        if len(faces) == 0:
            return None

        rect = faces[0]
        x1, y1 = rect.left(), rect.top()
        x2, y2 = rect.right(), rect.bottom()
        return _clamp_bbox_xyxy((x1, y1, x2, y2), img_gray.shape[1], img_gray.shape[0])


def resize_patch_for_hog(patch, patch_size=PATCH_SIZE):
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    if patch.shape[:2] != (patch_size[1], patch_size[0]):
        patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_AREA)

    return patch


def extract_hog_feature(gray_patch):
    gray_patch = resize_patch_for_hog(gray_patch)

    feature = hog(
        gray_patch,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        transform_sqrt=HOG_PARAMS["transform_sqrt"],
        feature_vector=HOG_PARAMS["feature_vector"],
    )

    return feature.astype(np.float32, copy=False)


def get_hog_feature_length():
    dummy_patch = np.zeros((PATCH_SIZE[1], PATCH_SIZE[0]), dtype=np.uint8)

    return len(extract_hog_feature(dummy_patch))


def create_face_mask(shape=(250, 250)):
    r, c = shape
    mask = np.zeros((r, c), dtype=np.float64)

    center_x = int(c / 2)
    center_y = int(r / 1.8)

    axis_x = int(c / 2.6)
    axis_y = int(r / 2)

    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 10)

    mask = np.power(mask, 1.5)

    return mask


def preprocess_face(img, face_detector, target_size=(250, 250)):
    # 1. To grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    # 2. Face detection (Haar Cascade or HOG)
    bbox = face_detector.detect(gray_img)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    # Check bounds to avoid cropping outside the image
    x1, y1, x2, y2 = _clamp_bbox_xyxy(
        (x1, y1, x2, y2), gray_img.shape[1], gray_img.shape[0]
    )
    face_crop = gray_img[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # 3. Resize to target size (250x250)
    face_resized = cv2.resize(face_crop, target_size)

    # 4. filtering (noise reduction but preserves edges) - Bilateral Filter
    face_denoised = cv2.bilateralFilter(face_resized, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_clahe = clahe.apply(face_denoised)

    # 6. Elliptical mask
    mask = create_face_mask(target_size)

    # Multiply pixel values by the mask to keep the face region and fade out the background
    face_processed = face_clahe.astype(np.float64) * mask
    face_processed = face_processed.astype(np.uint8)

    return face_processed
