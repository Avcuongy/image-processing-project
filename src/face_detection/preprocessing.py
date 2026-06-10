from pathlib import Path
import random

import cv2
import numpy as np


def read_image(image_path):
    """
    Read image from path using OpenCV.

    Return:
        BGR image
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def save_image(image, save_path):
    """
    Save image to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(save_path), image)

    if not success:
        raise ValueError(f"Cannot save image to: {save_path}")


# Grayscale + Gamma + CLAHE
def to_gray(image):
    """
    Convert BGR image to grayscale.
    If image is already grayscale, return a copy.
    """
    if len(image.shape) == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gamma_correction(gray_image, gamma=1.0):
    """
    Apply gamma correction.

    Formula:
        output = 255 * (input / 255) ^ gamma

    gamma < 1.0: brighter
    gamma = 1.0: unchanged
    gamma > 1.0: darker
    """
    if gamma <= 0:
        raise ValueError("gamma must be greater than 0.")

    gray_image = to_gray(gray_image)

    normalized = gray_image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    return corrected


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply light CLAHE for local contrast enhancement.
    """
    gray_image = to_gray(gray_image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(gray_image)


def preprocess_for_detection(
    image,
    gamma=1.0,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8)
):
    """
    Common preprocessing for both Haar Cascade and HOG + SVM.

    Pipeline:
        BGR image
            -> Grayscale
            -> Gamma Correction
            -> CLAHE
    """
    gray = to_gray(image)

    gray = apply_gamma_correction(
        gray,
        gamma=gamma
    )

    gray = apply_clahe(
        gray,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size
    )

    return gray


# HOG patch preprocessing
def resize_patch_for_hog(patch, patch_size=(64, 64)):
    """
    Resize cropped patch/window to fixed size for HOG + SVM.

    Input:
        patch: grayscale or BGR patch

    Output:
        grayscale patch with size 64x64
    """
    gray_patch = to_gray(patch)

    resized_patch = cv2.resize(
        gray_patch,
        patch_size,
        interpolation=cv2.INTER_AREA
    )

    return resized_patch


# Bounding box utilities
def clamp_bbox_xyxy(bbox, image_width, image_height):
    """
    Clamp bbox to image boundary.

    bbox format:
        x1, y1, x2, y2
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))

    if x2 <= x1:
        x2 = min(image_width, x1 + 1)

    if y2 <= y1:
        y2 = min(image_height, y1 + 1)

    return x1, y1, x2, y2


def yolo_to_xyxy(
    x_center,
    y_center,
    box_width,
    box_height,
    image_width,
    image_height
):
    """
    Convert YOLO bbox to xyxy format.

    YOLO format:
        class_id x_center y_center width height

    If values <= 1.5, they are treated as normalized coordinates.
    """
    values = [x_center, y_center, box_width, box_height]

    if max(values) <= 1.5:
        x_center *= image_width
        y_center *= image_height
        box_width *= image_width
        box_height *= image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return clamp_bbox_xyxy(
        bbox=(x1, y1, x2, y2),
        image_width=image_width,
        image_height=image_height
    )


def read_yolo_label_file(label_path, image_width, image_height):
    """
    Read YOLO label file.

    Expected format per line:
        class_id x_center y_center width height

    Return:
        list of boxes in xyxy format
    """
    label_path = Path(label_path)

    if not label_path.exists():
        return []

    boxes = []

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        box = yolo_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            box_width=box_width,
            box_height=box_height,
            image_width=image_width,
            image_height=image_height
        )

        boxes.append(box)

    return boxes


def crop_bbox_xyxy(image, bbox, padding=0.0):
    """
    Crop image by xyxy bbox.

    padding:
        0.15 means expand bbox by 15% on each side.
    """
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1, y1, x2, y2 = clamp_bbox_xyxy(
        bbox=(x1, y1, x2, y2),
        image_width=image_width,
        image_height=image_height
    )

    return image[y1:y2, x1:x2]


def box_iou(box_a, box_b):
    """
    Calculate IoU between two boxes.

    box format:
        x1, y1, x2, y2
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# Negative sampling for HOG + SVM
def generate_random_negative_box(
    image_width,
    image_height,
    min_size=64,
    max_size=256
):
    """
    Generate random square box inside image.
    """
    max_possible_size = min(image_width, image_height, max_size)

    if max_possible_size <= min_size:
        size = min_size
    else:
        size = random.randint(min_size, max_possible_size)

    x1 = random.randint(0, max(0, image_width - size))
    y1 = random.randint(0, max(0, image_height - size))

    x2 = x1 + size
    y2 = y1 + size

    return clamp_bbox_xyxy(
        bbox=(x1, y1, x2, y2),
        image_width=image_width,
        image_height=image_height
    )


def is_valid_negative_box(candidate_box, face_boxes, iou_threshold=0.1):
    """
    A negative box is valid if it has low overlap with all face boxes.
    """
    for face_box in face_boxes:
        iou = box_iou(candidate_box, face_box)

        if iou > iou_threshold:
            return False

    return True


def generate_negative_boxes(
    image_width,
    image_height,
    face_boxes,
    num_negatives=1,
    iou_threshold=0.1,
    max_attempts=100
):
    """
    Generate negative boxes that do not overlap much with face boxes.
    """
    negative_boxes = []
    attempts = 0

    while len(negative_boxes) < num_negatives and attempts < max_attempts:
        attempts += 1

        candidate_box = generate_random_negative_box(
            image_width=image_width,
            image_height=image_height
        )

        if is_valid_negative_box(
            candidate_box=candidate_box,
            face_boxes=face_boxes,
            iou_threshold=iou_threshold
        ):
            negative_boxes.append(candidate_box)

    return negative_boxes


def create_preprocessed_detection(
    df,
    output_dir,
    gamma=1.0,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    image_ext=".png"
):
    """
    Save preprocessed full images and corresponding YOLO labels.

    Input df must contain:
        image_path
        label_path

    Output:
        output_dir/images/*.png
        output_dir/labels/*.txt

    Pipeline:
        original image
            -> grayscale + gamma correction + CLAHE
            -> save full-size preprocessed image

    Note:
        The image size is unchanged, so the original YOLO labels remain valid.
        This function is suitable for:
            - Haar Cascade validation/testing
            - HOG + SVM detector-level validation/testing
    """

    output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not image_ext.startswith("."):
        image_ext = "." + image_ext

    total_images_saved = 0
    total_labels_saved = 0
    total_missing_labels = 0
    total_failed = 0

    records = []

    for _, row in df.iterrows():
        try:
            image_path = Path(row["image_path"])
            label_path = Path(row["label_path"])

            original_image = read_image(image_path)

            preprocessed_image = preprocess_for_detection(
                image=original_image,
                gamma=gamma,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size
            )

            output_image_path = images_dir / f"{image_path.stem}{image_ext}"
            output_label_path = labels_dir / f"{image_path.stem}.txt"

            save_image(preprocessed_image, output_image_path)
            total_images_saved += 1

            if label_path.exists():
                output_label_path.write_bytes(label_path.read_bytes())
                total_labels_saved += 1
                label_exists = True
            else:
                total_missing_labels += 1
                label_exists = False

            records.append({
                "image_name": image_path.name,
                "processed_image_name": output_image_path.name,
                "label_name": output_label_path.name,
                "source_image_path": str(image_path),
                "source_label_path": str(label_path),
                "processed_image_path": str(output_image_path),
                "processed_label_path": str(output_label_path),
                "label_exists": label_exists,
                "status": "ok"
            })

        except Exception as error:
            total_failed += 1

            records.append({
                "image_name": row.get("image_name", None),
                "processed_image_name": None,
                "label_name": None,
                "source_image_path": row.get("image_path", None),
                "source_label_path": row.get("label_path", None),
                "processed_image_path": None,
                "processed_label_path": None,
                "label_exists": False,
                "status": "failed",
                "error": str(error)
            })

    summary = {
        "total_images_saved": total_images_saved,
        "total_labels_saved": total_labels_saved,
        "total_missing_labels": total_missing_labels,
        "total_failed": total_failed
    }

    return records, summary


# Create positive/negative images for HOG + SVM
def create_hog_positive_negative_patches(
    df,
    output_dir,
    patch_size=(64, 64),
    gamma=1.0,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    positive_padding=0.15,
    negative_per_face=1,
    negative_iou_threshold=0.1
):
    """
    Create positive and negative image patches for HOG + SVM.

    Input df must contain:
        image_path
        label_path

    Output:
        output_dir/positive/*.png
        output_dir/negative/*.png

    Pipeline:
        original image
            -> grayscale + gamma + CLAHE
            -> crop positive/negative
            -> resize 64x64
            -> save patch images
    """
    output_dir = Path(output_dir)

    positive_dir = output_dir / "positive"
    negative_dir = output_dir / "negative"

    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    total_positive = 0
    total_negative = 0
    total_failed = 0

    records = []

    for _, row in df.iterrows():
        try:
            image_path = Path(row["image_path"])
            label_path = Path(row["label_path"])

            original_image = read_image(image_path)
            image_height, image_width = original_image.shape[:2]

            face_boxes = read_yolo_label_file(
                label_path=label_path,
                image_width=image_width,
                image_height=image_height
            )

            preprocessed_image = preprocess_for_detection(
                image=original_image,
                gamma=gamma,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size
            )

            current_positive = 0
            current_negative = 0

            # Positive patches
            for face_idx, face_box in enumerate(face_boxes):
                face_crop = crop_bbox_xyxy(
                    image=preprocessed_image,
                    bbox=face_box,
                    padding=positive_padding
                )

                if face_crop.size == 0:
                    continue

                face_patch = resize_patch_for_hog(
                    patch=face_crop,
                    patch_size=patch_size
                )

                save_path = positive_dir / f"{image_path.stem}_face_{face_idx}.png"
                save_image(face_patch, save_path)

                total_positive += 1
                current_positive += 1

            # Negative patches
            if len(face_boxes) > 0:
                num_negatives = max(1, len(face_boxes) * negative_per_face)
            else:
                num_negatives = 1

            negative_boxes = generate_negative_boxes(
                image_width=image_width,
                image_height=image_height,
                face_boxes=face_boxes,
                num_negatives=num_negatives,
                iou_threshold=negative_iou_threshold
            )

            for neg_idx, neg_box in enumerate(negative_boxes):
                neg_crop = crop_bbox_xyxy(
                    image=preprocessed_image,
                    bbox=neg_box,
                    padding=0.0
                )

                if neg_crop.size == 0:
                    continue

                neg_patch = resize_patch_for_hog(
                    patch=neg_crop,
                    patch_size=patch_size
                )

                save_path = negative_dir / f"{image_path.stem}_neg_{neg_idx}.png"
                save_image(neg_patch, save_path)

                total_negative += 1
                current_negative += 1

            records.append({
                "image_name": image_path.name,
                "num_faces": len(face_boxes),
                "num_positive_saved": current_positive,
                "num_negative_saved": current_negative,
                "status": "ok"
            })

        except Exception as error:
            total_failed += 1

            records.append({
                "image_name": row.get("image_name", None),
                "num_faces": None,
                "num_positive_saved": 0,
                "num_negative_saved": 0,
                "status": "failed",
                "error": str(error)
            })

    summary = {
        "total_positive": total_positive,
        "total_negative": total_negative,
        "total_failed": total_failed
    }

    return records, summary