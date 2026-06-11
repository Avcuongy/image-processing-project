from pathlib import Path

# Root project
ROOT_DIR = Path(__file__).resolve().parents[2]

# Main folders
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Detection data
RAW_DETECTION_DIR = RAW_DATA_DIR / "face_detection"
PROCESSED_DETECTION_DIR = PROCESSED_DATA_DIR / "face_detection"

# Recognition data
RAW_RECOGNITION_DIR = RAW_DATA_DIR / "face_recognition"
PROCESSED_RECOGNITION_DIR = PROCESSED_DATA_DIR / "face_recognition"

# Train / val / test
TRAIN_DETECTION = PROCESSED_DETECTION_DIR / "train"
VAL_DETECTION = PROCESSED_DETECTION_DIR / "val"
TEST_DETECTION = PROCESSED_DETECTION_DIR / "test"

TRAIN_RECOGNITION = PROCESSED_RECOGNITION_DIR / "train"
VAL_RECOGNITION = PROCESSED_RECOGNITION_DIR / "val"
TEST_RECOGNITION = PROCESSED_RECOGNITION_DIR / "test"

# Model folders
DETECTION_MODEL_DIR = ROOT_DIR / "models" / "face_detection"
RECOGNITION_MODEL_DIR = ROOT_DIR / "models" / "face_recognition"

# Image settings
FACE_SIZE = (100, 100)

# Random seed
RANDOM_STATE = 42

def load_detection_dataframe(images_dir, labels_dir, source_name):
    """
    Load image paths and corresponding YOLO label paths.

    Expected structure:
        images_dir/
            xxx.jpg
        labels_dir/
            xxx.txt

    Return:
        DataFrame with image_path, label_path, label_exists.
    """

    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(images_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    records = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"

        records.append({
            "source": source_name,
            "image_name": image_path.name,
            "image_stem": image_path.stem,
            "image_path": str(image_path),
            "label_name": label_path.name,
            "label_path": str(label_path),
            "label_exists": label_path.exists()
        })

    df = pd.DataFrame(records)

    return df

def read_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def clamp_bbox_xyxy(bbox, image_width, image_height):
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


def yolo_to_xyxy(x_center, y_center, box_width, box_height, image_width, image_height):
    values = [x_center, y_center, box_width, box_height]

    # YOLO normalized
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


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    output = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    return output

def load_haar_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Cannot load Haar Cascade model.")

    return detector

def detect_faces_haar(
    gray_image,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30)
):
    """
    Detect faces using Haar Cascade.

    Input:
        gray_image: preprocessed grayscale image

    Output:
        boxes_xyxy: list of boxes [x1, y1, x2, y2]
        elapsed_time: processing time
    """

    start_time = time.time()

    faces_xywh = haar_detector.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    elapsed_time = time.time() - start_time

    boxes_xyxy = []

    for (x, y, w, h) in faces_xywh:
        boxes_xyxy.append([
            int(x),
            int(y),
            int(x + w),
            int(y + h)
        ])

    return boxes_xyxy, elapsed_time

def draw_boxes(
    image,
    boxes,
    color=(0, 255, 0),
    thickness=2,
    label=None,
    show=False,
    title="Detection Result",
    figsize=(8, 6)
):

    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        if label is not None:
            cv2.putText(
                output,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return output

def evaluate_single_image(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluate detection result for one image.

    Logic:
        A prediction is TP if IoU >= threshold with an unmatched GT box.
    """

    matched_gt = set()

    true_positive = 0
    false_positive = 0

    iou_values = []

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = box_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            true_positive += 1
            matched_gt.add(best_gt_idx)
            iou_values.append(best_iou)
        else:
            false_positive += 1

    false_negative = len(gt_boxes) - len(matched_gt)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "matched_ious": iou_values
    }

def evaluate(
    images_dir,
    labels_dir,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30),
    iou_threshold=0.5,
    max_images=None
):
    image_paths = sorted(list(Path(images_dir).glob("*")))

    if max_images is not None:
        image_paths = image_paths[:max_images]

    records = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0.0
    all_ious = []

    for image_path in tqdm(image_paths, desc="Evaluating Haar Cascade"):
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if gray_image is None:
            continue

        height, width = gray_image.shape[:2]

        label_path = Path(labels_dir) / f"{image_path.stem}.txt"

        gt_boxes = read_yolo_label_file(
            label_path=label_path,
            image_width=width,
            image_height=height
        )

        pred_boxes, elapsed_time = detect_faces_haar(
            gray_image=gray_image,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size
        )

        result = evaluate_single_image(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=iou_threshold
        )

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_time += elapsed_time
        all_ious.extend(result["matched_ious"])

        records.append({
            "image_name": image_path.name,
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "time": elapsed_time
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    summary = {
        "scale_factor": scale_factor,
        "min_neighbors": min_neighbors,
        "min_size": min_size,
        "iou_threshold": iou_threshold,
        "total_images": len(image_paths),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_iou": avg_iou,
        "avg_time": avg_time,
        "fps": fps
    }

    records_df = pd.DataFrame(records)

    return summary, records_df

def get_image_paths(folder):
    folder = Path(folder)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(folder.glob(ext))

    return sorted(image_paths)

def read_gray_patch(image_path, patch_size=(64, 64)):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if image.shape[:2] != (patch_size[1], patch_size[0]):
        image = cv2.resize(
            image,
            patch_size,
            interpolation=cv2.INTER_AREA
        )

    return image


def extract_hog_feature(gray_patch):
    feature = hog(
        gray_patch,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        transform_sqrt=HOG_PARAMS["transform_sqrt"],
        feature_vector=HOG_PARAMS["feature_vector"]
    )

    return feature

def sample_paths(paths, max_count, seed=RANDOM_STATE):
    paths = list(paths)

    if max_count is None or len(paths) <= max_count:
        return paths

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=max_count, replace=False)

    return [paths[index] for index in sorted(indices)]


def get_hog_feature_length():
    dummy_patch = np.zeros((PATCH_SIZE[1], PATCH_SIZE[0]), dtype=np.uint8)

    return len(extract_hog_feature(dummy_patch))


def build_hog_dataset(
    positive_paths,
    negative_paths,
    dataset_name="dataset",
    max_per_class=None,
    use_cache=True
):
    positive_paths = sample_paths(
        positive_paths,
        max_count=max_per_class,
        seed=RANDOM_STATE
    )
    negative_paths = sample_paths(
        negative_paths,
        max_count=max_per_class,
        seed=RANDOM_STATE + 1
    )

    cache_name = f"{dataset_name}_hog_{len(positive_paths)}pos_{len(negative_paths)}neg.npz"
    cache_path = FEATURE_CACHE_DIR / cache_name

    if use_cache and cache_path.exists():
        cached = np.load(cache_path)
        print(f"Loaded cached {dataset_name} features: {cache_path}")

        return cached["X"], cached["y"]

    labeled_paths = [(path, 1) for path in positive_paths]
    labeled_paths += [(path, 0) for path in negative_paths]
    labeled_paths = shuffle(labeled_paths, random_state=RANDOM_STATE)

    feature_length = get_hog_feature_length()
    X = np.empty((len(labeled_paths), feature_length), dtype=np.float32)
    y = np.empty(len(labeled_paths), dtype=np.int32)

    write_index = 0
    failed_paths = []

    for image_path, label in tqdm(labeled_paths, desc=f"{dataset_name} HOG"):
        try:
            patch = read_gray_patch(image_path, patch_size=PATCH_SIZE)
            feature = extract_hog_feature(patch).astype(np.float32, copy=False)

            if feature.shape[0] != feature_length:
                raise ValueError(f"Unexpected HOG length: {feature.shape[0]}")

            X[write_index] = feature
            y[write_index] = label
            write_index += 1
        except Exception as error:
            failed_paths.append((str(image_path), str(error)))

    X = X[:write_index]
    y = y[:write_index]

    if failed_paths:
        print(f"Skipped {len(failed_paths)} unreadable/invalid patches.")

    if use_cache:
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"Saved cached {dataset_name} features: {cache_path}")

    return X, y

def image_pyramid(image, scale_factor=0.8, min_size=(64, 64)):
    current_image = image.copy()
    current_scale = 1.0

    yield current_scale, current_image

    while True:
        new_width = int(current_image.shape[1] * scale_factor)
        new_height = int(current_image.shape[0] * scale_factor)

        if new_width < min_size[0] or new_height < min_size[1]:
            break

        current_image = cv2.resize(
            current_image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        current_scale *= scale_factor

        yield current_scale, current_image


def sliding_window(image, window_size=(64, 64), step_size=16):
    window_w, window_h = window_size

    for y in range(0, image.shape[0] - window_h + 1, step_size):
        for x in range(0, image.shape[1] - window_w + 1, step_size):
            window = image[y:y + window_h, x:x + window_w]

            yield x, y, window

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection

        iou = intersection / np.maximum(union, 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    final_boxes = boxes[keep].astype(int).tolist()
    final_scores = scores[keep].tolist()

    return final_boxes, final_scores

def detect_faces_hog_svm(
    gray_image,
    score_threshold=0.8,
    scale_factor=0.8,
    step_size=16,
    nms_iou_threshold=0.3
):
    candidate_boxes = []
    candidate_scores = []

    start_time = time.time()

    for scale, resized_gray in image_pyramid(
        gray_image,
        scale_factor=scale_factor,
        min_size=PATCH_SIZE
    ):
        for x, y, window in sliding_window(
            resized_gray,
            window_size=PATCH_SIZE,
            step_size=step_size
        ):
            if window.shape[0] != PATCH_SIZE[1] or window.shape[1] != PATCH_SIZE[0]:
                continue

            feature = extract_hog_feature(window)
            feature = feature.reshape(1, -1)

            score = hog_svm_model.decision_function(feature)[0]

            if score >= score_threshold:
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + PATCH_SIZE[0]) / scale)
                y2 = int((y + PATCH_SIZE[1]) / scale)

                candidate_boxes.append([x1, y1, x2, y2])
                candidate_scores.append(float(score))

    final_boxes, final_scores = non_max_suppression(
        candidate_boxes,
        candidate_scores,
        iou_threshold=nms_iou_threshold
    )

    elapsed_time = time.time() - start_time

    return final_boxes, final_scores, elapsed_time

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, label=None):
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        if label is not None:
            cv2.putText(
                output,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return output

def check_detection_folder(images_dir, labels_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_paths = sorted([
        p for p in images_dir.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    label_paths = sorted(labels_dir.glob("*.txt"))

    print("Images dir exists:", images_dir.exists())
    print("Labels dir exists:", labels_dir.exists())
    print("Number of images:", len(image_paths))
    print("Number of labels:", len(label_paths))

    if len(image_paths) == 0:
        raise RuntimeError(
            "Không tìm thấy ảnh đã preprocessing. "
            "Hãy chạy notebook 1_preprocessing.ipynb trước."
        )

    return image_paths, label_paths

def find_existing_lbp_cascade():
    """
    Tìm file LBP Cascade ở các vị trí thường gặp.
    """
    candidate_paths = [
        LBP_CASCADE_PATH,
        Path(cv2.data.haarcascades) / LBP_CASCADE_FILENAME,
        Path(cv2.data.haarcascades).parent / "lbpcascades" / LBP_CASCADE_FILENAME,
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    return None


def download_lbp_cascade(save_path=LBP_CASCADE_PATH):
    """
    Tải lbpcascade_frontalface.xml nếu local chưa có.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None

    for url in OPENCV_LBP_URLS:
        try:
            print("Downloading LBP Cascade from:")
            print(url)

            urllib.request.urlretrieve(url, save_path)

            if save_path.exists() and save_path.stat().st_size > 0:
                print("Downloaded to:", save_path)
                return save_path

        except Exception as error:
            last_error = error
            print("Download failed:", error)

    raise RuntimeError(
        "Không thể tải lbpcascade_frontalface.xml tự động.\\n"
        f"Hãy tải file thủ công và đặt tại: {save_path}\\n"
        "Gợi ý nguồn: opencv/data/lbpcascades/lbpcascade_frontalface.xml\\n"
        f"Lỗi cuối cùng: {last_error}"
    )


def prepare_lbp_cascade_file():
    existing_path = find_existing_lbp_cascade()

    if existing_path is not None:
        print("Found existing LBP Cascade file:")
        print(existing_path)

        # Copy về thư mục models/face_detection để quản lý thống nhất
        if existing_path != LBP_CASCADE_PATH:
            LBP_CASCADE_PATH.write_bytes(existing_path.read_bytes())
            print("Copied to project model folder:")
            print(LBP_CASCADE_PATH)

        return LBP_CASCADE_PATH

    return download_lbp_cascade(LBP_CASCADE_PATH)


def load_lbp_detector():
    cascade_path = prepare_lbp_cascade_file()

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError(
            "Cannot load LBP Cascade model. "
            "File XML có thể bị thiếu, sai định dạng hoặc tải chưa hoàn chỉnh."
        )

    return detector, cascade_path

def detect_faces_lbp(
    gray_image,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30)
):
    """
    Detect faces using LBP Cascade.

    Input:
        gray_image: ảnh grayscale đã preprocessing

    Output:
        boxes_xyxy: list bbox theo format [x1, y1, x2, y2]
        elapsed_time: thời gian xử lý
    """

    start_time = time.time()

    faces_xywh = lbp_detector.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    elapsed_time = time.time() - start_time

    boxes_xyxy = []

    for (x, y, w, h) in faces_xywh:
        boxes_xyxy.append([
            int(x),
            int(y),
            int(x + w),
            int(y + h)
        ])

    return boxes_xyxy, elapsed_time

def draw_boxes(
    image,
    boxes,
    color=(0, 255, 0),
    thickness=2,
    label=None,
    show=False,
    title="Detection Result",
    figsize=(8, 6)
):
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        if label is not None:
            cv2.putText(
                output,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return output

def evaluate_single_image(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluate detection result for one image.

    Logic:
        Một prediction là TP nếu IoU >= threshold với một GT box chưa được match.
    """

    matched_gt = set()

    true_positive = 0
    false_positive = 0

    iou_values = []

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = box_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            true_positive += 1
            matched_gt.add(best_gt_idx)
            iou_values.append(best_iou)
        else:
            false_positive += 1

    false_negative = len(gt_boxes) - len(matched_gt)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "matched_ious": iou_values
    }

def evaluate_lbp(
    images_dir,
    labels_dir,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30),
    iou_threshold=0.5,
    max_images=None
):
    image_paths = sorted([
        p for p in Path(images_dir).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    if max_images is not None:
        image_paths = image_paths[:max_images]

    records = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0.0
    all_ious = []

    for image_path in tqdm(image_paths, desc="Evaluating LBP Cascade"):
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if gray_image is None:
            continue

        height, width = gray_image.shape[:2]

        label_path = Path(labels_dir) / f"{image_path.stem}.txt"

        gt_boxes = read_yolo_label_file(
            label_path=label_path,
            image_width=width,
            image_height=height
        )

        pred_boxes, elapsed_time = detect_faces_lbp(
            gray_image=gray_image,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size
        )

        result = evaluate_single_image(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=iou_threshold
        )

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_time += elapsed_time
        all_ious.extend(result["matched_ious"])

        records.append({
            "image_name": image_path.name,
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "time": elapsed_time
        })

    evaluated_images = len(image_paths)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    avg_time = total_time / evaluated_images if evaluated_images > 0 else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    summary = {
        "method": "LBP Cascade",
        "scale_factor": scale_factor,
        "min_neighbors": min_neighbors,
        "min_size": min_size,
        "iou_threshold": iou_threshold,
        "total_images": evaluated_images,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_iou": avg_iou,
        "avg_time": avg_time,
        "fps": fps
    }

    records_df = pd.DataFrame(records)

    return summary, records_df

def list_class_dirs(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    return sorted([d for d in root_dir.iterdir() if d.is_dir()])


def list_images(folder):
    folder = Path(folder)
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        images.extend(folder.glob(ext))
    return sorted(images)


def pick_sample_image(root_dir):
    root_dir = Path(root_dir)
    class_dirs = list_class_dirs(root_dir)

    if not class_dirs:
        return None, None

    class_dir = random.choice(class_dirs)
    image_paths = list_images(class_dir)

    if not image_paths:
        return class_dir.name, None

    return class_dir.name, random.choice(image_paths)


def show_image(image_path, title):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    return image_path