from pathlib import Path
import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from face_recognition import *

import warnings

warnings.filterwarnings("ignore")


def load_data(data_dir: Path):
    X = []
    y = []
    class_names = []

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    for label_int, person_name in enumerate(classes):
        class_names.append(person_name)
        person_dir = data_dir / person_name

        for img_path in person_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                y.append(label_int)

    return X, np.array(y), class_names


def visualize_random_predictions(
    fisher_dict, lbph_model, raw_test_dir, face_detector, n_samples=6, cols=3
):
    pca = fisher_dict["pca"]
    lda = fisher_dict["lda"]
    classifier = fisher_dict["classifier"]
    class_names = fisher_dict["class_names"]

    raw_test_dir = Path(raw_test_dir)
    all_test_images = []

    for person_name in class_names:
        person_dir = raw_test_dir / person_name
        if not person_dir.exists():
            continue
        for img_path in person_dir.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                all_test_images.append((str(img_path), person_name))

    if len(all_test_images) == 0:
        print("Không tìm thấy ảnh nào trong thư mục Test Gốc!")
        return

    random.shuffle(all_test_images)

    valid_samples = []

    for img_path, true_name in all_test_images:
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            continue
        processed_img = preprocess_face(img_raw, face_detector, target_size=(250, 250))
        if processed_img is not None:
            valid_samples.append((img_raw, processed_img, true_name))
            if len(valid_samples) == n_samples:
                break
    if len(valid_samples) == 0:
        print("Không tìm thấy khuôn mặt nào trong các ảnh Test!")
        return

    rows = (len(valid_samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (img_raw, processed_img, true_name) in enumerate(valid_samples):
        ax = axes[i]
        img_flat = processed_img.flatten().reshape(1, -1)
        X_pca = pca.transform(img_flat)
        X_lda = lda.transform(X_pca)
        fisher_pred_idx = classifier.predict(X_lda)[0]
        fisher_pred_name = class_names[fisher_pred_idx]
        lbph_pred_idx, confidence = lbph_model.predict(processed_img)
        lbph_pred_name = class_names[lbph_pred_idx]
        title_text = (
            f"Thực tế: {true_name}\n"
            f"Fisher: {fisher_pred_name}\n"
            f"LBPH: {lbph_pred_name}"
        )

        if fisher_pred_name != true_name and lbph_pred_name != true_name:
            title_color = "red"
        elif fisher_pred_name == true_name and lbph_pred_name == true_name:
            title_color = "green"
        else:
            title_color = "grey"

        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")
        ax.set_title(
            title_text, color=title_color, fontsize=11, fontweight="bold", pad=10
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
