from pathlib import Path
import os
import cv2
import glob
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from face_recognition import *

import warnings

warnings.filterwarnings("ignore")


def load_processed_data(data_dir):
    X = []
    y = []
    class_names = []

    classes = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    classes.sort()

    for label, person_name in enumerate(classes):
        class_names.append(person_name)
        person_dir = os.path.join(data_dir, person_name)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img.flatten())
                y.append(label)

    return np.array(X), np.array(y), class_names


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
