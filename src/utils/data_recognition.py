import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def download_face_recognition_dataset():
    dataset_name = "vasukipatel/face-recognition-dataset"

    path = kagglehub.dataset_download(
        dataset_name, force_download=True, output_dir=str(DATA_DIR)
    )

    print(f"[Data recognition] Dataset downloaded to: {path}")


if __name__ == "__main__":
    download_face_recognition_dataset()
