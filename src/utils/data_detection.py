import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

from config import RAW_DETECTION_DIR


def download_face_detection_dataset():
    dataset_name = "iamtushara/face-detection-dataset"

    target_dir = RAW_DETECTION_DIR / "kaggle_face_detection"

    if target_dir.exists():
        print(f"[Data detection] Removing old dataset folder: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Data detection] Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"[Data detection] Downloading dataset...")
    api.dataset_download_files(dataset_name, path=str(target_dir), unzip=True)

    print(f"[Data detection] Dataset downloaded to: {target_dir}")

    print(f"[Data detection] Dataset structure:")
    for path in target_dir.rglob("*"):
        print(path.relative_to(target_dir))


if __name__ == "__main__":
    download_face_detection_dataset()
