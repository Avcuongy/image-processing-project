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
