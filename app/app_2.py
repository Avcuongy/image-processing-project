CONFIDENCE_TH = 70

HAAR_CASCADE_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
LBPH_MODEL = ".output/lbph_model.yml"
DATASET_DIR = "dataset"
LABEL_MAP = ".output/label_map.json"

CAPTURE_COUNT = 20
SKIP_FRAMES = 2
TARGET_SIZE = (250, 250)

BLUR_THRESHOLD = 50.0
BRIGHTNESS_MIN = 40.0
BRIGHTNESS_MAX = 200.0

C_BG = "#0f1117"
C_PANEL = "#1a1d27"
C_ACCENT = "#00e5ff"
C_GREEN = "#00e676"
C_RED = "#ff1744"
C_WARN = "#ffd600"
C_TEXT = "#e0e0e0"
C_MUTED = "#616161"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(".output", exist_ok=True)

def load_label_map():
    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_label_map(lmap):
    with open(LABEL_MAP, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in lmap.items()}, f, ensure_ascii=False, indent=2)


def next_label_id(lmap):
    return max(lmap.keys(), default=-1) + 1


def train_lbph():
    faces, labels = [], []
    for folder in os.listdir(DATASET_DIR):
        fpath = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(fpath):
            continue
        try:
            label = int(folder)
        except ValueError:
            continue
        for img_name in os.listdir(fpath):
            img_path = os.path.join(fpath, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label)
    if not faces:
        return False, "Không có dữ liệu để huấn luyện."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(LBPH_MODEL)
    return True, f"Huấn luyện xong: {len(faces)} ảnh, {len(set(labels))} người."