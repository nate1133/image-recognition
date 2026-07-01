from pathlib import Path

BASE_DIR = Path("/home/homeserver/projects/image-recognition-lab")
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"
TRAINING_DIR = DATA_DIR / "training"
TESTING_DIR = DATA_DIR / "testing"
VALIDATION_DIR = DATA_DIR / "validation"
MODEL_DIR = BASE_DIR / "models"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

for folder in [
    UPLOAD_DIR,
    TRAINING_DIR,
    TESTING_DIR,
    VALIDATION_DIR,
    MODEL_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)