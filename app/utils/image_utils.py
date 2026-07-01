import streamlit as st
from pathlib import Path
from PIL import Image

from app.utils.paths import (
    UPLOAD_DIR,
    TRAINING_DIR,
    TESTING_DIR,
    VALIDATION_DIR,
    IMAGE_EXTS,
)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0

    total = 0
    for ext in IMAGE_EXTS:
        total += len(list(folder.rglob(f"*{ext}")))

    return total


def get_uploaded_images():
    return sorted([
        p for p in UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def clean_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def get_all_classes():
    class_dirs = [TRAINING_DIR, TESTING_DIR, VALIDATION_DIR]
    classes = set()

    for folder in class_dirs:
        if folder.exists():
            for p in folder.iterdir():
                if p.is_dir():
                    classes.add(p.name)

    return sorted(classes)


def create_class_folders(class_name: str):
    clean_name = clean_class_name(class_name)

    for bucket in [TRAINING_DIR, TESTING_DIR, VALIDATION_DIR]:
        class_path = bucket / clean_name
        class_path.mkdir(parents=True, exist_ok=True)

    return clean_name


def preview_image(image_path: Path, caption=True, width=None):
    try:
        img = Image.open(image_path)
        st.image(
            img,
            caption=image_path.name if caption else None,
            width=width,
            use_container_width=False if width else True,
        )
    except Exception:
        st.warning(f"Could not preview {image_path.name}")