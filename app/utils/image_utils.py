import re
from pathlib import Path

import streamlit as st
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

    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def get_uploaded_images():
    return sorted([
        p for p in UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def clean_class_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned.lower() or "unnamed"


def clean_file_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_. -]+", "_", Path(name).name.strip())
    cleaned = cleaned.strip(" ._-")
    return cleaned or "uploaded_file"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 1

    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")

        if not candidate.exists():
            return candidate

        counter += 1


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
            width=width or "stretch",
        )
    except Exception:
        st.warning(f"Could not preview {image_path.name}")
