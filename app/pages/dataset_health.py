import pandas as pd
import streamlit as st
from PIL import Image

from app.utils.paths import (
    IMAGE_EXTS,
    TESTING_DIR,
    TRAINING_DIR,
    UPLOAD_DIR,
    VALIDATION_DIR,
)


def render_dataset_health_tab():
    st.header("Dataset Health Checker")

    scan_dirs = {
        "Uploads": UPLOAD_DIR,
        "Training": TRAINING_DIR,
        "Validation": VALIDATION_DIR,
        "Testing": TESTING_DIR,
    }

    if not st.button("Run Dataset Health Scan", key="run_dataset_health_scan"):
        return

    broken_images = []
    tiny_images = []
    wrong_file_types = []
    empty_class_folders = []
    duplicate_filenames = {}

    all_seen_names = {}

    for bucket_name, bucket_dir in scan_dirs.items():
        if not bucket_dir.exists():
            continue

        for path in bucket_dir.rglob("*"):
            if path.is_dir():
                if path != bucket_dir and not any(path.iterdir()):
                    empty_class_folders.append({
                        "Bucket": bucket_name,
                        "Folder": str(path)
                    })
                continue

            if path.is_file():
                if path.suffix.lower() not in IMAGE_EXTS:
                    wrong_file_types.append({
                        "Bucket": bucket_name,
                        "File": str(path),
                        "Extension": path.suffix
                    })
                    continue

                if path.name not in all_seen_names:
                    all_seen_names[path.name] = []

                all_seen_names[path.name].append(str(path))

                try:
                    with Image.open(path) as img:
                        width, height = img.size

                        if width < 100 or height < 100:
                            tiny_images.append({
                                "Bucket": bucket_name,
                                "File": str(path),
                                "Width": width,
                                "Height": height
                            })

                        img.verify()

                except Exception:
                    broken_images.append({
                        "Bucket": bucket_name,
                        "File": str(path)
                    })

    duplicate_filenames = {
        name: locations
        for name, locations in all_seen_names.items()
        if len(locations) > 1
    }

    st.success("Dataset health scan complete.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Broken Images", len(broken_images))

    with col2:
        st.metric("Tiny Images", len(tiny_images))

    with col3:
        st.metric("Wrong File Types", len(wrong_file_types))

    with col4:
        st.metric("Duplicate Filenames", len(duplicate_filenames))

    st.divider()

    if broken_images:
        st.subheader("Broken Images")
        st.dataframe(pd.DataFrame(broken_images), use_container_width=True)

    if tiny_images:
        st.subheader("Very Small Images")
        st.dataframe(pd.DataFrame(tiny_images), use_container_width=True)

    if wrong_file_types:
        st.subheader("Wrong File Types")
        st.dataframe(pd.DataFrame(wrong_file_types), use_container_width=True)

    if empty_class_folders:
        st.subheader("Empty Class Folders")
        st.dataframe(pd.DataFrame(empty_class_folders), use_container_width=True)

    if duplicate_filenames:
        st.subheader("Duplicate Filenames")

        duplicate_rows = []

        for name, locations in duplicate_filenames.items():
            duplicate_rows.append({
                "Filename": name,
                "Count": len(locations),
                "Locations": " | ".join(locations)
            })

        st.dataframe(pd.DataFrame(duplicate_rows), use_container_width=True)

    if not any([
        broken_images,
        tiny_images,
        wrong_file_types,
        empty_class_folders,
        duplicate_filenames
    ]):
        st.success("No obvious dataset problems found.")
