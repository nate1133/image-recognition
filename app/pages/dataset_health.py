import pandas as pd
import streamlit as st

from app.utils.paths import (
    IMAGE_EXTS,
    TESTING_DIR,
    TRAINING_DIR,
    UPLOAD_DIR,
    VALIDATION_DIR,
)
from src.data.dataset_health import scan_dataset_health


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

    results = scan_dataset_health(
        scan_dirs=scan_dirs,
        image_exts=IMAGE_EXTS,
    )

    broken_images = results["broken_images"]
    tiny_images = results["tiny_images"]
    wrong_file_types = results["wrong_file_types"]
    empty_class_folders = results["empty_class_folders"]
    duplicate_filenames = results["duplicate_filenames"]
    duplicate_images = results["duplicate_images"]

    st.success("Dataset health scan complete.")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Broken Images", len(broken_images))

    with col2:
        st.metric("Tiny Images", len(tiny_images))

    with col3:
        st.metric("Wrong File Types", len(wrong_file_types))

    with col4:
        st.metric("Duplicate Filenames", len(duplicate_filenames))

    with col5:
        st.metric("Duplicate Images", len(duplicate_images))

    st.divider()

    if broken_images:
        st.subheader("Broken Images")
        st.dataframe(pd.DataFrame(broken_images), width="stretch")

    if tiny_images:
        st.subheader("Very Small Images")
        st.dataframe(pd.DataFrame(tiny_images), width="stretch")

    if wrong_file_types:
        st.subheader("Wrong File Types")
        st.dataframe(pd.DataFrame(wrong_file_types), width="stretch")

    if empty_class_folders:
        st.subheader("Empty Class Folders")
        st.dataframe(pd.DataFrame(empty_class_folders), width="stretch")

    if duplicate_filenames:
        st.subheader("Duplicate Filenames")

        duplicate_rows = []

        for name, locations in duplicate_filenames.items():
            duplicate_rows.append({
                "Filename": name,
                "Count": len(locations),
                "Locations": " | ".join(locations)
            })

        st.dataframe(pd.DataFrame(duplicate_rows), width="stretch")

    if duplicate_images:
        st.subheader("Duplicate Images")

        duplicate_image_rows = []

        for image_hash, locations in duplicate_images.items():
            duplicate_image_rows.append({
                "Image Hash": image_hash,
                "Count": len(locations),
                "Locations": " | ".join(
                    location["File"] for location in locations
                ),
            })

        st.dataframe(pd.DataFrame(duplicate_image_rows), width="stretch")

    if not any([
        broken_images,
        tiny_images,
        wrong_file_types,
        empty_class_folders,
        duplicate_filenames,
        duplicate_images,
    ]):
        st.success("No obvious dataset problems found.")
