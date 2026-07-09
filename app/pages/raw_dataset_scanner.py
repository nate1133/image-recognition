from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd
import streamlit as st

from src.data.prepare_dataset import import_selected_classes
from src.data.scan_dataset import scan_dataset
from app.utils.paths import (
    IMAGE_EXTS,
    RAW_DIR,
    TESTING_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
)
from app.utils.image_utils import clean_class_name, clean_file_name, unique_path


IMAGE_TYPES = [ext.lstrip(".") for ext in IMAGE_EXTS]


def is_safe_zip_member(member_name: str) -> bool:
    member_path = PurePosixPath(member_name)

    if member_path.is_absolute():
        return False

    return ".." not in member_path.parts


def render_raw_dataset_file_picker(raw_dir: Path):
    st.subheader("Add Raw Dataset Files")

    upload_mode = st.radio(
        "File source",
        ["Image files", "ZIP dataset"],
        horizontal=True,
        key="raw_dataset_upload_mode"
    )

    if upload_mode == "Image files":
        dataset_name = st.text_input(
            "Dataset name",
            value="manual_upload",
            key="raw_upload_dataset_name"
        )

        class_name = st.text_input(
            "Class name",
            placeholder="Example: toyota",
            key="raw_upload_class_name"
        )

        uploaded_files = st.file_uploader(
            "Choose raw image files",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
            key="raw_image_file_uploader"
        )

        if st.button("Save Raw Image Files", key="save_raw_image_files"):
            if not uploaded_files:
                st.warning("Choose at least one image file first.")
                return

            if not dataset_name.strip():
                st.warning("Enter a dataset name.")
                return

            if not class_name.strip():
                st.warning("Enter a class name.")
                return

            dataset_dir = raw_dir / clean_class_name(dataset_name)
            class_dir = dataset_dir / clean_class_name(class_name)
            class_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0

            for file in uploaded_files:
                save_path = unique_path(class_dir / clean_file_name(file.name))

                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

                saved_count += 1

            st.session_state.scan_results = scan_dataset(raw_dir)
            st.success(f"Saved {saved_count} raw image file(s).")

        return

    dataset_name = st.text_input(
        "Dataset name for ZIP import",
        value="zip_upload",
        key="zip_upload_dataset_name"
    )

    uploaded_zip = st.file_uploader(
        "Choose a ZIP dataset",
        type=["zip"],
        key="raw_zip_file_uploader"
    )

    if st.button("Extract ZIP Dataset", key="extract_raw_zip_dataset"):
        if not uploaded_zip:
            st.warning("Choose a ZIP file first.")
            return

        if not dataset_name.strip():
            st.warning("Enter a dataset name.")
            return

        dataset_dir = raw_dir / clean_class_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0
        skipped_count = 0

        try:
            with ZipFile(uploaded_zip) as zip_file:
                for member in zip_file.infolist():
                    if member.is_dir():
                        continue

                    if not is_safe_zip_member(member.filename):
                        skipped_count += 1
                        continue

                    member_path = PurePosixPath(member.filename)

                    if member_path.suffix.lower() not in IMAGE_EXTS:
                        skipped_count += 1
                        continue

                    destination = unique_path(dataset_dir / Path(*member_path.parts))
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    with zip_file.open(member) as source:
                        with open(destination, "wb") as target:
                            target.write(source.read())

                    extracted_count += 1

        except Exception as e:
            st.error("Could not extract ZIP dataset.")
            st.exception(e)
            return

        st.session_state.scan_results = scan_dataset(raw_dir)
        st.success(f"Extracted {extracted_count} image file(s).")

        if skipped_count:
            st.warning(f"Skipped {skipped_count} unsupported or unsafe ZIP item(s).")


def render_raw_dataset_scanner_tab():
    st.header("Raw Dataset Scanner / Import Wizard")

    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    st.write(f"Scanning: `{raw_dir}`")

    render_raw_dataset_file_picker(raw_dir)

    st.divider()

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = scan_dataset(raw_dir)

    if st.button("Scan Raw Datasets"):
        with st.spinner("Scanning datasets..."):
            results = scan_dataset(raw_dir)

        st.session_state.scan_results = results

    results = st.session_state.scan_results

    if results is None:
        st.info("Click Scan Raw Datasets to inspect your raw image folders.")
        return

    if len(results) == 0:
        st.warning("No image folders found.")
        return

    df = pd.DataFrame(results)

    st.success(
        f"Found {len(df)} image folders with "
        f"{df['images'].sum():,} total images."
    )

    st.dataframe(df, width="stretch")

    st.divider()

    st.subheader("Import Selected Classes")

    available_classes = sorted(df["class"].unique())

    selected_classes = st.multiselect(
        "Choose classes to import",
        available_classes
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        train_pct = st.number_input(
            "Training %",
            min_value=1,
            max_value=98,
            value=80
        )

    with col2:
        val_pct = st.number_input(
            "Validation %",
            min_value=1,
            max_value=98,
            value=10
        )

    with col3:
        test_pct = st.number_input(
            "Testing %",
            min_value=1,
            max_value=98,
            value=10
        )

    clear_existing = st.checkbox(
        "Clear existing folders for selected classes before importing",
        value=False
    )

    pct_total = train_pct + val_pct + test_pct

    if pct_total != 100:
        st.error("Training + Validation + Testing must equal 100.")
        return

    import_button = st.button("Import Selected Classes")

    if not import_button:
        return

    if not selected_classes:
        st.error("Select at least one class.")
        return

    selected_df = df[df["class"].isin(selected_classes)]
    selected_rows = selected_df.to_dict("records")

    with st.spinner("Importing and splitting images..."):
        summary = import_selected_classes(
            selected_rows=selected_rows,
            training_dir=TRAINING_DIR,
            validation_dir=VALIDATION_DIR,
            testing_dir=TESTING_DIR,
            train_pct=train_pct / 100,
            val_pct=val_pct / 100,
            test_pct=test_pct / 100,
            clear_existing=clear_existing
        )

    summary_df = pd.DataFrame(summary)

    st.success("Import complete.")
    st.dataframe(summary_df, width="stretch")

    st.rerun()
