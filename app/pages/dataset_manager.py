import random
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils.paths import (
    IMAGE_EXTS,
    RAW_DIR,
    TESTING_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
)
from app.utils.image_utils import (
    clean_class_name,
    count_images,
    create_class_folders,
    get_all_classes,
    get_uploaded_images,
    preview_image,
)


def render_dataset_manager_tab():
    st.header("Dataset Class Manager")

    st.write(
        "Create labels/classes, then move uploaded images into "
        "training, testing, or validation folders."
    )

    class_dirs = {
        "Training": TRAINING_DIR,
        "Testing": TESTING_DIR,
        "Validation": VALIDATION_DIR,
    }

    st.subheader("Create New Class")

    new_class = st.text_input(
        "New class name",
        placeholder="Example: toyota"
    )

    if st.button("Create Class"):
        if not new_class.strip():
            st.error("Enter a class name first.")
        else:
            created_class = create_class_folders(new_class)
            st.success(f"Created class folders for: {created_class}")
            st.rerun()

    st.divider()

    st.subheader("Current Classes")

    all_classes = get_all_classes()

    if not all_classes:
        st.info("No classes created yet.")
    else:
        rows = []

        for class_name in all_classes:
            row = {"Class": class_name}

            for bucket_name, bucket_dir in class_dirs.items():
                row[bucket_name] = count_images(bucket_dir / class_name)

            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("Move Uploaded Images Into Buckets")

    uploaded_images = get_uploaded_images()

    if not uploaded_images:
        st.info("No uploaded images waiting to be sorted.")
    elif not all_classes:
        st.warning("Create at least one class before sorting images.")
    else:
        selected_image = st.selectbox(
            "Select uploaded image",
            uploaded_images,
            format_func=lambda x: x.name
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_class = st.selectbox("Class", all_classes)

        with col2:
            selected_bucket = st.selectbox(
                "Bucket",
                ["Training", "Testing", "Validation"]
            )

        with col3:
            st.write("")
            st.write("")
            move_button = st.button("Move Image")

        if selected_image:
            preview_image(selected_image, width=350)

        if move_button:
            destination = (
                class_dirs[selected_bucket]
                / selected_class
                / selected_image.name
            )

            if destination.exists():
                st.error("That file already exists in the destination folder.")
            else:
                selected_image.rename(destination)
                st.success(
                    f"Moved {selected_image.name} to "
                    f"{selected_bucket}/{selected_class}"
                )
                st.rerun()

    st.divider()
    st.subheader("Bulk Dataset Splitter")

    st.write(
        "Move images from one raw class folder into training, validation, and testing folders."
    )

    raw_source_dir = st.text_input(
        "Raw source folder path",
        value=str(RAW_DIR),
        help="Example: /home/homeserver/projects/image-recognition-lab/data/raw/famous_brand_logos/logos/NIKE"
    )

    split_class_name = st.text_input(
        "Class name for split",
        placeholder="Example: toyota"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        train_pct = st.number_input(
            "Training %",
            min_value=0,
            max_value=100,
            value=70
        )

    with col2:
        val_pct = st.number_input(
            "Validation %",
            min_value=0,
            max_value=100,
            value=15
        )

    with col3:
        test_pct = st.number_input(
            "Testing %",
            min_value=0,
            max_value=100,
            value=15
        )

    copy_files = st.checkbox(
        "Copy files instead of move",
        value=True,
        help="Recommended. Keeps your raw dataset untouched."
    )

    if st.button("Preview Split"):
        source_path = Path(raw_source_dir)

        if not source_path.exists():
            st.error("Source folder does not exist.")
        elif not split_class_name.strip():
            st.error("Enter a class name.")
        elif train_pct + val_pct + test_pct != 100:
            st.error("Training + Validation + Testing must equal 100.")
        else:
            files = [
                p for p in source_path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]

            random.shuffle(files)

            total = len(files)
            train_count = int(total * train_pct / 100)
            val_count = int(total * val_pct / 100)
            test_count = total - train_count - val_count

            st.success("Preview ready.")

            st.write({
                "Total Images": total,
                "Training": train_count,
                "Validation": val_count,
                "Testing": test_count,
            })

    if st.button("Run Split"):
        source_path = Path(raw_source_dir)

        if not source_path.exists():
            st.error("Source folder does not exist.")
        elif not split_class_name.strip():
            st.error("Enter a class name.")
        elif train_pct + val_pct + test_pct != 100:
            st.error("Training + Validation + Testing must equal 100.")
        else:
            clean_name = clean_class_name(split_class_name)

            train_dest = TRAINING_DIR / clean_name
            val_dest = VALIDATION_DIR / clean_name
            test_dest = TESTING_DIR / clean_name

            train_dest.mkdir(parents=True, exist_ok=True)
            val_dest.mkdir(parents=True, exist_ok=True)
            test_dest.mkdir(parents=True, exist_ok=True)

            files = [
                p for p in source_path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]

            random.shuffle(files)

            total = len(files)
            train_count = int(total * train_pct / 100)
            val_count = int(total * val_pct / 100)

            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]

            def transfer_files(file_list, destination_dir):
                moved = 0
                skipped = 0

                for file_path in file_list:
                    destination = destination_dir / file_path.name

                    if destination.exists():
                        skipped += 1
                        continue

                    if copy_files:
                        shutil.copy2(file_path, destination)
                    else:
                        shutil.move(str(file_path), str(destination))

                    moved += 1

                return moved, skipped

            train_moved, train_skipped = transfer_files(train_files, train_dest)
            val_moved, val_skipped = transfer_files(val_files, val_dest)
            test_moved, test_skipped = transfer_files(test_files, test_dest)

            st.success("Dataset split complete.")

            st.write({
                "Class": clean_name,
                "Training Added": train_moved,
                "Training Skipped": train_skipped,
                "Validation Added": val_moved,
                "Validation Skipped": val_skipped,
                "Testing Added": test_moved,
                "Testing Skipped": test_skipped,
            })

            st.rerun()
