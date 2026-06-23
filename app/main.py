import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.scan_dataset import scan_dataset
from src.data.prepare_dataset import import_selected_classes

# -----------------------------
# Project Paths
# -----------------------------

BASE_DIR = Path("/home/homeserver/projects/image-recognition-lab")
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"
TRAINING_DIR = DATA_DIR / "training"
TESTING_DIR = DATA_DIR / "testing"
VALIDATION_DIR = DATA_DIR / "validation"

for folder in [UPLOAD_DIR, TRAINING_DIR, TESTING_DIR, VALIDATION_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


# -----------------------------
# Helper Functions
# -----------------------------

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
            use_container_width=False if width else True
        )
    except Exception:
        st.warning(f"Could not preview {image_path.name}")


# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(
    page_title="Image Recognition Lab",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Image Recognition Lab")

user_name = st.sidebar.text_input("User name")

if user_name:
    st.sidebar.success(f"Active user: {user_name}")

st.sidebar.divider()
st.sidebar.subheader("Dataset Counts")

st.sidebar.metric("Uploaded Images", count_images(UPLOAD_DIR))
st.sidebar.metric("Training Images", count_images(TRAINING_DIR))
st.sidebar.metric("Testing Images", count_images(TESTING_DIR))
st.sidebar.metric("Validation Images", count_images(VALIDATION_DIR))

st.sidebar.divider()
st.sidebar.caption("Running on port 8502")


# -----------------------------
# Main App
# -----------------------------

st.title("ML Image Recognition App")
st.write(
    "Upload images, organize them into class buckets, train a model, "
    "and test predictions."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload Images",
    "Dataset Manager",
    "Raw Dataset Scanner",
    "Train Model",
    "Predict"
])


# -----------------------------
# Tab 1: Upload Images
# -----------------------------

with tab1:
    st.header("Upload Images")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        saved_count = 0
        skipped_count = 0

        for file in uploaded_files:
            save_path = UPLOAD_DIR / file.name

            if save_path.exists():
                skipped_count += 1
                continue

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            saved_count += 1

        if saved_count:
            st.success(f"Uploaded {saved_count} image(s).")

        if skipped_count:
            st.warning(f"Skipped {skipped_count} duplicate image(s).")

    uploaded_images = get_uploaded_images()

    if uploaded_images:
        st.subheader("Preview Uploaded Images")

        cols = st.columns(4)

        for i, img_path in enumerate(uploaded_images[:40]):
            with cols[i % 4]:
                preview_image(img_path)
    else:
        st.info("No uploaded images yet.")


# -----------------------------
# Tab 2: Dataset Manager
# -----------------------------

with tab2:
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


# -----------------------------
# Tab 3: Raw Dataset Scanner / Import Wizard
# -----------------------------

with tab3:
    st.header("Raw Dataset Scanner / Import Wizard")

    RAW_DIR = DATA_DIR / "raw"

    st.write(f"Scanning: `{RAW_DIR}`")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None

    if st.button("Scan Raw Datasets"):
        with st.spinner("Scanning datasets..."):
            results = scan_dataset(RAW_DIR)

        st.session_state.scan_results = results

    results = st.session_state.scan_results

    if results is None:
        st.info("Click Scan Raw Datasets to inspect your raw image folders.")
    elif len(results) == 0:
        st.warning("No image folders found.")
    else:
        df = pd.DataFrame(results)

        st.success(
            f"Found {len(df)} image folders with "
            f"{df['images'].sum():,} total images."
        )

        st.dataframe(df, use_container_width=True)

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
        else:
            import_button = st.button("Import Selected Classes")

            if import_button:
                if not selected_classes:
                    st.error("Select at least one class.")
                else:
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
                    st.dataframe(summary_df, use_container_width=True)

                    st.rerun()


# -----------------------------
# Tab 4: Train Model
# -----------------------------

with tab4:
    st.header("Train Model")

    st.write("Train a logo/image classifier using the images in your training and validation folders.")

    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_classes = get_all_classes()

    if not all_classes:
        st.warning("Create classes and add training images before training.")
    else:
        rows = []

        for class_name in all_classes:
            rows.append({
                "Class": class_name,
                "Training Images": count_images(TRAINING_DIR / class_name),
                "Testing Images": count_images(TESTING_DIR / class_name),
                "Validation Images": count_images(VALIDATION_DIR / class_name),
            })

        dataset_df = pd.DataFrame(rows)
        st.dataframe(dataset_df, use_container_width=True)

        total_training = dataset_df["Training Images"].sum()
        total_validation = dataset_df["Validation Images"].sum()

        if total_training == 0:
            st.error("You need training images before you can train a model.")
        elif total_validation == 0:
            st.warning("You do not have validation images yet. Add some before serious training.")
        else:
            st.subheader("Training Settings")

            col1, col2, col3 = st.columns(3)

            with col1:
                image_size = st.selectbox(
                    "Image Size",
                    [128, 160, 224],
                    index=1
                )

            with col2:
                batch_size = st.selectbox(
                    "Batch Size",
                    [16, 32, 64],
                    index=1
                )

            with col3:
                epochs = st.number_input(
                    "Epochs",
                    min_value=1,
                    max_value=50,
                    value=5,
                    step=1
                )

            train_button = st.button("Train Model")

            if train_button:
                with st.spinner("Training model... this may take a while."):
                    try:
                        import tensorflow as tf
                        from tensorflow.keras import layers, models
                        from tensorflow.keras.applications import MobileNetV2
                        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                        import json

                        train_ds = tf.keras.utils.image_dataset_from_directory(
                            TRAINING_DIR,
                            image_size=(image_size, image_size),
                            batch_size=batch_size,
                            label_mode="categorical"
                        )

                        val_ds = tf.keras.utils.image_dataset_from_directory(
                            VALIDATION_DIR,
                            image_size=(image_size, image_size),
                            batch_size=batch_size,
                            label_mode="categorical"
                        )

                        class_names = train_ds.class_names
                        num_classes = len(class_names)

                        AUTOTUNE = tf.data.AUTOTUNE

                        train_ds = train_ds.map(
                            lambda x, y: (preprocess_input(x), y)
                        ).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

                        val_ds = val_ds.map(
                            lambda x, y: (preprocess_input(x), y)
                        ).cache().prefetch(buffer_size=AUTOTUNE)

                        base_model = MobileNetV2(
                            input_shape=(image_size, image_size, 3),
                            include_top=False,
                            weights="imagenet"
                        )

                        base_model.trainable = False

                        model = models.Sequential([
                            base_model,
                            layers.GlobalAveragePooling2D(),
                            layers.Dropout(0.25),
                            layers.Dense(num_classes, activation="softmax")
                        ])

                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss="categorical_crossentropy",
                            metrics=["accuracy"]
                        )

                        history = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=epochs
                        )
                        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                        model_filename = f"logo_classifier_{run_time}.keras"
                        classes_filename = f"classes_{run_time}.json"
                        info_filename = f"model_info_{run_time}.json"

                        model_path = MODEL_DIR / model_filename
                        classes_path = MODEL_DIR / classes_filename
                        info_path = MODEL_DIR / info_filename

                        latest_model_path = MODEL_DIR / "logo_classifier.keras"
                        latest_classes_path = MODEL_DIR / "classes.json"
                        latest_info_path = MODEL_DIR / "model_info.json"

                        training_log_path = MODEL_DIR / "training_runs.csv"

                        model.save(model_path)
                        model.save(latest_model_path)

                        with open(classes_path, "w") as f:
                            json.dump(class_names, f, indent=4)

                        with open(latest_classes_path, "w") as f:
                            json.dump(class_names, f, indent=4)

                        model_info = {
                            "run_time": run_time,
                            "model_name": model_filename,
                            "architecture": "MobileNetV2",
                            "image_size": image_size,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "classes": class_names,
                            "num_classes": len(class_names),
                            "training_images": int(total_training),
                            "validation_images": int(total_validation),
                            "final_training_accuracy": float(history.history["accuracy"][-1]),
                            "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
                            "final_training_loss": float(history.history["loss"][-1]),
                            "final_validation_loss": float(history.history["val_loss"][-1]),
                        }

                        with open(info_path, "w") as f:
                            json.dump(model_info, f, indent=4)

                        with open(latest_info_path, "w") as f:
                            json.dump(model_info, f, indent=4)

                        log_row = pd.DataFrame([{
                            "run_time": run_time,
                            "model_file": model_filename,
                            "architecture": "MobileNetV2",
                            "image_size": image_size,
                            "batch_size": batch_size,4
                            "epochs": epochs,
                            "num_classes": len(class_names),
                            "training_images": int(total_training),
                            "validation_images": int(total_validation),
                            "training_accuracy": float(history.history["accuracy"][-1]),
                            "validation_accuracy": float(history.history["val_accuracy"][-1]),
                            "training_loss": float(history.history["loss"][-1]),
                            "validation_loss": float(history.history["val_loss"][-1]),
                        }])

                        if training_log_path.exists():
                            old_log = pd.read_csv(training_log_path)
                            updated_log = pd.concat([old_log, log_row], ignore_index=True)
                        else:
                            updated_log = log_row

                        updated_log.to_csv(training_log_path, index=False)

                        st.success("Model trained and saved successfully.")

                        st.metric(
                            "Final Training Accuracy",
                            f"{history.history['accuracy'][-1] * 100:.2f}%"
                        )

                        st.metric(
                            "Final Validation Accuracy",
                            f"{history.history['val_accuracy'][-1] * 100:.2f}%"
                        )

                        history_df = pd.DataFrame({
                            "Epoch": list(range(1, epochs + 1)),
                            "Training Accuracy": history.history["accuracy"],
                            "Validation Accuracy": history.history["val_accuracy"],
                            "Training Loss": history.history["loss"],
                            "Validation Loss": history.history["val_loss"],
                        })

                        st.subheader("Training History")
                        st.dataframe(history_df, use_container_width=True)

                        st.line_chart(
                            history_df.set_index("Epoch")[[
                                "Training Accuracy",
                                "Validation Accuracy"
                            ]]
                        )

                        st.line_chart(
                            history_df.set_index("Epoch")[[
                                "Training Loss",
                                "Validation Loss"
                            ]]
                        )

                    except Exception as e:
                        st.error("Training failed.")
                        st.exception(e)


# -----------------------------
# Tab 5: Predict
# -----------------------------

with tab5:
    st.header("Predict")

    MODEL_DIR = BASE_DIR / "models"
    model_path = MODEL_DIR / "logo_classifier.keras"
    classes_path = MODEL_DIR / "classes.json"
    info_path = MODEL_DIR / "model_info.json"

    if not model_path.exists():
        st.warning("No trained model found yet. Train a model first.")

    elif not classes_path.exists():
        st.warning("Model exists, but classes.json is missing.")

    else:
        import json
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        with open(classes_path, "r") as f:
            class_names = json.load(f)

        image_size = 160

        if info_path.exists():
            with open(info_path, "r") as f:
                model_info = json.load(f)

            image_size = model_info.get("image_size", 160)

            st.subheader("Current Model Info")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Architecture",
                    model_info.get("architecture", "Unknown")
                )

            with col2:
                st.metric(
                    "Image Size",
                    image_size
                )

            with col3:
                st.metric(
                    "Classes",
                    len(class_names)
                )

            if "final_validation_accuracy" in model_info:
                st.metric(
                    "Validation Accuracy",
                    f"{model_info['final_validation_accuracy'] * 100:.2f}%"
                )

        prediction_file = st.file_uploader(
            "Upload an image for prediction",
            type=["jpg", "jpeg", "png", "webp"],
            key="prediction_upload_tab5"
        )

        if prediction_file:
            img = Image.open(prediction_file).convert("RGB")

            st.image(
                img,
                caption="Image for prediction",
                width=400
            )

            run_prediction = st.button(
                "Run Prediction",
                key="run_prediction_button"
            )

            if run_prediction:
                with st.spinner("Running prediction..."):
                    try:
                        model = tf.keras.models.load_model(model_path)

                        img_resized = img.resize(
                            (image_size, image_size)
                        )

                        img_array = np.array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = preprocess_input(img_array)

                        predictions = model.predict(img_array)[0]

                        top_index = int(np.argmax(predictions))
                        top_class = class_names[top_index]
                        top_confidence = float(predictions[top_index])

                        st.success(f"Prediction: {top_class}")

                        st.metric(
                            "Confidence",
                            f"{top_confidence * 100:.2f}%"
                        )

                        top_n = min(5, len(class_names))
                        top_indices = predictions.argsort()[-top_n:][::-1]

                        results = []

                        for idx in top_indices:
                            results.append({
                                "Class": class_names[int(idx)],
                                "Confidence": float(predictions[int(idx)])
                            })

                        results_df = pd.DataFrame(results)
                        results_df["Confidence %"] = (
                            results_df["Confidence"] * 100
                        )

                        st.subheader("Top Predictions")

                        st.dataframe(
                            results_df[["Class", "Confidence %"]],
                            use_container_width=True
                        )

                        chart_df = results_df.set_index("Class")[["Confidence %"]]
                        st.bar_chart(chart_df)

                        # -----------------------------
                        # Prediction Correction
                        # -----------------------------

                        st.divider()
                        st.subheader("Correct Prediction")

                        correct_class = st.selectbox(
                            "If the prediction is wrong, choose the correct class",
                            class_names,
                            key="correct_prediction_class_tab5"
                        )

                        save_correction = st.button(
                            "Save Image to Correct Training Class",
                            key="save_correction_button"
                        )

                        if save_correction:
                            correction_dir = TRAINING_DIR / correct_class
                            correction_dir.mkdir(
                                parents=True,
                                exist_ok=True
                            )

                            save_path = correction_dir / prediction_file.name

                            if save_path.exists():
                                st.warning(
                                    f"{prediction_file.name} already exists in "
                                    f"training/{correct_class}"
                                )
                            else:
                                img.convert("RGB").save(save_path)

                                st.success(
                                    f"Saved image to training/{correct_class}"
                                )

                    except Exception as e:
                        st.error("Prediction failed.")
                        st.exception(e)