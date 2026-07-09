import streamlit as st
import pandas as pd

from app.utils.paths import TRAINING_DIR, TESTING_DIR, VALIDATION_DIR, MODEL_DIR
from app.utils.image_utils import count_images, get_all_classes
from src.models.train_model import train_image_classifier
from src.models.backbones import SUPPORTED_BACKBONES


def get_training_balance_warnings(dataset_df, min_images_per_class=10, max_ratio=3.0):
    warnings = []

    if dataset_df.empty or "Training Images" not in dataset_df.columns:
        return warnings

    training_counts = dataset_df[["Class", "Training Images"]].copy()
    training_counts["Training Images"] = pd.to_numeric(
        training_counts["Training Images"],
        errors="coerce",
    ).fillna(0)

    empty_classes = training_counts[
        training_counts["Training Images"] == 0
    ]["Class"].tolist()

    if empty_classes:
        warnings.append(
            "Classes with no training images: "
            + ", ".join(empty_classes)
        )

    low_sample_classes = training_counts[
        (training_counts["Training Images"] > 0)
        & (training_counts["Training Images"] < min_images_per_class)
    ]

    if not low_sample_classes.empty:
        warnings.append(
            f"Classes with fewer than {min_images_per_class} training images: "
            + ", ".join(
                f"{row['Class']} ({int(row['Training Images'])})"
                for _, row in low_sample_classes.iterrows()
            )
        )

    nonzero_counts = training_counts[
        training_counts["Training Images"] > 0
    ]["Training Images"]

    if len(nonzero_counts) >= 2:
        min_count = int(nonzero_counts.min())
        max_count = int(nonzero_counts.max())

        if min_count and max_count / min_count >= max_ratio:
            warnings.append(
                "Training set is imbalanced: largest class has "
                f"{max_count} images and smallest non-empty class has "
                f"{min_count} images."
            )

    return warnings


def render_train_tab():
    st.header("Train Model")

    all_classes = get_all_classes()

    if not all_classes:
        st.warning("Create classes and add training images before training.")
        return

    rows = []

    for class_name in all_classes:
        rows.append({
            "Class": class_name,
            "Training Images": count_images(TRAINING_DIR / class_name),
            "Testing Images": count_images(TESTING_DIR / class_name),
            "Validation Images": count_images(VALIDATION_DIR / class_name),
        })

    dataset_df = pd.DataFrame(rows)
    st.dataframe(dataset_df, width="stretch")

    training_counts = dataset_df.set_index("Class")[["Training Images"]]
    st.subheader("Training Class Balance")
    st.bar_chart(training_counts)

    balance_warnings = get_training_balance_warnings(dataset_df)

    if balance_warnings:
        for warning in balance_warnings:
            st.warning(warning)
    else:
        st.success("Training classes look reasonably balanced.")

    total_training = dataset_df["Training Images"].sum()
    total_validation = dataset_df["Validation Images"].sum()

    if total_training == 0:
        st.error("You need training images before training.")
        return

    if total_validation == 0:
        st.warning("You need validation images before serious training.")
        return

    st.subheader("Training Settings")

    backbone = st.selectbox(
        "Model Backbone",
        SUPPORTED_BACKBONES,
        help="MobileNetV2 is fastest; EfficientNetB0 and ResNet50 offer alternatives.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        image_size = st.selectbox("Image Size", [128, 160, 224], index=1)

    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    with col3:
        epochs = st.number_input("Epochs", 1, 50, 5, 1)

    st.subheader("Experiment Notes")
    dataset_changes = st.text_area(
        "Dataset changes",
        placeholder="Example: Added 40 Toyota images and removed duplicate logos.",
        help="Record what changed in the dataset since the previous run.",
    )
    observations = st.text_area(
        "Manual observations",
        placeholder="Example: More varied backgrounds; watch confusion between Audi and VW.",
        help="Record hypotheses, caveats, or anything worth remembering.",
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                model_info, history_df = train_image_classifier(
                    training_dir=TRAINING_DIR,
                    validation_dir=VALIDATION_DIR,
                    model_dir=MODEL_DIR,
                    image_size=image_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    backbone=backbone,
                    dataset_changes=dataset_changes,
                    observations=observations,
                )

                st.success("Model trained and saved successfully.")

                st.metric(
                    "Final Training Accuracy",
                    f"{model_info['final_training_accuracy'] * 100:.2f}%"
                )

                st.metric(
                    "Final Validation Accuracy",
                    f"{model_info['final_validation_accuracy'] * 100:.2f}%"
                )

                st.subheader("Training History")
                st.dataframe(history_df, width="stretch")

                st.line_chart(
                    history_df.set_index("Epoch")[[
                        "Training Accuracy",
                        "Validation Accuracy",
                    ]]
                )

                st.line_chart(
                    history_df.set_index("Epoch")[[
                        "Training Loss",
                        "Validation Loss",
                    ]]
                )

            except Exception as e:
                st.error("Training failed.")
                st.exception(e)
