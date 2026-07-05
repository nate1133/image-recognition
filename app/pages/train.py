import streamlit as st
import pandas as pd

from app.utils.paths import TRAINING_DIR, TESTING_DIR, VALIDATION_DIR, MODEL_DIR
from app.utils.image_utils import count_images, get_all_classes
from src.models.train_model import train_image_classifier


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
    st.dataframe(dataset_df, use_container_width=True)

    total_training = dataset_df["Training Images"].sum()
    total_validation = dataset_df["Validation Images"].sum()

    if total_training == 0:
        st.error("You need training images before training.")
        return

    if total_validation == 0:
        st.warning("You need validation images before serious training.")
        return

    st.subheader("Training Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        image_size = st.selectbox("Image Size", [128, 160, 224], index=1)

    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    with col3:
        epochs = st.number_input("Epochs", 1, 50, 5, 1)

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
                st.dataframe(history_df, use_container_width=True)

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