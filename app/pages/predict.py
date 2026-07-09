import streamlit as st
import pandas as pd

from app.utils.paths import TRAINING_DIR, MODEL_DIR
from src.models.predict_model import (
    load_model_metadata,
    predict_image,
    save_prediction_correction,
)


def render_predict_tab():
    st.header("Predict")

    try:
        model_path, class_names, image_size, model_info = load_model_metadata(MODEL_DIR)
    except FileNotFoundError as e:
        st.warning(str(e))
        return

    if model_info:
        st.subheader("Current Model Info")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Architecture", model_info.get("architecture", "Unknown"))

        with col2:
            st.metric("Image Size", image_size)

        with col3:
            st.metric("Classes", len(class_names))

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

    if not prediction_file:
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_prediction_filename", None)
        return

    if st.button("Run Prediction", key="run_prediction_button"):
        with st.spinner("Running prediction..."):
            try:
                prediction = predict_image(MODEL_DIR, prediction_file)
                st.session_state.last_prediction = prediction
                st.session_state.last_prediction_filename = prediction_file.name
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)

    prediction = st.session_state.get("last_prediction")
    prediction_filename = st.session_state.get("last_prediction_filename")

    if not prediction or prediction_filename != prediction_file.name:
        return

    img = prediction["image"]
    class_names = prediction["class_names"]

    st.image(
        img,
        caption="Image for prediction",
        width=400
    )

    st.success(f"Prediction: {prediction['top_class']}")

    st.metric(
        "Confidence",
        f"{prediction['top_confidence'] * 100:.2f}%"
    )

    results_df = pd.DataFrame(prediction["results"])

    st.subheader("Top Predictions")

    st.dataframe(
        results_df[["Class", "Confidence %"]],
        width="stretch"
    )

    chart_df = results_df.set_index("Class")[["Confidence %"]]
    st.bar_chart(chart_df)

    st.divider()
    st.subheader("Correct Prediction")

    correct_class = st.selectbox(
        "If the prediction is wrong, choose the correct class",
        class_names,
        key="correct_prediction_class_tab5"
    )

    if st.button(
        "Save Image to Correct Training Class",
        key="save_correction_button"
    ):
        save_prediction_correction(
            training_dir=TRAINING_DIR,
            class_name=correct_class,
            image=img,
            original_filename=prediction_file.name,
        )
        st.success(f"Saved image to training/{correct_class}")
