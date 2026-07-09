import streamlit as st

from app.utils.paths import TESTING_DIR, MODEL_DIR
from app.utils.image_utils import count_images
from src.models.evaluate_model import evaluate_image_classifier


def render_evaluate_tab():
    st.header("Evaluate Model")

    model_path = MODEL_DIR / "logo_classifier.keras"
    classes_path = MODEL_DIR / "classes.json"

    if not model_path.exists():
        st.warning("No trained model found. Train a model first.")
        return

    if not classes_path.exists():
        st.warning("classes.json is missing. Train the model again.")
        return

    if count_images(TESTING_DIR) == 0:
        st.warning("No testing images found. Add images to the testing folders first.")
        return

    st.subheader("Evaluation Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Testing Images", count_images(TESTING_DIR))

    with col2:
        st.metric("Current Model", "logo_classifier.keras")

    if st.button("Run Evaluation", key="run_evaluation_button"):
        with st.spinner("Evaluating model on testing images..."):
            try:
                results = evaluate_image_classifier(
                    model_dir=MODEL_DIR,
                    testing_dir=TESTING_DIR,
                )

                st.success("Evaluation complete.")

                st.metric(
                    "Overall Test Accuracy",
                    f"{results['accuracy'] * 100:.2f}%"
                )

                if results["model_classes"] != results["testing_classes"]:
                    st.warning("Testing folder classes do not perfectly match model classes.")
                    st.write("Model classes:", results["model_classes"])
                    st.write("Testing folder classes:", results["testing_classes"])

                st.subheader("Confusion Matrix")
                st.dataframe(
                    results["confusion_matrix"],
                    width="stretch",
                )

                st.subheader("Per-Class Performance")
                st.dataframe(
                    results["classification_report"],
                    width="stretch",
                )

                st.subheader("Class Accuracy")
                st.dataframe(
                    results["class_accuracy"],
                    width="stretch",
                )

                st.bar_chart(
                    results["class_accuracy"].set_index("Class")[["Accuracy %"]]
                )

            except Exception as e:
                st.error("Evaluation failed.")
                st.exception(e)