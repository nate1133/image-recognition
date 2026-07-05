import shutil

import pandas as pd
import streamlit as st

from app.utils.paths import MODEL_DIR


def render_model_manager_tab():
    st.header("Model Manager")

    training_log_path = MODEL_DIR / "training_runs.csv"

    if not training_log_path.exists():
        st.info("No training run log found yet. Train a model first.")
        return

    runs_df = pd.read_csv(training_log_path)

    if runs_df.empty:
        st.info("Training log is empty.")
        return

    st.subheader("Training Runs")

    display_df = runs_df.copy()

    for col in [
        "training_accuracy",
        "validation_accuracy",
        "training_loss",
        "validation_loss"
    ]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    st.dataframe(display_df, use_container_width=True)

    st.divider()
    st.subheader("Current Best Model")

    if "validation_accuracy" in runs_df.columns:
        scored_runs = runs_df.dropna(subset=["validation_accuracy"])

        if scored_runs.empty:
            st.info("No validation scores found in the training log.")
        else:
            best_row = scored_runs.loc[scored_runs["validation_accuracy"].idxmax()]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Best Model", best_row["model_file"])

            with col2:
                st.metric(
                    "Validation Accuracy",
                    f"{best_row['validation_accuracy'] * 100:.2f}%"
                )

            with col3:
                st.metric("Epochs", int(best_row["epochs"]))

    st.divider()
    st.subheader("Set Model as Current")

    model_files = runs_df["model_file"].dropna().unique().tolist()

    if not model_files:
        st.info("No model files found in the training log.")
        return

    selected_model = st.selectbox(
        "Choose model version",
        model_files,
        key="selected_model_version"
    )

    if st.button("Set Selected Model as Current"):
        selected_model_path = MODEL_DIR / selected_model

        if not selected_model_path.exists():
            st.error("Selected model file does not exist.")
            return

        shutil.copy2(
            selected_model_path,
            MODEL_DIR / "logo_classifier.keras"
        )

        selected_run = runs_df[runs_df["model_file"] == selected_model].iloc[0]

        run_time = selected_run["run_time"]

        info_file = MODEL_DIR / f"model_info_{run_time}.json"
        classes_file = MODEL_DIR / f"classes_{run_time}.json"

        if info_file.exists():
            shutil.copy2(info_file, MODEL_DIR / "model_info.json")

        if classes_file.exists():
            shutil.copy2(classes_file, MODEL_DIR / "classes.json")

        st.success(f"Set {selected_model} as the current model.")
