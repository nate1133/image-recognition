import shutil

import pandas as pd
import streamlit as st

from app.utils.paths import MODEL_DIR


def prepare_model_comparison_df(runs_df):
    chart_columns = [
        "training_accuracy",
        "validation_accuracy",
        "training_loss",
        "validation_loss",
    ]

    available_columns = [
        col for col in chart_columns
        if col in runs_df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    comparison_df = runs_df.copy()

    for col in available_columns:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")

    comparison_df = comparison_df.dropna(
        subset=available_columns,
        how="all",
    )

    if comparison_df.empty:
        return comparison_df

    if "run_time" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("run_time")
        comparison_df["Run"] = comparison_df["run_time"].astype(str)
    else:
        comparison_df["Run"] = [
            f"Run {index + 1}" for index in range(len(comparison_df))
        ]

    if "model_file" in comparison_df.columns:
        comparison_df["Run"] = comparison_df.apply(
            lambda row: f"{row['Run']} - {row['model_file']}",
            axis=1,
        )

    return comparison_df.set_index("Run")


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

    st.dataframe(display_df, width="stretch")

    note_columns = [
        col for col in ["dataset_changes", "observations"]
        if col in runs_df.columns
    ]
    if note_columns:
        noted_runs = runs_df[
            runs_df[note_columns].fillna("").astype(str).apply(
                lambda row: any(value.strip() for value in row),
                axis=1,
            )
        ]
        if not noted_runs.empty:
            st.subheader("Experiment Notes")
            for _, run in noted_runs.iloc[::-1].iterrows():
                with st.expander(str(run.get("model_file", run.get("run_time", "Run")))):
                    dataset_changes = str(run.get("dataset_changes", "") or "").strip()
                    observations = str(run.get("observations", "") or "").strip()
                    if dataset_changes and dataset_changes.lower() != "nan":
                        st.markdown("**Dataset changes**")
                        st.write(dataset_changes)
                    if observations and observations.lower() != "nan":
                        st.markdown("**Manual observations**")
                        st.write(observations)

    comparison_df = prepare_model_comparison_df(runs_df)

    if not comparison_df.empty:
        st.divider()
        st.subheader("Model Comparison")

        accuracy_cols = [
            col for col in ["training_accuracy", "validation_accuracy"]
            if col in comparison_df.columns
        ]
        loss_cols = [
            col for col in ["training_loss", "validation_loss"]
            if col in comparison_df.columns
        ]

        if accuracy_cols:
            accuracy_chart_df = comparison_df[accuracy_cols] * 100
            accuracy_chart_df = accuracy_chart_df.rename(columns={
                "training_accuracy": "Training Accuracy %",
                "validation_accuracy": "Validation Accuracy %",
            })

            st.caption("Accuracy across training runs")
            st.line_chart(accuracy_chart_df)

        if loss_cols:
            loss_chart_df = comparison_df[loss_cols].rename(columns={
                "training_loss": "Training Loss",
                "validation_loss": "Validation Loss",
            })

            st.caption("Loss across training runs")
            st.line_chart(loss_chart_df)

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
