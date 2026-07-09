import streamlit as st

from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.pages.upload import render_upload_tab
from app.pages.dataset_manager import render_dataset_manager_tab
from app.pages.raw_dataset_scanner import render_raw_dataset_scanner_tab
from app.pages.predict import render_predict_tab
from app.pages.train import render_train_tab
from app.pages.evaluate import render_evaluate_tab
from app.pages.model_manager import render_model_manager_tab
from app.pages.dataset_health import render_dataset_health_tab
from app.pages.correction_review import render_correction_review_tab
from app.pages.transfer import render_transfer_tab

from app.utils.paths import (
    UPLOAD_DIR,
    TRAINING_DIR,
    TESTING_DIR,
    VALIDATION_DIR,
)

from app.utils.image_utils import (
    count_images,
)

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Upload Images",
    "Dataset Manager",
    "Raw Dataset Scanner",
    "Train Model",
    "Predict",
    "Evaluate Model",
    "Model Manager",
    "Dataset Health",
    "Correction Review",
    "Export / Import",
])


with tab1:
    render_upload_tab()

with tab2:
    render_dataset_manager_tab()

with tab3:
    render_raw_dataset_scanner_tab()


# -----------------------------
# Tab 4: Train Model
# -----------------------------

with tab4:
    render_train_tab()

# -----------------------------
# Tab 5: Predict
# -----------------------------

with tab5:
    render_predict_tab()

# -----------------------------
# Tab 6: Evaluate Model
# -----------------------------

with tab6:
    render_evaluate_tab()

with tab7:
    render_model_manager_tab()

with tab8:
    render_dataset_health_tab()

with tab9:
    render_correction_review_tab()

with tab10:
    render_transfer_tab()
