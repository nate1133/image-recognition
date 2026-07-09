from datetime import datetime

import streamlit as st

from app.utils.paths import MODEL_DIR, TESTING_DIR, TRAINING_DIR, VALIDATION_DIR
from src.data.bundle_transfer import create_portable_bundle, import_portable_bundle


def _split_dirs():
    return {
        "training": TRAINING_DIR,
        "validation": VALIDATION_DIR,
        "testing": TESTING_DIR,
    }


def render_transfer_tab():
    st.header("Export / Import")
    st.write("Share the current model and dataset splits as one portable ZIP bundle.")

    st.subheader("Export Bundle")
    include_model = st.checkbox("Include current model", value=True)
    include_dataset = st.checkbox("Include dataset splits", value=True)
    if include_model or include_dataset:
        bundle = create_portable_bundle(
            MODEL_DIR, _split_dirs(), include_model, include_dataset
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Download Bundle",
            bundle,
            file_name=f"image_recognition_lab_{stamp}.zip",
            mime="application/zip",
        )

    st.divider()
    st.subheader("Import Bundle")
    uploaded = st.file_uploader("Choose a bundle ZIP", type=["zip"])
    if uploaded:
        import_model = st.checkbox("Import model", value=True)
        import_dataset = st.checkbox("Import dataset splits", value=True)
        overwrite = st.checkbox("Overwrite files with matching names", value=False)
        if st.button("Import Bundle"):
            try:
                imported = import_portable_bundle(
                    uploaded.getvalue(), MODEL_DIR, _split_dirs(),
                    import_model, import_dataset, overwrite,
                )
                st.success(f"Imported {len(imported)} files.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
