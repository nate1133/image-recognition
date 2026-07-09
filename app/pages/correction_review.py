import streamlit as st

from app.utils.paths import CORRECTION_REVIEW_DIR, TRAINING_DIR
from src.data.correction_queue import list_corrections, review_correction


def render_correction_review_tab():
    st.header("Correction Review")
    st.write("Approve corrected predictions before they become training data.")
    items = list_corrections(CORRECTION_REVIEW_DIR)
    if not items:
        st.info("No prediction corrections are waiting for review.")
        return

    st.metric("Pending Corrections", len(items))
    for index, item in enumerate(items):
        with st.container(border=True):
            left, right = st.columns([1, 2])
            with left:
                st.image(str(item["path"]), width=240)
            with right:
                st.write(f"**Proposed class:** {item['class_name']}")
                if item["prediction"]:
                    st.write(f"Original prediction: {item['prediction']}")
                st.caption(item["original_filename"])
                approve, reject = st.columns(2)
                if approve.button("Approve", key=f"approve_{index}"):
                    destination = review_correction(
                        item["path"], CORRECTION_REVIEW_DIR, TRAINING_DIR, True
                    )
                    st.success(f"Added to training/{item['class_name']}/{destination.name}")
                    st.rerun()
                if reject.button("Reject", key=f"reject_{index}"):
                    review_correction(
                        item["path"], CORRECTION_REVIEW_DIR, TRAINING_DIR, False
                    )
                    st.success("Correction rejected.")
                    st.rerun()
