import streamlit as st

from app.utils.paths import UPLOAD_DIR
from app.utils.image_utils import clean_file_name, get_uploaded_images, preview_image


def render_upload_tab():
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
            save_path = UPLOAD_DIR / clean_file_name(file.name)

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

    st.divider()
    st.subheader("Upload Cleanup Tools")

    uploaded_images = get_uploaded_images()

    if not uploaded_images:
        st.info("No uploaded images to clean up.")
        return

    image_names = [p.name for p in uploaded_images]

    selected_to_delete = st.multiselect(
        "Select uploaded images to delete",
        image_names,
        key="delete_uploaded_images_select"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Delete Selected Uploads", key="delete_selected_uploads"):
            if not selected_to_delete:
                st.warning("Select at least one uploaded image first.")
            else:
                deleted = 0

                for image_name in selected_to_delete:
                    image_path = UPLOAD_DIR / image_name

                    if image_path.exists():
                        image_path.unlink()
                        deleted += 1

                st.success(f"Deleted {deleted} uploaded image(s).")
                st.rerun()

    with col2:
        confirm_delete_all = st.checkbox(
            "Confirm delete all uploaded images",
            key="confirm_delete_all_uploads"
        )

        if st.button("Delete All Uploads", key="delete_all_uploads"):
            if not confirm_delete_all:
                st.warning("Check the confirmation box first.")
            else:
                deleted = 0

                for image_path in uploaded_images:
                    if image_path.exists():
                        image_path.unlink()
                        deleted += 1

                st.success(f"Deleted {deleted} uploaded image(s).")
                st.rerun()
