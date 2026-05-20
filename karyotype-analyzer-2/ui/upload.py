"""Image upload section."""

import streamlit as st
from PIL import Image
from typing import Optional


def display_upload_section() -> Optional[Image.Image]:
    """Render the image upload section and return the uploaded image."""
    st.header("📤 Upload Metaphase Spread Image")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            help="Upload a high-quality metaphase spread image (max 10MB)",
        )
        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit.")
                return None
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Image", width="stretch")
            st.info(
                f"**Image Details:**\n"
                f"- Format: {image.format or 'N/A'}\n"
                f"- Size: {image.size[0]} x {image.size[1]} pixels\n"
                f"- Mode: {image.mode}\n"
                f"- File size: {uploaded_file.size / 1024:.1f} KB"
            )
            return image

    with col2:
        st.markdown("""
        ### Guidelines:
        - Use high-resolution images
        - Ensure clear chromosome spread
        - Good contrast and lighting

        ### Supported Formats:
        - PNG, JPG, JPEG, TIFF, BMP
        - Max size: 10MB
        """)
    return None
