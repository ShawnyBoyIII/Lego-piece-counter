import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from detector import detect_lego_pieces
import io
import json

st.set_page_config(page_title="Lego Scanner Prototype", layout="wide")

st.title("🧱 Lego Piece Scanner Prototype")
st.write("""
Upload an image containing Lego pieces. The scanner will identify objects,
draw bounding boxes around them, and simulate part number and color identification.

*Note: This prototype uses a generic object detection model (YOLOv8). It simulates part numbers and colors for demonstration purposes. To identify actual Lego part numbers, a custom YOLO model trained on a Lego dataset needs to be substituted in `detector.py`.*
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.write("### Processing Image...")

    with st.spinner('Running detection model...'):
        # Run detection
        processed_image_np, detections = detect_lego_pieces(image_np)

    # Layout with columns to show original and processed image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected Pieces")
        st.image(processed_image_np, use_container_width=True)

    # Results Section
    st.write("---")
    st.write(f"### Detection Results ({len(detections)} pieces found)")

    if detections:
        # Convert detections to a Pandas DataFrame
        df = pd.DataFrame(detections)

        # Display the table without the raw bounding box coordinates for a cleaner look
        display_df = df.drop(columns=["Bounding Box"])
        st.dataframe(display_df, use_container_width=True)

        # Aggregate counts
        st.write("### Summary")
        summary_df = df.groupby(["Part Number", "Color"]).size().reset_index(name="Count")
        st.dataframe(summary_df, use_container_width=True)

        st.write("---")
        st.write("### Export Data")

        col3, col4 = st.columns(2)

        # Download CSV
        with col3:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download List as CSV",
                data=csv,
                file_name='lego_pieces.csv',
                mime='text/csv',
            )

        # Download JSON
        with col4:
            json_str = json.dumps(detections, indent=4)
            st.download_button(
                label="📥 Download List as JSON",
                data=json_str,
                file_name='lego_pieces.json',
                mime='application/json',
            )
    else:
        st.info("No pieces were detected in this image. Try uploading a clearer picture or one with larger objects.")
