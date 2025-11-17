import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.title("🔬 Nanoparticle Area Analyzer (SEM Images)")

st.write("""
Upload an SEM image, enter your scale bar information, and the app will detect nanoparticles,
outline them, and compute their areas.
""")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload SEM image", 
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# Stop here until an image is uploaded
if uploaded_file is not None:

    # Read image into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Inputs for scale ---
    micron_scale = st.number_input(
        "Scale bar length (microns)", min_value=0.0, value=1.0, step=0.1
    )

    pixel_scale = st.number_input(
        "Pixel length of scale bar", min_value=0.0, value=100.0, step=1.0
    )

    if micron_scale > 0 and pixel_scale > 0:

        # Convert to nm/pixel
        nm_per_pixel = (micron_scale * 1000) / pixel_scale
        st.write(f"**Nanometers per pixel:** {nm_per_pixel:.3f} nm/px")

        # --- Preprocessing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        st.subheader("Thresholded Image")
        st.image(thresh, use_column_width=True)

        # --- Contour detection ---
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        areas_px = []
        outlined_img = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                areas_px.append(area)
                cv2.drawContours(outlined_img, [cnt], -1, (0, 0, 255), 1)

        # --- Area calculations ---
        areas_nm2 = np.array(areas_px) * (nm_per_pixel ** 2)
        areas_um2 = areas_nm2 / 1e6

        # --- Statistics ---
        avg_area = np.mean(areas_um2) if len(areas_um2) > 0 else 0
        std_area = np.std(areas_um2) if len(areas_um2) > 0 else 0

        st.subheader("📊 Results")
        st.write(f"**Detected particles:** {len(areas_um2)}")
        st.write(f"**Average area:** {avg_area:.4f} µm²")
        st.write(f"**Std. deviation:** {std_area:.4f} µm²")

        # --- Outlined image ---
        st.subheader("Outlined Nanoparticles")
        st.image(outlined_img, use_column_width=True)

        # --- Histogram ---
        st.subheader("Histogram of Nanoparticle Areas")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(areas_um2, bins=30, color='steelblue', edgecolor='black')
        ax.set_xlabel("Particle area (µm²)")
        ax.set_ylabel("Count")
        ax.set_title("Nanoparticle Size Distribution")
        st.pyplot(fig)

        # --- Download CSV ---
        df = pd.DataFrame(areas_um2, columns=["Area_µm²"])
        csv_bytes = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download CSV of Areas",
            data=csv_bytes,
            file_name="nanoparticle_areas.csv",
            mime="text/csv"
        )

        # --- Download outlined image ---
        _, buffer = cv2.imencode(".jpg", outlined_img)
        st.download_button(
            "Download Outlined Image",
            data=buffer.tobytes(),
            file_name="sem_outlined.jpg",
            mime="image/jpeg"
        )
