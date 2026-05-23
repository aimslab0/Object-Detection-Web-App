import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="YOLO Object Detection App",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Object Detection Web App")
st.write("Upload an image and detect objects using YOLOv8.")

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    results = model.predict(image_np, conf=confidence)

    detected_image = results[0].plot()

    with col2:
        st.subheader("Detected Image")
        st.image(detected_image, use_container_width=True)

    st.subheader("Detection Results")

    boxes = results[0].boxes

    if len(boxes) == 0:
        st.warning("No objects detected. Try lowering the confidence threshold.")
    else:
        names = results[0].names

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            conf_score = float(box.conf[0])

            st.write(f"✅ {class_name} — Confidence: {conf_score:.2f}")