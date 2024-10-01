import streamlit as st
import numpy as np
import cv2
from models import load_models
from detection import (
    generate_important_points, 
    generate_obb_corners, 
    generate_mask_corners
)
from prediction import (
    ktp_wrapped, 
    preprocess_wrapped_image, 
    generate_predictions
)

st.title("KTP DATA EXTRACTOR")

@st.cache_resource
def load_all_models():
    return load_models(
        segment_model_dir="segment_model.pt",
        obb_model_dir="obb_model.pt",
        processor_dir="donut_ktp_processor",
        donut_model_dir="donut_ktp_model"
    )

def predict(image, segment_model, obb_model, processor, model):
    bar = st.progress(0, text="Detect KTP")
    
    obb_points, segment_boxes, segment_masks = generate_important_points(image, segment_model, obb_model)
    
    if obb_points is None or segment_boxes is None or segment_masks is None:
        bar.progress(100, text="Completed")
        bar.empty()
        return "No objects detected in the image, skipping further processing."

    bar.progress(30, text="Get KTP Corner")
    obb_top_left, obb_top_right, obb_bottom_left, obb_bottom_right = generate_obb_corners(segment_boxes, obb_points)
    
    mask_top_left, mask_top_right, mask_bottom_left, mask_bottom_right = generate_mask_corners(
        obb_top_left, obb_top_right, obb_bottom_left, obb_bottom_right, segment_masks
    )
    
    bar.progress(50, text="Wrap Detected KTP")
    wrapped_img = ktp_wrapped(image, mask_top_left, mask_top_right, mask_bottom_left, mask_bottom_right)
    
    st.image(wrapped_img, channels="BGR")
    
    bar.progress(70, text="Preprocess Wrapped KTP")
    pixel_values, decoder_input_ids = preprocess_wrapped_image(processor, wrapped_img)
    
    bar.progress(80, text="Generate Output")
    output = generate_predictions(model, processor, pixel_values, decoder_input_ids)
    
    bar.progress(100, text="Completed")
    bar.empty()
    return output

segment_model, obb_model, processor, model = load_all_models()

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR")
    
    if st.button("Retrieve"):
        prediction = predict(image, segment_model, obb_model, processor, model)
        st.header("Data KTP")
        st.write(prediction)
