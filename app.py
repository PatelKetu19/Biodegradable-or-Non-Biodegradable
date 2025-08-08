import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# Image preprocessing function
def preprocess_image(image: Image.Image, target_size):
    image = image.convert("RGB").resize(target_size)
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def classify_image(image: Image.Image):
    input_shape = input_details[0]['shape'][1:3]  # height, width
    processed = preprocess_image(image, input_shape)
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)
    class_index = int(np.argmax(prediction))
    return labels[class_index], float(np.max(prediction))

# Streamlit UI
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("♻️ Biodegradable vs Non-Biodegradable Classifier")

# Upload Section
uploaded_file = st.file_uploader("📁 Browse File", type=["jpg", "jpeg", "png"])

# Camera input
use_camera = st.toggle("📷 Use Camera")

if use_camera:
    image = st.camera_input("Take a photo")
else:
    image = uploaded_file

# Only show button if image is available
if image:
    if st.button("✅ Get Result"):
        img = Image.open(image)
        label, confidence = classify_image(img)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.success(f"Prediction: **{label}** ({confidence*100:.2f}%)")
