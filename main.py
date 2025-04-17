import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import base64
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("bgimg.jpg")

st.markdown(
    f"""
<head>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>

<style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>

    """,
    unsafe_allow_html=True
)


#s3 details
BUCKET_NAME = 'plant-disease-prediction-system'
MODEL_KEY = 'model.h5'
LOCAL_PATH = 'model.h5'

#function to download model from s3
def download_model():
    s3 = boto3.client(
        's3',
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='eu-north-1'
    )

    s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_PATH)

#function to load model
def load_model():
    if not os.path.exists(LOCAL_PATH):
        download_model()
    return tf.keras.models.load_model(LOCAL_PATH)

working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = "model.h5"
# Load the pre-trained model
model = load_model()

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.markdown('<h1 class="pb-5 m-5 text-3xl md:text-5xl font-bold text-center text-green-800">☘️ Plant Disease Detector️</h1>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Diagnose'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f'<div class="py-3 px-2 rounded-md mt-4 bg-green-400 font-bold text-center text-black ">Diagnosis: {str(prediction)}</div>', unsafe_allow_html=True)
