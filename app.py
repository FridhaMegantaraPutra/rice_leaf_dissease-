import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('rice_leaf_disease_model.h5')

# Define categories
CATEGORIES = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Function to preprocess image


def preprocess_image(image):
    IMG_SIZE = 32
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.astype(float) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict image class


def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return CATEGORIES[predicted_class]


# Streamlit app
st.title("Rice Leaf Disease Prediction")
st.write("Upload an image of a rice leaf to predict the disease category.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image)
    st.write(f"The predicted category is: {prediction}")
