import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# model path
model_path = "model_tf.keras"

# Try to load the model with error handling
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Preprocessing parameters
target_size = (224, 224)  # Resize images to this size
rescale = 1. / 255  # Rescale pixel values

# ImageDataGenerator preprocessing (used for standardizing images)
infer_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=rescale
)

# Function to preprocess and predict the uploaded image
def predict_image(img):
    # Resize image to target size
    img = img.resize(target_size)
    # Convert to numpy array and scale
    img_array = np.array(img).astype("float32")
    # Expand dimensions for batch (model expects batches)
    img_array = np.expand_dims(img_array, axis=0)
    # Standardize the image using the ImageDataGenerator
    img_array = infer_datagen.standardize(img_array)
    
    # Make prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit app layout
st.title("Safety Gear Detection in Images")
st.write("Hacktiv8 - Phase 2 - Full Time Data Scientist")
st.write("Graded Challenge 7 - RMT 034 - Shinta Amalia")
st.markdown('---')

# Sidebar for uploading images
st.sidebar.subheader("Upload Image here")
uploaded_files = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        # Predict using the model
        prediction = predict_image(img)
        
        # Prediction result (Assuming binary classification, threshold at 0.5)
        # If the model is multi-class, adjust this logic accordingly
        if prediction[0][0] > 0.5:
            st.subheader(f"The image {uploaded_file.name} is classified as:")
            st.subheader(f"Wearing Safety Gear.")
        else:
            st.subheader(f"The image {uploaded_file.name} is classified as:")
            st.subheader(f"Not Wearing Safety Gear.")
        
        st.markdown('---')
else:
    st.info("Please upload an image for prediction.")