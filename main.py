import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

from util import set_background



# Function to crop and apply binary threshold to the image
def process_image(image):
    # Specify the coordinates of the rectangle
    x1, y1 = 500, 630
    x2, y2 = 628, 750
    
    # Crop the specified rectangle area
    cropped_image = image[y1:y2, x1:x2]
    
    # Convert the cropped image to grayscale
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary_image = cv2.threshold(grayscale_image, 245, 255, cv2.THRESH_BINARY)
    
    return cropped_image, binary_image

# Main Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Kolkata Thunderstorm Classification on Satellite Imagery 🌩️", page_icon=":cloud_with_lightning_and_rain:")
    
    set_background('bgs/1.png')
    # Title
    st.title("Kolkata Thunderstorm Classification on Satellite Imagery 🌩️")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a TIR based SATELLITE IMAGE 🌩️ ", type=["jpg", "jpeg"])
    
    # Load the classification model
    model = load_model("model_files\keras_Model.h5", compile=False)
    class_names = open("model_files\labels.txt", "r").readlines()
    
    # Check if image is uploaded
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="RGB")
        
        # Process the image
        cropped_image, binary_image = process_image(np.array(image))
        
        st.subheader("AI it WOoorRK ...")
        # Display cropped image and binary threshold image side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cropped Image")
            st.image(cropped_image, channels="BGR")
        with col2:
            st.subheader("Binary Threshold Image")
            st.image(binary_image, channels="GRAY")
        
        # Prepare the image for classification
        resized_image = cv2.resize(binary_image, (224, 224))
        normalized_image_array = (resized_image.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = np.stack((normalized_image_array,) * 3, axis=-1)
        
        # Predict using the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        s = 1 - confidence_score
        
        
        # Display classification results
        st.subheader("Classification Results")
        st.write("🌩️  SITUATION:", class_name[2:])
        st.write("CONFIDENCE LEVEL :", confidence_score ," condifent")
        st.write("SIGNIFICANCE LEVEL :", s ," condifent")

if __name__ == "__main__":
    main()
