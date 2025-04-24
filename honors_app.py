import streamlit as st
import cv2
import numpy as np
import os
import random
from keras.models import load_model

# Load your trained model
# model = load_model(r"C:\Users\Ganesh\Downloads\ppe_honors.h5")

import requests
import tempfile

@st.cache_resource
def load_remote_model():
    url = "https://huggingface.co/ganeshmohane/ppe_detection/resolve/main/ppe_honors.h5"
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            return load_model(tmp.name)
    else:
        st.error("Failed to download model.")
        return None

model = load_remote_model()

# Define desired classes
desired_classes = ['ppe_suit', 'Face_Shield', 'Glove', 'Goggle', 'Mask']

def predict_ppe(image, model):
    # Preprocess the image
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)[0]
    
    # Set a threshold for predicted classes
    threshold = 0.4  # Adjust threshold as needed
    predicted_classes = [desired_classes[i] for i in range(len(predictions)) if predictions[i] > threshold]

    # If no classes predicted, print a warning
    if not predicted_classes:
        st.warning("No classes predicted, consider lowering the threshold.")

    return predicted_classes

def predict_random_image(model, base_path):
    # Get a list of all image files in the specified folder
    valid_extensions = ['.jpg', '.jpeg', '.png']  # Define valid image file extensions
    image_files = [f for f in os.listdir(base_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        st.warning("No images found in the specified directory.")
        return [], None

    # Select a random image from the list
    random_file = random.choice(image_files)
    image_path = os.path.join(base_path, random_file)
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (128, 128))  # Resize to match model input size
    image_array = np.array(image_resized, dtype='float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Predict using the model
    predictions = model.predict(image_array)[0]  # Get predictions for the image
    
    # Set a threshold for predicted classes
    threshold = 0.4  # Adjust threshold as needed
    predicted_classes = [desired_classes[i] for i in range(len(predictions)) if predictions[i] > threshold]
    
    # If no classes predicted, print a warning
    if not predicted_classes:
        st.warning("No classes predicted, consider lowering the threshold.")
    
    return predicted_classes, image_path  # Return both predictions and image path

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>🛡️ SafetyCheck</h1>", unsafe_allow_html=True)
st.write("**SafetyCheck** ensures that only authorized healthcare workers can enter the vicinity of infected patients by verifying their use of essential Personal Protective Equipment (PPE). Upload an image to check compliance with safety standards.")

# Dropdown for patient condition selection
patient_condition = st.selectbox("Select Patient Condition:", ("Infected", "Non-Infected", "Critical", "Others"))

# File uploader for user-uploaded images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_file is not None:
        # Predict the PPE in the uploaded image
        prediction_result = predict_ppe(uploaded_file, model)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write(f"Person is wearing {', '.join(prediction_result)}")
        
        # Logic based on patient condition
        if patient_condition == "Infected":
            # Check for essential PPE for infected patients
            required_classes = {'ppe_suit', 'Face_Shield', 'Mask', 'Glove'}
            if required_classes.issubset(set(prediction_result)):
                st.success("The healthcare worker is allowed to enter the patient room.")
            else:
                st.warning("The healthcare worker is NOT allowed to enter the patient room. Ensure all required PPE is worn.")

        elif patient_condition == "Non-Infected":
            # Check for essential PPE for non-infected patients
            required_classes = {'Glove'}
            if 'Glove' in prediction_result:
                st.success("The healthcare worker is allowed to enter the patient room.")
            else:
                st.warning("The healthcare worker is NOT allowed to enter the patient room. Ensure gloves are worn.")
                
        elif patient_condition == "Critical":
            # Check for essential PPE for critical patients
            required_classes = {'ppe_suit', 'Face_Shield', 'Glove', 'Mask'}
            if required_classes.issubset(set(prediction_result)):
                st.success("The healthcare worker is allowed to enter the critical patient room.")
            else:
                st.warning("The healthcare worker is NOT allowed to enter the critical patient room. Ensure all required PPE is worn.")
        
        elif patient_condition == "Others":
            # Check for essential PPE for other conditions (customize as needed)
            required_classes = {'Glove'}  # Assuming gloves are needed for other conditions
            if 'Glove' in prediction_result:
                st.success("The healthcare worker is allowed to enter the patient room for other conditions.")
            else:
                st.warning("The healthcare worker is NOT allowed to enter the patient room. Ensure gloves are worn.")
    else:
        st.error("Please upload an image.")

# Random image prediction button
base_path = r"D:\Desktop\PROJs\safteycheck\dataset\ppe_equipments\train"  # Replace with your actual base path
if st.button("Predict Random Image"):
    random_prediction, random_image_path = predict_random_image(model, base_path)
    
    if random_image_path:  # Check if an image path is returned
        st.image(random_image_path, caption='Randomly Picked Image', use_column_width=True)
    st.write(f"Person is wearing {', '.join(random_prediction)}")

# Feedback section
# feedback = st.text_area("Provide feedback on the prediction:")
# if st.button("Submit Feedback"):
#    st.success("Thank you for your feedback!")  
