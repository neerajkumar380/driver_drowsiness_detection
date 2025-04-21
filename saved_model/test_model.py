import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("saved_model/drowsiness_model.h5")

# Define categories based on your dataset
CATEGORIES = ["Closed_Eyes", "Open_Eyes"]  
IMG_SIZE = (64, 64)  # Update if needed

# Function to predict drowsiness from an image
def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, IMG_SIZE)  # Resize
    img = np.array(img).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0  # Normalize and reshape
    
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return CATEGORIES[class_index]

# Example Test Image
image_path = "test_image1.png"  # Place a test image in the project folder
result = predict_image(image_path, model)
print(f"Drowsiness Prediction: {result}")
