import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

dataset_path = r"C:\Users\neera\OneDrive\Desktop\finalProject\train"  # Change this to your dataset path

# Define image size and categories
IMG_SIZE = (64, 64)  # Resize images to 64x64
CATEGORIES = ["Closed_Eyes", "Open_Eyes"]  # Modify according to your dataset labels

# Load images and labels
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(dataset_path, category)
    class_num = CATEGORIES.index(category)  # Assign numerical labels

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            img_array = cv2.resize(img_array, IMG_SIZE)  # Resize to fixed size
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img}: {e}")

# Convert to NumPy arrays
data = np.array(data).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)  # Add channel dimension
labels = np.array(labels)

# Normalize pixel values
data = data / 255.0

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print dataset shape
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#build cnn model for it

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: "Open" and "Closed"
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
#accuracy is a classification metrics means it is used to evaluate classification based model

# Print model summary
model.summary()

#training the model

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

model.save("saved_model/drowsiness_model.h5")
print("Model saved successfully!")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

