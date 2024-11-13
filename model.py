import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define class labels
class_labels = {0: "Bacterial leaf blight", 1: "Blast", 2: "Brownspot"}

def create_model(input_shape):
    """
    Creates and compiles a CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_labels), activation='softmax')  # Use len(class_labels) for number of classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_disease(model, image):
    """
    Predict the disease type given an image.
    """
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence  # Return class name and confidence

# Example usage (if needed):
# model = tf.keras.models.load_model('rice_disease_model.h5')
# img = cv2.imread('path_to_image.jpg')
# label, conf = predict
