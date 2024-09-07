import os
import openai
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet

# Load environment variables from .env file
load_dotenv()

# Load the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("API key not found. Please make sure it is set in the .env file.")
openai.api_key = api_key

# Load encryption key for model
model_key = os.getenv("MODEL_KEY").encode()
if not model_key:
    raise Exception("Model encryption key not found. Please make sure it is set in the .env file.")

# Initialize the cipher suite
cipher_suite = Fernet(model_key)

def load_decrypted_model(model_path):
    """Decrypt and load the TensorFlow model."""
    with open(model_path, 'rb') as file:
        encrypted_model = file.read()
    decrypted_model = cipher_suite.decrypt(encrypted_model)
    with open('decrypted_model.keras', 'wb') as file:
        file.write(decrypted_model)
    model = tf.keras.models.load_model('decrypted_model.keras')
    return model

# Load the model
model_path = os.getenv("MODEL_PATH")
if not model_path:
    raise Exception("Model path not found. Please make sure it is set in the .env file.")
model = load_decrypted_model(model_path)

def predict_disease(img_path):
    """Predict the disease from an image."""
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_names = ['RiceBlast', 'BrownSpot', 'BacterialLeafBlight']
    predicted_class = class_names[np.argmax(predictions)]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted disease: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class

def ask_openai(question):
    """Query OpenAI API for answers."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        n=1,
        temperature=0.7,
    )
    answer = response.choices[0].message["content"].strip()
    return answer

def get_farmers_questions(disease_name):
    """Generate a list of questions related to the predicted disease."""
    questions = [
        f"What is {disease_name} in rice crops?",
        f"How can I treat {disease_name} in rice crops?",
        f"What are the symptoms of {disease_name}?",
        f"How can I prevent {disease_name} in the future?",
        f"What are the environmental conditions that favor {disease_name}?",
        f"Is {disease_name} dangerous for rice yields?",
        f"Can {disease_name} spread to other crops?",
        f"What chemicals should I use to treat {disease_name}?",
        f"How do I improve soil health to prevent {disease_name}?",
        f"When should I apply fungicides for {disease_name}?"
    ]
    return questions

def process_disease_image(image_path):
    """Process the image and query OpenAI API for answers based on the prediction."""
    disease = predict_disease(image_path)
    print(f"Predicted Disease: {disease}")
    
    questions = get_farmers_questions(disease)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = ask_openai(question)
        print(f"Answer: {answer}\n")

# Example usage
image_path = 'path_to_your_image.jpg'  # Replace with the path to your encrypted image
process_disease_image(image_path)
