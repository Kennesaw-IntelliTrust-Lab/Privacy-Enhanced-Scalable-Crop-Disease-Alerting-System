import os
import openai
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the .env file
openai.api_key = os.getenv("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")

# Check if the API key was loaded correctly
if not openai.api_key:
    raise Exception("API key not found. Please make sure it is set in the .env file.")

# Load the model (ensure the model path is correct)
model = tf.keras.models.load_model('rice_disease_model.keras')

# Function to predict the disease
def predict_disease(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)ls

    class_names = ['RiceBlast', 'BrownSpot', 'BacterialLeafBlight']
    predicted_class = class_names[np.argmax(predictions)]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted disease: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class

# Function to query OpenAI API for answers (using newer API format)
# Function to query OpenAI API for answers (using newer API format)
def ask_openai(question):
    response = openai.ChatCompletion.create(  # Correct method: ChatCompletion.create
        model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
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


# Function to get questions based on the predicted disease
def get_farmers_questions(disease_name):
    # List of questions that the farmer may ask
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

# Main flow
def process_disease_image(image_path):
    # Predict the disease
    disease = predict_disease(image_path)
    
    print(f"Predicted Disease: {disease}")
    
    # Get the list of questions related to the disease
    questions = get_farmers_questions(disease)
    
    # Ask OpenAI for answers to those questions
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = ask_openai(question)
        print(f"Answer: {answer}\n")

# Example usage:
image_path = '/Users/chantirajumylay/Desktop/Privacy-Enhanced-Scalable-Crop-Disease-Alerting-System/Rice_disease_dataset/3_test/2_brownSpot/orig/brownspot_orig_009.jpg'
process_disease_image(image_path)
