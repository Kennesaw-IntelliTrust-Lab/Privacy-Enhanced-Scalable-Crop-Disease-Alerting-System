import os
from PIL import Image
import numpy as np
import random

# Function to encrypt the image using XOR
def encrypt_image(image_path, output_dir):
    # Load the image
    image = Image.open(image_path)
    img_arr = np.array(image)
    
    img_rows, img_cols, channels = img_arr.shape
    
    # Create a key of the same size
    key_arr = np.random.randint(0, 256, size=(img_rows, img_cols, channels), dtype=np.uint8)
    
    # XOR operation between the image and the key
    encrypted_img_arr = np.bitwise_xor(img_arr, key_arr)
    
    # Convert back to image
    encrypted_image = Image.fromarray(encrypted_img_arr)
    
    # Generate file names
    base_filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    encrypted_image_path = os.path.join(output_dir, f"{filename_without_ext}_encrypted.png")
    key_path = os.path.join(output_dir, f"{filename_without_ext}_key.png")
    
    # Save the encrypted image and the key
    encrypted_image.save(encrypted_image_path)
    Image.fromarray(key_arr).save(key_path)
    
    print(f"Encrypted image saved to {encrypted_image_path}")
    print(f"Key saved to {key_path}")

# Function to process all images in a directory
def process_dataset(dataset_dir, output_dir):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the dataset
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}...")
                encrypt_image(image_path, output_dir)

# Paths
dataset_dir = # Set this to the directory containing your dataset'  # Set this to the directory containing your dataset
output_dir =  # Set this to where you want to save obfuscated images

# Start processing the dataset
process_dataset(dataset_dir, output_dir)
