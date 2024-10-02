import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, var=0.1):
    # Generate Gaussian noise
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    
    # Add the Gaussian noise to the image
    noisy_image = image + gauss

    # Clip the pixel values to stay in valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_noise_to_dataset(input_dir, output_dir, noise_function):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
                
                # Add noise to the image
                noisy_image = noise_function(image)
                
                # Save the noisy image
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, noisy_image)

# Example usage
input_directory = "/Users/chantirajumylay/Desktop/Privacy-Enhanced-Scalable-Crop1-Disease-Alerting-System/Rice_disease_dataset/1_train/3_Bacterialleafblight"
output_directory = "/Users/chantirajumylay/Desktop/Privacy-Enhanced-Scalable-Crop1-Disease-Alerting-System/Obfuscation_images/1_train/3_bacterialleafblight"
add_noise_to_dataset(input_directory, output_directory, add_gaussian_noise)
