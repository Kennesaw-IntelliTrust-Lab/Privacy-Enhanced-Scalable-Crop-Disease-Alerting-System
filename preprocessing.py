import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_images(dataset_dir, image_size=(224, 224)):
    """
    Load and preprocess images from a directory structure.
    Ignore XML files, and load images from the 'orig' and 'rotated' folders.
    """
    images = []
    labels = []
    
    # Loop through each class folder (e.g., 'Bacterial leaf blight', 'Brown Spot', etc.)
    for class_label in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_label)
        
        if os.path.isdir(class_path):  # Ensure it's a directory
            for subfolder in ['orig', 'rotated']:
                subfolder_path = os.path.join(class_path, subfolder)
                
                if os.path.isdir(subfolder_path):
                    for image_file in os.listdir(subfolder_path):
                        img_path = os.path.join(subfolder_path, image_file)

                        # Check if the file is an image
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            img = cv2.imread(img_path)

                            # Ensure the image is loaded correctly
                            if img is None:
                                print(f"Warning: Could not read image {img_path}. Skipping.")
                                continue

                            # Resize the image and append to the list
                            img = cv2.resize(img, image_size)
                            images.append(img)
                            labels.append(class_label)  # Use the class folder name as label

    # Convert lists to numpy arrays and normalize images
    images = np.array(images, dtype='float32') / 255.0  # Normalize pixel values to [0, 1]

    # Convert labels to numerical format using LabelEncoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(np.array(labels))

    # Check if images were loaded
    if len(images) == 0:
        raise ValueError("No images were loaded. Check the dataset directory structure.")

    print(f"Loaded {len(images)} images from {dataset_dir}.")
    return images, labels, label_encoder  # Return the encoder to decode predictions later

def split_data_and_save_samples(images, labels, test_size=0.2, sample_count=5, sample_dir='test_samples'):
    """
    Split the dataset into train and test sets, and save a few test samples.
    """
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    
    # Ensure the sample directory exists
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save a few sample images from the test set
    for i in range(min(sample_count, len(X_test))):
        sample_image = (X_test[i] * 255).astype(np.uint8)  # Convert back to original scale
        sample_label = y_test[i]
        sample_file_path = os.path.join(sample_dir, f'sample_{i}_{sample_label}.png')

        success = cv2.imwrite(sample_file_path, sample_image)
        if success:
            print(f"Sample image {i} saved as {sample_file_path}.")
        else:
            print(f"Failed to save sample image {i}.")
    
    return X_train, X_test, y_train, y_test
