import numpy as np
from model import create_model
from preprocessing import load_and_preprocess_images, split_data_and_save_samples
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Define dataset directory
dataset_dir = 'Rice_disease_dataSet/1_train'

# Load and preprocess images
images, labels, label_encoder = load_and_preprocess_images(dataset_dir)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data_and_save_samples(images, labels, sample_count=5)

# Create the model
input_shape = (224, 224, 3)
model = create_model(input_shape)

# Define a custom callback to track and calculate average accuracy across epochs
class AverageAccuracyCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.accuracies.append(logs.get('accuracy'))

    def on_train_end(self, logs=None):
        average_accuracy = np.mean(self.accuracies)
        print(f"Average Accuracy across epochs: {average_accuracy * 100:.2f}%")

# Instantiate the callback
average_accuracy_callback = AverageAccuracyCallback()

# Train the model with the callback
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[average_accuracy_callback])

# Save the model
model.save('rice_disease_model.h5')
print("Model saved as 'rice_disease_model.h5'")

# Optionally save the label encoder for later use
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Evaluate the model on the test set to get accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Optionally, you can also predict the classes and compute accuracy manually
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class indices
y_true_classes = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Calculate average accuracy on test set
average_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Average Accuracy on Test Set (manual calculation): {average_accuracy * 100:.2f}%")
