import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau

train_dir = 'RiceDiseaseDataset/1_train/'
validation_dir = 'RiceDiseaseDataset/2_validation/'
test_dir = 'RiceDiseaseDataset/3_test/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,git config --global credential.helper osxkeychain

    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical'
)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    layers.Dropout(0.5),  
    layers.Dense(3, activation='softmax')  
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

class AverageAccuracyCallback(Callback):
    def on_train_begin(self, logs=None):
        self.accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.accuracies.append(logs.get('accuracy'))
        running_avg_accuracy = np.mean(self.accuracies)
        print(f'Running average accuracy after epoch {epoch + 1}: {running_avg_accuracy:.4f}')

    def on_train_end(self, logs=None):
        avg_accuracy = np.mean(self.accuracies)
        print(f'Average accuracy over all epochs: {avg_accuracy:.4f}')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

average_accuracy_callback = AverageAccuracyCallback()

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, average_accuracy_callback]
)

model.save('rice_disease_model.keras')

def predict_disease(img_path):
  
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    predictions = model.predict(img_array)
    class_names = ['RiceBlast', 'BrownSpot', 'BacterialLeafBlight']
    predicted_class = class_names[np.argmax(predictions)]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted disease: {predicted_class}")
    plt.axis('off')  # Hide axis
    plt.show()

predict_disease('RiceDiseaseDataset/3_test/2_brownSpot/orig/brownspot_orig_009.jpg')