import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Set the path to the dataset folder
dataset_path = 'C:/Users/Parmeet/Downloads/leaf detection new/Leaf Dataset/Leaf Dataset'

# Set the image dimensions and batch size
image_size = (64, 64)
batch_size = 32

# Create an ImageDataGenerator instance with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load the training data with augmentation
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Create an ImageDataGenerator instance for validation data
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load the validation data
val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define your updated model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Hyperparameter tuning
epochs = 20

# Train the model with augmented data
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the trained model
model.save('leaf_model_augmented.h5')
