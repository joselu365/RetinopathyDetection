""" ///////////////////////////////////////////////////////////////////////////////
//                   Radt Eye
// Date:         14/11/2023
//
// File: Image_Processing_TrainModel_RetinopathyDetection.py
// Description: AI model training for eye issues detection
/////////////////////////////////////////////////////////////////////////////// """


# DIRECTLY FROM CHAT GPT NEEDS WORKKKKKKKKKKKKKKKKKKKKKKKKKK



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the path to your dataset
dataset_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/2Resized-ColorCorrect/'

# Set the image dimensions and batch size
img_height, img_width = 1000, 1000
batch_size = 32

# Create an ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the data into training and validation sets
)

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Since there are two classes (right and left eyes)
    subset='training'  # Specify that this is the training set
)

# Load and preprocess the validation dataset
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify that this is the validation set
)

# Create a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Save the model
model.save('eye_detection_model.h5')
