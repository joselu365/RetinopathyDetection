"""
///////////////////////////////////////////////////////////////////////////////
//  Project: Development of an Automated Diabetic Retinopathy
//                   Detection System Based on Deep Learning Techniques
// Date:         22/01/2024
//
// File: utils.py
// Description: Contains utility functions for data manipulation, file operations, 
//              and other miscellaneous tasks to support the main application.
///////////////////////////////////////////////////////////////////////////////
"""

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


def load_data(folder_path, batch_size):
    """
    Loads data from a specified folder path, applies preprocessing, and creates training and validation generators.
    
    Args:
        folder_path (str): Path to the folder containing the dataset.
        batch_size (int): The size of the batches to use.
    
    Returns:
        tuple: A tuple of (train_generator, validation_generator) used for training and validation.
    """
    img_height = 512
    img_width = 512

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2  # Splitting the dataset into training (80%) and validation (20%).
    )

    train_generator = train_datagen.flow_from_directory(
        folder_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',  # Using 'categorical' because we assume a multi-class classification.
        subset='training'  # Specifies that this is the training dataset.
    )

    validation_generator = train_datagen.flow_from_directory(
        folder_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',  # Similarly for validation.
        subset='validation'  # Specifies that this is the validation dataset.
    )

    return train_generator, validation_generator

def build_model(train_generator, initial_learning_rate):
    """
    Builds a transfer learning model using InceptionV3 as the base model with a custom top layer.
    
    Args:
        train_generator (ImageDataGenerator): The training data generator.
        initial_learning_rate (float): Initial learning rate for the optimizer.
    
    Returns:
        Model: The compiled Keras model ready for training.
    """
    base_model = InceptionV3(weights='imagenet', include_top=False)  # Loading InceptionV3 as the base model without the top layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Adding a global spatial average pooling layer.
    x = Dense(1024, activation='relu')(x)  # Adding a fully connected layer.
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Final layer with softmax activation for classification.

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freezing the layers of the base model.
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the learning rate schedule.
    decay_steps = int(0.2 * len(train_generator))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True
    )

    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def load_and_prepare_image(file_path):
    """
    Load and prepare an image for model prediction.
    
    This function takes the path to an image file, loads the image,
    resizes it to a fixed size of 512x512 pixels, normalizes its pixel values
    to the range [0, 1] by scaling with a factor of 1/255, and adds a batch
    dimension to make it compatible with model input requirements.

    Parameters:
    - file_path (str): The path to the image file to be processed.
    
    Returns:
    - np.ndarray: A 4D numpy array of shape (1, 512, 512, 3) representing the
      processed image ready for model prediction, where 1 indicates the batch size,
      512x512 is the image size, and 3 stands for the three color channels (RGB).
    """
    img_height = 512
    img_width = 512
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # Load the image
    img = image.load_img(file_path, target_size=(img_height, img_width))
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Normalize the image (scale pixel values to [0, 1])
    img = normalization_layer(img)
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img