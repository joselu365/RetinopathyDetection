""" ///////////////////////////////////////////////////////////////////////////////
//                   Radt Eye
// Date:         14/11/2023
//
// File: Image_Processing_TrainModel_RetinopathyDetection.py
// Description: AI model training for eye issues detection
/////////////////////////////////////////////////////////////////////////////// """

import matplotlib.pyplot as plt 
import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid annoying errors

batch_size = 16
epoch = 60

def load_data(folder_path):
    img_height = 512
    img_width = 512

    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        validation_split=0.2,
        subset="training",
        shuffle=True,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        validation_split=0.2,
        subset="validation",
        shuffle=True,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return train_ds, val_ds

def normalized_model(ds):
    # Standarize RGB from 0 to 1
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

def augment_model(ds):
    # Function for brightness adjustment
    def adjust_brightness(image):
        # Generate a random value for brightness adjustment between -0.2 and 0.2
        delta = tf.random.uniform([], -0.3, 0.3)
        return tf.image.adjust_brightness(image, delta=delta)

    # Function for random contrast adjustment
    def adjust_contrast(image):
        # Generate a random value for contrast adjustment between 0.8 and 1.2
        contrast_factor = tf.random.uniform([], 0.8, 1.2)
        return tf.image.adjust_contrast(image, contrast_factor=contrast_factor)

    # Data augmentation and normalization
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.Lambda(adjust_brightness),  # Apply brightness adjustment
        tf.keras.layers.Lambda(adjust_contrast),    # Apply contrast adjustment
    ])

    # Apply augmentation and normalization to the dataset
    augmented_samples = ds.map(lambda x, y: (data_augmentation(x), y))

    # Concatenate the augmented samples with the original dataset
    augmented_ds = ds.concatenate(augmented_samples)

    # Configure the dataset for performance
    augmented_ds = augmented_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return normalized_model(augmented_ds)

def build_model(train_normalized_ds, val_normalized_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_normalized_ds = train_normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_normalized_ds = val_normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    num_classes = 6
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu'),  # Increased filters in Conv2D layers
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),  # Increased filters
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),  # Increased filters
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),  # Increased units in Dense layer
        tf.keras.layers.Dropout(0.5),  # Adding dropout for regularization
        tf.keras.layers.Dense(num_classes)
    ])

    # Adjusted learning rate
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

folder_path = 'Database/NewBigMama/2Organized'
train_ds, val_ds = load_data(folder_path)

train_normalized_ds = augment_model(train_ds)
val_normalized_ds = normalized_model(val_ds)

model = build_model(train_normalized_ds, val_normalized_ds)

######################## Train model ######################## https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options
checkpoint_path = "Training2/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
keras_callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        mode='min',
        min_delta=0.00001
    ),
    ModelCheckpoint(
        checkpoint_path, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True,
        mode='min'
    )
]

# Train
model.fit(
    train_normalized_ds,
    validation_data=val_normalized_ds,
    epochs=epoch,
    batch_size=batch_size,
    callbacks=[keras_callbacks]  # Pass callback to training
)

model.save("Training2/NewLayer.h5")