"""
///////////////////////////////////////////////////////////////////////////////
//  Project: Development of an Automated Diabetic Retinopathy
//                   Detection System Based on Deep Learning Techniques
// Date:         22/01/2024
//
// File: train.py
// Description: Orchestrates the training process of the machine learning model,
//              including data loading, model initialization,
//              and training loop execution.
///////////////////////////////////////////////////////////////////////////////
"""

import tensorflow as tf
import pickle
from utils import load_data, build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuring GPU for memory growth to avoid allocating all memory upfront
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Setting up training parameters
batch_size = 8  # Number of samples processed before the model is updated
epoch = 500  # Number of complete passes through the training dataset
initial_learning_rate = 0.001  # Starting learning rate for training
patience = 20  # Number of epochs with no improvement after which training will be stopped

# Path definition
folder_path = '../Database/2Organized'
save_path = f"./model"
model_name = f"good_model.h5"
history_name = f"History_good_model.pickle"

# Load data using the load_data function
train_generator, validation_generator = load_data(folder_path, batch_size)

# Build model using the build_model function
model = build_model(train_generator, initial_learning_rate)

# Setup callbacks for early stopping and model checkpointing
keras_callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience),
    ModelCheckpoint(f"{save_path}/{model_name}", monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epoch,
    shuffle=True,
    callbacks=keras_callbacks  # Callbacks for monitoring
)

# Save the training history for later analysis
with open(f"{save_path}/{history_name}", 'wb') as history_file:
    pickle.dump(history.history, history_file)
    
# Save the model as an h5 file
model.save(f"{save_path}/{model_name}")