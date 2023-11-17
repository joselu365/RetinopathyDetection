""" ///////////////////////////////////////////////////////////////////////////////
//                   Radt Eye
// Date:         14/11/2023
//
// File: Image_Processing_TrainModel_RetinopathyDetection.py
// Description: AI model training for eye issues detection
/////////////////////////////////////////////////////////////////////////////// """
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = img / 255.0
    return img, label

def load_data(folder_path):
    filenames = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, filename) for filename in filenames]
    labels = np.array([int(filename.split('')[1]) for filename in filenames])

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42)

    train_data = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_data = train_data.map(preprocess_image).batch(32)

    val_data = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_data = val_data.map(preprocess_image).batch(32)

    return train_data, val_data

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # You can add more convolutional layers if needed
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 categories (0-5)
    ])
    return model

def main():
    folder_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/1Flipped/'
    train_data, val_data = load_data(folder_path)

    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, validation_data=val_data, epochs=10)

    loss, accuracy = model.evaluate(val_data)
    print("Validation accuracy:", accuracy)