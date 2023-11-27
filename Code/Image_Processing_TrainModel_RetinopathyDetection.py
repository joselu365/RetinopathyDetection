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

from keras.callbacks import ModelCheckpoint, EarlyStopping

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid annoying errors

batch_size = 32
epoch = 200

def load_data(folder_path):
    img_height = 1000
    img_width = 1000
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

def build_model(train_normalized_ds, val_normalized_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_normalized_ds = train_normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_normalized_ds = val_normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    num_classes = 6

    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

folder_path = '/home/nnds3a/Documents/RadtEye/Database/2Organized'
train_ds, val_ds = load_data(folder_path)

# Class names atributtes aka health status
class_names = train_ds.class_names
print(class_names)

# Show samples of the dataset
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")


# Notice the pixel values are now in `[0,1]`.
train_normalized_ds = normalized_model(train_ds)
val_normalized_ds = normalized_model(val_ds)

# show
image_batch, labels_batch = next(iter(train_normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

model = build_model(train_normalized_ds, val_normalized_ds)

######################## Train model ######################## https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options

# # Evaluate the model
# loss, acc = model.evaluate(image_batch, labels_batch, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# # Loads the weights
# model.load_weights(checkpoint_path)

# # Re-evaluate the model
# loss, acc = model.evaluate(image_batch, labels_batch, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model_path = "Training/MyModel.keras"
checkpoint_path = "Training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.load_weights(checkpoint_path) # load checkpoint

# Create a callback that saves the model's weights
keras_callbacks   = [
    EarlyStopping(
        monitor='val_loss', 
        patience=30, 
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

model.save(model_path)

######################## Save complet mode to H5 ########################
# model_final = "/Training/model_Health.h5"
# checkpoint_path = "/Training/"

# model.load_weights(checkpoint_path) # load checkpoint

# model.save(model_final, save_format="h5")