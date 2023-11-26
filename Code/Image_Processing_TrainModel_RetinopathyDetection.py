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

from keras.callbacks import ModelCheckpoint

IMG_HEIGHT = 1000
IMG_WIDTH = 1000
BATCH_SIZE = 32
EPOCH = 5

def load_data(folder_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        validation_split=0.2,
        subset="training",
        shuffle=True,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        validation_split=0.2,
        subset="validation",
        shuffle=True,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
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

folder_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/2Organized/'
train_ds, val_ds = load_data(folder_path)

# Class names atributtes aka health status
class_names = train_ds.class_names
print(class_names)

# Show samples of the dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# Notice the pixel values are now in `[0,1]`.
train_normalized_ds = normalized_model(train_ds)
val_normalized_ds = normalized_model(val_ds)

# show
image_batch, labels_batch = next(iter(train_normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

model = build_model(train_normalized_ds, val_normalized_ds)

######################## First run of the model ########################
################# Comment out to work on the same model #################
checkpoint_path = "C:/Users/JoseLu/Desktop/Fundus_dataflow/Training/cp_Health.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

# Run test training
model.fit(
    train_normalized_ds,
    validation_data=val_normalized_ds,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]  # Pass callback to training
)

######################## Load model from checkpoint and train ########################
# checkpoint_path = "C:/Users/JoseLu/Desktop/Fundus_dataflow/Training/cp_Health.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# checkpoint = ModelCheckpoint(
#     checkpoint_path, 
#     monitor='loss', 
#     verbose=1, 
#     save_best_only=True, 
#     mode='min'
# )

# # Load the latest checkpoint
# latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
# if latest_checkpoint:
#     # Restore the model and optimizer states
#     checkpoint.restore(latest_checkpoint)
#     print("Model restored from checkpoint.")

# # Continue training or use the loaded model for inference
# model.fit(
#     train_normalized_ds,
#     validation_data=val_normalized_ds,
#     epochs=EPOCH,
#     batch_size=BATCH_SIZE,
#     callbacks=[checkpoint]  # Pass callback to training
# )


######################## Save complet mode to H5 ########################
# model_final = "'C:/Users/JoseLu/Desktop/Fundus_dataflow/Training/model_Health.h5"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# # Run test training
# model.fit(
#     train_normalized_ds,
#     validation_data=val_normalized_ds,
#     epochs=EPOCH,
#     batch_size=BATCH_SIZE,
#     callbacks=[callbacks_list]  # Pass callback to training
# )