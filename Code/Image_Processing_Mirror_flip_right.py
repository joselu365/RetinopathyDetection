import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('C:/Users/JoseLu/Desktop/Fundus_dataflow/Code/Left_or_right_model_100_accuracy.h5')

# Set up data directory
data_dir = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/original'

# Set up output directory
output_dir = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/mirrored'

# Load the list of image files
img_list = os.listdir(data_dir)

# Loop through the images and make predictions
for img_path in img_list:
    # Load and preprocess the image
    img = image.load_img(os.path.join(data_dir, img_path), target_size=(1000, 1000))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Make predictions using the loaded model
    classes = model.predict(images, batch_size=10)
    probability_left = classes[0][0]
    probability_right = 1 - probability_left

    if probability_left >= 0.9:
        # Save original image for left eye
        img.save(os.path.join(output_dir, img_path.split('.')[0] + '_left_eye.jpg'))
    elif probability_right >= 0.9:
        # Flip and save image for right eye
        flipped_img = Image.fromarray(np.flip(img, axis=1))
        flipped_img.save(os.path.join(output_dir, img_path.split('.')[0] + '_right_eye_mirror_flipped.jpg'))
    else:
        # Save insufficient image
        img.save(os.path.join(output_dir, img_path.split('.')[0] + '_insufficient_image.jpg'))
