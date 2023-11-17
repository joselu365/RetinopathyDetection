""" ///////////////////////////////////////////////////////////////////////////////
//                   Radt Eye
// Date:         14/11/2023
//
// File: Image_Processing_Resize_and_color_correct.py
// Description: Resize and color correct to have a consistan image set to feed
//              the model to be trained.
//
//      NOT WORKING
//
/////////////////////////////////////////////////////////////////////////////// """

import cv2  # to install (pip install opencv-python)
import numpy as np
import os

# Set the directory path and file extension for the eye fundus images
dir_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/1Flipped/'

# Set the desired output directory
out_dir = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/2Resized-ColorCorrect/'

# Set the desired suppression size
suppress_size = 1000

# Set the Hough circle detection parameters
dp = 1
minDist = 100
param1 = 50
param2 = 30
minRadius = 100
maxRadius = 0

# Set the color correction parameters
red_target = 105
green_target = 105
blue_target = 105
threshold = 50

# Loop over all the image files in the directory and process each one
for file_name in os.listdir(dir_path):
    # Check if the file is a supported image file
    if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.png') or file_name.lower().endswith('.tif') or file_name.lower().endswith('.tiff'):
        # Load the image
        img_path = os.path.join(dir_path, file_name)
        img = cv2.imread(img_path)

        # Convert non-JPG images to JPG format
        if not file_name.lower().endswith('.jpg') and not file_name.lower().endswith('.jpeg'):
            file_name = file_name.rsplit('.', 1)[0] + '.jpg'
            img_path = os.path.join(dir_path, file_name)
            cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Suppress the image to the desired size
        scale_percent = suppress_size / max(img.shape[:2])
        img = cv2.resize(img, (0,0), fx=scale_percent, fy=scale_percent)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Find circles using Hough circle detection
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        # Check if a circle was found
        if circles is not None and len(circles) == 1:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Assume the first circle detected is the fundus
            x, y, r = circles[0]

            # Crop the image to the diameter of the fundus
            img_crop = img[max(y-r, 0):y+r, max(x-r, 0):x+r]

            # Resize the image to the desired size
            img_crop = cv2.resize(img_crop, (suppress_size, suppress_size))

            # Color correction using a target color and threshold
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            img_mean = np.mean(img_crop, axis=(0,1))
            red_diff = red_target - img_mean[0]
            green_diff = green_target - img_mean[1]
            blue_diff = blue_target - img_mean[2]
            dist = np.sqrt(red_diff**2 + green_diff**2 + blue_diff**2)
            if dist > threshold:
                img_crop[:, :, 0] = np.clip(img_crop[:, :, 0] + red_diff, 0, 255)
                img_crop[:, :, 1] = np.clip(img_crop[:, :, 1] + green_diff, 0, 255)
                img_crop[:, :, 2] = np.clip(img_crop[:, :, 2] + blue_diff, 0, 255)
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, file_name), img_crop)


