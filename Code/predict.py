"""
///////////////////////////////////////////////////////////////////////////////
//  Project: Development of an Automated Diabetic Retinopathy
//                   Detection System Based on Deep Learning Techniques
// Date:         28/01/2024
//
// File: predict.py
// Description: This script is designed to predict diabetic retinopathy in 
//              eye images using a deep learning model. It takes an image file 
//              as input, processes it, and then uses a trained model to predict 
//              the presence of diabetic retinopathy.
///////////////////////////////////////////////////////////////////////////////
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from utils import load_and_prepare_image  # Custom utility function for image loading and preprocessing
from keras.models import load_model

# Change console color for better readability (Windows-specific)
os.system('color')

def main():
    """
    Main function to handle image prediction workflow.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict diabetic retinopathy in an eye image.')
    parser.add_argument('filename', help='Path to the image file; should be a .jpg file.')
    parser.add_argument('--model', default='../model/good_model.h5', help='Path to the trained model file; default is "../model/good_model.h5".')
    
    args = parser.parse_args()

    # Load the image and prepare it for prediction
    try:
        image = load_and_prepare_image(args.filename)
    except Exception as e:
        print(f'ERROR: Unable to load the file from {args.filename}. Error: {e}')
        exit()

    # Load the trained model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f'ERROR: Unable to load the model from {args.model}. Error: {e}')
        exit()

    print(f'\nUsing model: {args.model}')
    print(f"Testing the image: {args.filename}")  # Log the image being tested
    
    # Perform prediction
    prediction = model.predict(image)
    
    # Calculate probabilities using softmax to determine the predicted class
    probabilities = tf.nn.softmax(prediction).numpy()
    predicted_class = np.argmax(probabilities, axis=1)
    
    # Display prediction results
    print("Probabilities:", probabilities)
    print("Predicted class:", predicted_class, "\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('\nERROR occurred during prediction. Please check the following:\n- Ensure the correct model is selected for your data.\n- Verify the path to the model file is correct.\nDetails:', e)

