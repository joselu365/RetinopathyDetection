# Development of an Automated Diabetic Retinopathy Detection System Based on Deep Learning Techniques

## Description
This project involves a machine learning model for predicting outcomes based on input images. It includes scripts for training the model, making predictions, preprocessing data, and utility functions to assist these processes.

## Getting Started

### Dependencies
Ensure you have the following packages installed:
- TensorFlow (and Keras)
- NumPy
- Pickle
- shutil
- csv
- argparse
- os

You can install these dependencies via pip:
```bash
pip install tensorflow numpy pickle5 shutil csv argparse os
```

### Setting Up
**Download the Required Model**: 
- Ensure you have downloaded the necessary model file (`good_model.h5`). The model file should be placed in a directory named `model` at the root of the project directory. 
- The model file can be downloaded from the following link: [Download good_model.h5](https://hawhamburgedu-my.sharepoint.com/:f:/g/personal/wpv825_haw-hamburg_de/Er8SdAaErbpDjX0_aHVmUOABDQV1k8MHwE-oINvkmHid2g?e=tKYs1c). Please ensure that you have access rights to download the file.

### Training the Model
- Use `data_to_file.py` for preparing your data set. This script is designed to rename and process files according to the project's requirements.
- To train the model, run `train.py`. This script orchestrates the model's training process, including data loading, model initialization, and saving the trained model and history.
- Ensure your dataset is structured appropriately, as expected by the script. For custom dataset structures, modify `utils.py` accordingly.

### Making Predictions
- Use `predict.py` to make predictions with the trained model. This script takes an image file as input, processes it, and uses the trained model to predict the outcome.
- Example usage:
  ```bash
  python predict.py --image path/to/your/image.jpg --model path/to/your/model/good_model.h5
  ```

### Utilities
- `utils.py` includes functions for model building, data loading, and image processing, supporting both the training and prediction processes.

## Demo
Two Jupyter notebooks are included to demonstrate the functionalities of this model:

- `Demo_Use.ipynb`: This notebook shows how to use the pre-trained model to make predictions on new data. It guides you through loading the model, preparing your data, and finally, making and interpreting predictions.

- `Demo_Training.ipynb`: This notebook provides a detailed walkthrough of how to train the model from scratch using your dataset. It covers the entire process from data preprocessing, model training, evaluating performance, and saving the trained model for future use.

These notebooks are designed to help you understand the workflow of using and training the model, making it easier to integrate into your projects or adapt to your needs.
