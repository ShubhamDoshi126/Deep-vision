#Deep vision
#Fall detection for safety concerns regarding Elderly people.

# Fall Detection with MediaPipe and Deep Learning

This project demonstrates a human fall detection system using MediaPipe for pose estimation and a deep learning model for classification.

## Overview

The system works in two main stages:

1. **Pose Estimation with MediaPipe:** Extracts skeletal landmarks and calculates joint angles from images using MediaPipe.
2. **Fall Classification with Deep Learning:** A deep learning model (ResNet50 + custom layers) is trained to classify falls based on image features and extracted pose data.

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- OpenCV (cv2)
- MediaPipe
- Pandas
- NumPy

## Installation

1. Install the required libraries:
pip install mediapipe opencv-python tensorflow pandas numpy

2. Ensure you have a Google Colab environment or Jupyter Notebook setup.

## Usage

1. **Data Preparation:**
   - Place your images in the `input_folder` specified in the code.
   - The code will process images, extract pose data, and save it to `metadata_file`.
   - It will also save images with the skeleton drawn in `output_folder`.

2. **Model Training:**
   - The code defines a deep learning model that combines image features and pose data.
   - You need to load your training data (images, pose data, labels) and train the model.

3. **Fall Detection:**
   - Load the trained model.
   - Preprocess new images (resize, normalize) and extract pose data.
   - Feed the image and pose data to the model for predictions.

## Code Structure

- **Pose Estimation:** Uses MediaPipe to extract pose landmarks, calculate joint angles, and save the data to a CSV file.
- **Model Building:** Defines a deep learning model that combines image features and pose data.
- **Data Loading:** Loads images, pose data, and labels for training/prediction.
- **Training/Prediction:** Trains the model and performs predictions on new data.

## Notes

- The model is trained using the `categorical_crossentropy` loss function and the `adam` optimizer.
- Adjust the hyperparameters and model architecture as needed.
- The `metadata_file` stores the pose data extracted from the images.
- Ensure the image and pose data are correctly aligned during training and prediction.
