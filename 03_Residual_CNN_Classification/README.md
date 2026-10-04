# Land-Use Classification via Custom Residual CNN

This project explores Deep Learning techniques for classifying aerial images into 21 land-use categories. It focuses on designing a custom Convolutional Neural Network (CNN) with residual blocks and comparing different training strategies.

## Features
- **Custom Architecture:** A bespoke CNN featuring custom-built residual blocks, a flattening layer, and an MLP classifier with Dropout and Batch Normalization.
- **Training Strategies Comparison:**
  - *Strategy 1:* Transfer Learning. Pre-training the backbone on a large-scale dataset (AID) and fine-tuning it on the target dataset (UC Merced), experimenting with layer freezing.
  - *Strategy 2:* Training from scratch exclusively on the UC Merced dataset.
- **Robust Evaluation:** Implementation of Early Stopping, stratified dataset splitting (70% Train, 15% Val, 15% Test), and performance tracking (F1-score, Accuracy, Confusion Matrix).
- **Inference Script:** A complete pipeline for loading, pre-processing (resizing, normalization), and predicting new RGB images.

## Datasets & Setup
Please download the following datasets to reproduce the experiments:
1. **AID (Aerial Image Dataset)** - Used for the pre-training phase: [Download here](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)
2. **UC Merced Land Use Dataset** - Used for fine-tuning and from-scratch training: [Download here](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset)

**Directory setup:** Extract the datasets and place them in the same folder of the scripts, before running.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Scikit-Learn
- NumPy
- Matplotlib