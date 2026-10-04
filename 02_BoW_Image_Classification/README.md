# Aerial Image Classification via Bag-of-Visual-Words

This project implements a traditional Machine Learning pipeline to classify aerial RGB images into 21 distinct land-use categories using a Bag-of-Visual-Words (BoW) approach.

## Features
- **Feature Extraction:** Detects keypoints and extracts local descriptors (SIFT) using OpenCV.
- **Visual Vocabulary:** Uses K-Means clustering to group descriptors into $K$ visual words (centroids).
- **Histogram Representation:** Maps image descriptors to the nearest visual words to build normalized frequency histograms.
- **Classification:** Trains shallow classifiers (RBF SVM, Random Forest) via 3-fold cross-validation.
- **Inference Pipeline:** Includes a script to classify new, unseen images using the best-trained model and vocabulary.

## Datasets & Setup
Due to GitHub size limits, the datasets must be downloaded manually:
1. **AID (Aerial Image Dataset)** - Used *only* to build the visual vocabulary: [Download here](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)
2. **UC Merced Land Use Dataset** - Used to train and evaluate the classifiers (21 classes): [Download here](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset)

**Directory setup:** Extract the datasets and place them in the same folder of the scripts, before running.

## Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-Learn
- Pickle (for saving vocabulary and models)
- Matplotlib