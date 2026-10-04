# Multimodal Image Registration

This project implements a complete pipeline for the geometric alignment of multimodal aerial images (Thermal and RGB sensors). The system estimates the optimal 2D roto-translation parameters to align a moving image to a reference image by maximizing their Mutual Information (MI).

## Features
- **Mutual Information Calculation:** Custom implementation of 2D joint histograms, joint probability distributions, and marginal entropies.
- **Direct Optimization:** Uses optimization methods (Powell, Nelder-Mead, or BFGS via `scipy.optimize`) to estimate the 3-parameter transformation (rotation and translation).
- **Evaluation:** Hyperparameter tuning on histogram bin sizes (64, 128, 256) and performance evaluation based on average translation and rotation error against Ground Truth data.

## Dataset
A lightweight dataset containing the RGB/Thermal image pairs and the Ground Truth CSV is **included in this repository**. No external downloads are required.

## Technologies Used
- Python
- NumPy
- OpenCV
- SciPy
- Marplotlib

## How to Run
1. Ensure the required libraries are installed.
2. Run the main script to process the validation and test pairs.
3. The script will output the aligned images, the difference images, and the quantitative error metrics.