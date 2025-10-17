# Veritas Vision: Explainable AI for Image Authenticity

## Overview

In an era where AI-generated images are becoming increasingly indistinguishable from real ones, "Veritas Vision" offers a robust solution for classifying images and understanding the basis of that classification. This project utilizes a deep learning model, specifically a Vision Transformer (ViT), to not only predict whether an image is from a known dataset (like CIFAR-10) but also to provide visual explanations for its decisions using Gradient-weighted Class Activation Mapping (Grad-CAM).

This repository contains two main components:

1.  **Model Training (`FinalProject_GenAI_Image_Prediction.ipynb`):** A Jupyter notebook for training and saving the Vision Transformer model.
2.  **Visualization (`FinalProject_GenAI_Visualization.ipynb`):** A Jupyter notebook to load the trained model, make predictions on new images, and generate Grad-CAM visualizations to interpret the results.

## Key Features

* **High-Accuracy Classification:** Fine-tunes a pre-trained Vision Transformer (ViT) model for the image classification task, leveraging the power of transfer learning.
* **Explainable AI (XAI):** Implements Grad-CAM to produce heatmaps that highlight the specific regions of an image the model focused on to make its prediction.
* **Modular Code:** The project is separated into distinct notebooks for training and visualization, making the workflow easy to understand and manage.
* **Tech Stack:** Built with Python and leading machine learning libraries including PyTorch, `timm`, and Matplotlib.

## How It Works

### 1. Model Training

The `FinalProject_GenAI_Image_Prediction.ipynb` notebook handles the entire training pipeline:

* **Data Loading:** It loads the CIFAR-10 dataset and applies necessary transformations (resizing, normalization).
* **Model Architecture:** A pre-trained Vision Transformer model is loaded from the `timm` library, and its classification head is adapted for the 10 classes of CIFAR-10.
* **Training Loop:** The model is trained using the Adam optimizer and CrossEntropyLoss. A validation loop runs at the end of each epoch to monitor performance.
* **Saving the Model:** Once training is complete, the model's learned weights are saved to a `.pth` file for later use in inference and visualization.

### 2. Prediction and Visualization

The `FinalProject_GenAI_Visualization.ipynb` notebook uses the trained model to provide insights:

* **Model Loading:** It reconstructs the ViT architecture and loads the saved weights from the training phase.
* **Image Processing:** It takes new images from a `test_images` directory and prepares them for the model.
* **Grad-CAM Implementation:** For each image, it calculates the Grad-CAM heatmap, which shows the model's "attention" areas.
* **Output:** The notebook displays the original image alongside the image with the heatmap overlay, clearly labeling the model's top predictions and their confidence scores.

## Technology Stack

* **Programming Language:** Python
* **Machine Learning:** PyTorch, `timm` (PyTorch Image Models)
* **Data Handling:** NumPy
* **Visualization:** Matplotlib, PIL
* **Environment:** Jupyter Notebook

## Setup and Usage

To get this project running on your local machine, follow these steps:

**1. Clone the Repository**

```bash
git clone [https://github.com/your-username/veritas-vision.git](https://github.com/your-username/veritas-vision.git)
cd veritas-vision
