# Skin Cancer Detection using CNN

This project leverages **Convolutional Neural Networks (CNNs)** to detect **skin cancer** from images. It uses deep learning to classify skin lesions as benign or malignant, aiding in early diagnosis and reducing the dependency on manual screening.

## Project Overview

Skin cancer is one of the most common forms of cancer globally, and early detection significantly improves treatment outcomes. This project applies machine learning techniques to image classification in order to automatically detect cancerous skin lesions with high accuracy.

## Model Architecture

* The model is built using **TensorFlow** and **Keras**.
* It uses several convolutional and pooling layers followed by dense layers.
* **Activation functions** like ReLU and Softmax are used.
* The final output layer predicts the class of the lesion (malignant or benign).

## Dataset

* The dataset is obtained from Kaggle or publicly available skin lesion datasets.
* It includes labeled images for training and validation.
* Images are preprocessed using resizing, normalization, and data augmentation (e.g., rotation, zoom).

## Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **Confusion Matrix**
* **Loss Curves** and **Accuracy Curves** (visualized using matplotlib)

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Google Colab

## Outputs

* Model training accuracy and loss over epochs
* Confusion matrix on test dataset
* Random prediction outputs on test images

## How to Run

1. Open the [Colab Notebook](https://colab.research.google.com/drive/1lfmYOFCYqCE9zQ8ypkUxgMWn5D9yk0pU?usp=sharing)
2. Upload your dataset in the required format.
3. Run all the cells in order.
4. Monitor training and validation results.
5. Evaluate the model using test images.

## Author

**Dhruthi N**

