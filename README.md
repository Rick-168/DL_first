# 🧠 MNIST Digit Classification with Deep Learning

## 🚀 Project Overview
This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the well-known MNIST dataset. Built with TensorFlow and Keras, the model achieves high accuracy and visualizes predictions, making it easy to understand model performance.

## 🌟 Features

- **📊 MNIST Dataset**: 70,000 images (60,000 for training, 10,000 for testing) of handwritten digits in grayscale.
- **🤖 Deep Learning Model**: A CNN model designed for high accuracy in digit classification.
- **💻 High Accuracy**: Achieves impressive accuracy on the test dataset.
- **🎨 Interactive Visualization**: Shows input digits alongside model predictions.

## 🛠️ Tools & Technologies

- **Python** 
- **TensorFlow & Keras**: For building and training the neural network
- **NumPy**: For numerical calculations
- **Matplotlib**: For visualizations
- **Jupyter Notebook**: For interactive exploration and development

## 📂 Project Structure

- **digit_classification.ipynb**: The main Jupyter notebook for loading data, building, training, and evaluating the model.
- **README.md**: Documentation file (this file).

## 🧠 Model Architecture

The CNN model architecture includes:

1. **Input Layer**: 28x28 grayscale image of a handwritten digit.
2. **Convolutional Layers**: Extracts meaningful features from the input image.
3. **MaxPooling Layers**: Reduces dimensionality for efficient learning.
4. **Flatten Layer**: Converts 2D matrices into 1D vectors for the dense layers.
5. **Dense Layers**: Fully connected layers to classify features.
6. **Output Layer**: 10 units with softmax activation, representing each digit class (0–9).

## 🔍 Results

The model achieves **99% accuracy** on the MNIST test dataset after a few training epochs, demonstrating strong classification performance.
