# HexSoftwares_Image_Classification-Using-CNN-Model
Third Project for HexSoftware House as a Intern


# CIFAR-10 Image Classification using CNN and ResNet-style Model

This project is an image classification task completed as part of the **HexSoftwares Internship Project**.  
The objective is to classify images from the **CIFAR-10 dataset** into 10 predefined categories using deep learning models.

Two models are implemented and compared:

1. **Custom Convolutional Neural Network (CNN)**
2. **ResNet-style model with residual connections**

The complete implementation is provided in a Google Colab/Jupyter Notebook.

---

## Project Overview

Image classification is a computer vision task where a model learns visual patterns from image data and predicts the correct class/category of an image.

In this project, the models classify CIFAR-10 images into the following classes:

| Class ID | Class Name |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

---

## Dataset

The project uses the **CIFAR-10 dataset**, which is available through TensorFlow/Keras.

### Dataset Details

| Property | Value |
|---|---:|
| Training images | 50,000 |
| Testing images | 10,000 |
| Image size | 32 × 32 pixels |
| Color channels | 3 (RGB) |
| Number of classes | 10 |
| Pixel range before normalization | 0 to 255 |
| Pixel range after normalization | 0 to 1 |

The dataset is loaded directly using:

```python
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

---

## Project Workflow

The notebook follows a complete image classification pipeline:

1. Import required libraries
2. Load CIFAR-10 dataset
3. Display dataset description
4. Show sample images
5. Normalize image data
6. One-hot encode labels
7. Build CNN model
8. Train CNN model
9. Evaluate CNN model
10. Build ResNet-style model
11. Train ResNet-style model
12. Evaluate ResNet-style model
13. Compare CNN and ResNet-style accuracy
14. Plot accuracy and loss graphs
15. Display confusion matrix
16. Generate classification report
17. Test model on random images
18. Save trained models

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## Data Preprocessing

The following preprocessing steps are applied:

### 1. Label Flattening

Labels are converted from 2D format to 1D format for analysis and evaluation.

```python
y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()
```

### 2. Image Normalization

Pixel values are normalized from the range **0–255** to **0–1**.

```python
X_train_norm = X_train.astype("float32") / 255.0
X_test_norm = X_test.astype("float32") / 255.0
```

### 3. One-hot Encoding

Class labels are converted into one-hot encoded vectors for multi-class classification.

```python
from tensorflow.keras.utils import to_categorical

y_train_cat = to_categorical(y_train_flat, 10)
y_test_cat = to_categorical(y_test_flat, 10)
```

---

## Model 1: Custom CNN

The first model is a custom CNN built from scratch.

### CNN Architecture

The CNN model includes:

- Convolutional layers
- Batch normalization
- ReLU activation
- Max pooling
- Dropout
- Global average pooling
- Dense layer
- Softmax output layer

### CNN Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Categorical Crossentropy |
| Validation split | 20% |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

---

## Model 2: ResNet-style Model

The second model is a ResNet-style model designed for CIFAR-10 images.

### ResNet-style Architecture

The ResNet-style model uses residual blocks. Each residual block contains:

- Convolutional layer
- Batch normalization
- ReLU activation
- Second convolutional layer
- Batch normalization
- Shortcut connection
- Add layer
- Final ReLU activation

Residual connections help improve gradient flow and allow the model to learn deeper feature representations.

### ResNet-style Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 12 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Categorical Crossentropy |
| Validation split | 20% |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

---

## Model Evaluation

Both models are evaluated on the CIFAR-10 test set using:

- Test loss
- Test accuracy
- Classification report
- Precision
- Recall
- F1-score
- Confusion matrix
- Random image prediction testing

---

## Results

### Overall Model Comparison

| Model | Test Loss | Test Accuracy |
|---|---:|---:|
| CNN | 0.7216 | 0.7366 |
| ResNet-style | 0.7558 | 0.8116 |

The **ResNet-style model** achieved the best test accuracy.

---

## CNN Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Airplane | 0.85 | 0.66 | 0.75 | 1000 |
| Automobile | 0.89 | 0.89 | 0.89 | 1000 |
| Bird | 0.50 | 0.83 | 0.62 | 1000 |
| Cat | 0.55 | 0.60 | 0.58 | 1000 |
| Deer | 0.89 | 0.49 | 0.63 | 1000 |
| Dog | 0.67 | 0.59 | 0.62 | 1000 |
| Frog | 0.83 | 0.75 | 0.79 | 1000 |
| Horse | 0.76 | 0.83 | 0.79 | 1000 |
| Ship | 0.79 | 0.92 | 0.85 | 1000 |
| Truck | 0.93 | 0.81 | 0.87 | 1000 |

**CNN Accuracy:** 0.74  
**Macro Average F1-score:** 0.74  
**Weighted Average F1-score:** 0.74  

---

## ResNet-style Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Airplane | 0.85 | 0.80 | 0.82 | 1000 |
| Automobile | 0.94 | 0.88 | 0.91 | 1000 |
| Bird | 0.74 | 0.76 | 0.75 | 1000 |
| Cat | 0.62 | 0.69 | 0.65 | 1000 |
| Deer | 0.84 | 0.75 | 0.79 | 1000 |
| Dog | 0.79 | 0.66 | 0.72 | 1000 |
| Frog | 0.79 | 0.90 | 0.84 | 1000 |
| Horse | 0.91 | 0.81 | 0.86 | 1000 |
| Ship | 0.88 | 0.92 | 0.90 | 1000 |
| Truck | 0.82 | 0.94 | 0.87 | 1000 |

**ResNet-style Accuracy:** 0.81  
**Macro Average F1-score:** 0.81  
**Weighted Average F1-score:** 0.81  

---

## Best Performing Model

The **ResNet-style model** performed better than the custom CNN model.

| Metric | CNN | ResNet-style |
|---|---:|---:|
| Test Accuracy | 73.66% | 81.16% |
| Macro F1-score | 74% | 81% |
| Weighted F1-score | 74% | 81% |

This shows that residual connections helped the model learn better features and improved classification performance.

---

## Random Image Testing

The notebook also tests both trained models on random CIFAR-10 test images.  
For each image, the notebook displays:

- Actual class label
- Predicted class label
- Green title if prediction is correct
- Red title if prediction is incorrect

This helps visually check the model's prediction performance.

---

## Saved Models

After training, both models are saved in Keras format:

```text
cifar10_cnn_model.keras
cifar10_resnet_style_model.keras
```

These saved models can be loaded later for prediction or further training.

Example:

```python
from tensorflow.keras.models import load_model

cnn_model = load_model("cifar10_cnn_model.keras")
resnet_model = load_model("cifar10_resnet_style_model.keras")
```
---

## Key Learnings

Through this project, I learned:

- How image classification works
- How to load and explore CIFAR-10 dataset
- How to preprocess image data
- How to build a CNN model from scratch
- How to build a ResNet-style model using residual connections
- How to train and evaluate deep learning models
- How to compare models using accuracy, classification report, and confusion matrix
- How to test trained models on random images
- How to save trained Keras models

---

## Conclusion

This project successfully demonstrates image classification using deep learning.  
A custom CNN model and a ResNet-style model were trained and evaluated on the CIFAR-10 dataset.

The CNN model achieved **73.66%** test accuracy, while the ResNet-style model achieved **81.16%** test accuracy.  
The ResNet-style model performed better and was selected as the best-performing model for this project.

---

## Author

**Internship Project:** HexSoftwares Image Classification Task  
Implement By Fani2323
Computer Science 
**Dataset:** CIFAR-10  
**Models:** CNN and ResNet-style Model
