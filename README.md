# Machine Learning Practice

This repository contains a comprehensive collection of Jupyter notebooks and Python scripts for implementing and understanding key concepts in machine learning, deep learning, and PyTorch.
It serves as a hands-on guide for learners and students for the course Applied Machine Learning, focusing on practical implementations and a final project example.

---

## Author:
- Keeby (@Keeby-Astro)

---

## **Table of Contents**

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Key Notebooks and Concepts](#key-notebooks-and-concepts)
4. [Additional Scripts](#additional-scripts)
5. [Installation](#installation)
6. [Usage](#usage)
7. [License](#license)

---

## **Getting Started**
This repository provides an applied approach to understanding machine learning concepts, using **Python** and the **PyTorch** deep learning framework.

---

## **Project Structure**
The repository is organized as follows:

```plaintext
Machine-Learning-Practice/
|
├── Basics of PyTorch and OOP.ipynb
├── Basics to Intermediate Concepts with Python.ipynb
├── Basics to Intermediate NumPy Concepts with Python.ipynb
├── Batch Norm_Learning Rate_Normal vs Gradient Descent.ipynb
├── Convolutional Neural Networks.ipynb
├── Data Augmentation.ipynb
├── Introduction to Python.ipynb
├── Introduction to Random Variables, Probability, and Likelihood.ipynb
├── Linear Regression.ipynb
├── Logistic Regression and Binary Classification.ipynb
├── MNIST Classification using PyTorch Lightning.ipynb
├── Multi-Layer Perceptron (MLP) Model with PyTorch.ipynb
├── NumPy with Python.ipynb
├── Recurrent Neural Networks.ipynb
├── Regression Model With Pytorch.ipynb
├── Regression With Pytorch.ipynb
├── Transformers with Pytorch.ipynb
├── F107 Solar Radio Flux CNN-Transformer.py
└── F107 Solar Radio Flux Plotter.py
```

---

## **Key Notebooks and Concepts**
Here are the highlights of each notebook:

### **1. Basics and Python Foundations**
- **`Introduction to Python.ipynb`**: Introduction to Python programming for beginners.
- **`NumPy with Python.ipynb`**: Fundamental concepts and operations using NumPy.
- **`Basics to Intermediate Concepts with Python.ipynb`**: Python programming exercises for ML foundations.
- **`Basics to Intermediate NumPy Concepts with Python.ipynb`**: Essential and advanced operations with NumPy.

### **2. Regression Models**
- **`Linear Regression.ipynb`**: Implementation of linear regression from scratch.
- **`Logistic Regression and Binary Classification.ipynb`**: Logistic regression for binary classification tasks.
- **`Regression Model With Pytorch.ipynb`**: Building regression models using PyTorch.
- **`Regression With Pytorch.ipynb`**: Advanced PyTorch-based regression models.

### **3. Deep Learning with PyTorch**
- **`Basics of PyTorch and OOP.ipynb`**: Introduction to PyTorch and object-oriented programming.
- **`Batch Norm_Learning Rate_Normal vs Gradient Descent.ipynb`**: Understanding batch normalization and learning rates.
- **`Multi-Layer Perceptron (MLP) Model with PyTorch.ipynb`**: Implementing MLPs for classification.

### **4. Convolutional Neural Networks (CNNs)**
- **`Convolutional Neural Networks.ipynb`**: Introduction and implementation of CNNs.
- **`Data Augmentation.ipynb`**: Techniques to augment data for improving CNN performance.
- **`MNIST Classification using PyTorch Lightning.ipynb`**: Using PyTorch Lightning to classify handwritten digits.

### **5. Sequence Models**
- **`Recurrent Neural Networks.ipynb`**: Implementing RNNs for sequential data.
- **`Transformers with Pytorch.ipynb`**: Understanding and applying Transformers using PyTorch.

### **6. Probability and Statistics**
- **`Introduction to Random Variables, Probability, and Likelihood.ipynb`**: Core statistical concepts for ML.

---

## **Additional Scripts**
These Python scripts focus on advanced applications and visualizations for time series data:

### **F107 Solar Radio Flux Prediction**

#### **1. F107 Solar Radio Flux CNN-Transformer.py**
- **Purpose**: Predicts the 10.7-cm solar radio flux using a hybrid CNN-Transformer model.
- **Key Features**:
  - Preprocesses time series data for solar flux measurements.
  - Implements a hybrid model combining **CNN** (for local features) and **Transformer Encoder** (for global dependencies).
  - Provides visualizations such as residual plots, loss curves, and prediction scatter plots.
- **Input**: `solar_flux.txt` (Solar flux measurements dataset).
- **Output**: Predictions, residual plots, and performance metrics for `Obs`, `Adj`, and `URSI-D` columns.

#### **2. F107 Solar Radio Flux Plotter.py**
- **Purpose**: Visualizes solar flux measurements.
- **Key Features**:
  - Cleans and preprocesses solar flux data.
  - Generates combined and separate time series plots for `Obs`, `Adj`, and `URSI-D` columns.
  - Ensures data consistency by handling missing and invalid values.
- **Output**: Plots showcasing trends in solar flux measurements.

---

## **Installation**
To use the notebooks and scripts in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Machine-Learning-Practice.git
   cd Machine-Learning-Practice
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Navigate to the desired notebook or script and execute cells to explore the concepts and run code.
3. For standalone Python scripts, run them directly using:
   ```bash
   python script_name.py
   ```

## **License**
This project is licensed under the MIT License.

---

### **Acknowledgments**
Special thanks to:
- PyTorch developers and the open-source community.
- The professor for Applied Machine Learning
