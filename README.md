# Ear Recognition using CNN

This project implements a Convolutional Neural Network (CNN) for human ear recognition based on the architecture described in the article:

> Hidayati, N., Maulidah, M., & Saputra, E. P. (2022). *Ear identification using convolution neural network. Jurnal Mantik, 6*(2), 263–270. Retrieved May 29, 2025, from https://www.iocscience.org/ejournal/index.php/mantik/article/download/2263/1800 

## 🧠 Model Summary

The CNN model consists of:
- Grayscale input of size **128x128**
- 3 convolutional layers (16, 32, 64 filters)
- Max-pooling after each convolution
- A fully connected layer with 500 neurons
- Output layer with softmax activation for classification

## 📁 Dataset

Two datasets were used:

1. **Kaggle Dataset** (13 classes)  
   - Used for baseline testing
   - Structure: `train/`, `validation/`, `test/` folders each containing 13 class directories
   - Available from https://www.kaggle.com/datasets/omarhatif/datasets-for-ear-detection-and-recognition 

2. **EarVN1.0** (164 classes)  
   - Used for extended evaluation
   - Contains 164 classes
   - Available from https://doi.org/10.17632/yws3v3mwx3.4

## 📦 Requirements

Install dependencies in a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
or on Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Train the Model

```
python same_set.py
...
python other_set.py
```
