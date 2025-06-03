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
   - 📎 [Dataset on Kaggle](https://www.kaggle.com/datasets/omarhatif/datasets-for-ear-detection-and-recognition)

2. **EarVN1.0** (164 classes)  
   - Used for extended evaluation
   - Contains 164 classes
   - 📎 [EarVN1.0 on Mendeley](https://doi.org/10.17632/yws3v3mwx3.4)

## 📦 Requirements

I used Python 3.10.0 to ensure compatibility with all the components (Tensorflow, Keras).

Install dependencies in a virtual environment:

```
python -m venv .venv
```
or
```
python3.x -m venv .venv
```
```
source .venv/bin/activate
```
or on Windows:
```
.venv\Scripts\activate
```
```
pip install -r requirements.txt
```

## 🚀 Train the Model

```
python training/same_set.py
...
python training/other_set.py
```

## 🧪 Evaluate the Model

```
python evaluation/main.py
```

It will make a prediction for a random image from both datasets.

## 📊 My Results So Far

For the base dataset:

![accuracy](https://github.com/user-attachments/assets/ee2c1ed9-1c8f-4e99-8221-00541665f23d)

For the EarVN1.0 dataset:

![accuracyEarVN1 0](https://github.com/user-attachments/assets/4f315513-13bd-43b2-93f3-211bba6c324a)



The model is clearly struggling to generalize for 164 classes. Further architecture improvement is needed.

I tried to build another model, and it produced slightly better results:

![accuracy_deep_cnn_with_dropout_EarVN1 0](https://github.com/user-attachments/assets/1a86ac9c-1b4f-40d0-96ae-eb7374853f92)


However, after 30 epochs it struggles to learn, and the training accuracy increases without the increase in the validation accuracy.
