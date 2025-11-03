# 🏠 Boston Housing Price Prediction using Backpropagation Neural Network (BPNN)

This project implements a **Backpropagation Neural Network (BPNN)** from scratch using only **NumPy** to predict **house prices** based on the **Boston Housing dataset**.  
It demonstrates fundamental deep learning concepts like **feedforward propagation**, **gradient descent**, **weight updates**, and **error minimization** — all without any deep learning frameworks.

---

## 📘 Project Overview

The model is built entirely from first principles to help understand how neural networks learn.  
It includes:
- One input layer  
- One hidden layer  
- One output layer  
- Configurable learning rate and hidden neurons  

Model performance is evaluated using **5-fold** and **10-fold cross-validation** across different learning rates and hidden layer sizes.

---

## ⚙️ Features

✅ Implementation from scratch using **only NumPy**  
✅ User-defined hyperparameters (learning rate, hidden neurons, folds)  
✅ Cross-validation (5-fold and 10-fold)  
✅ Mean Squared Error (MSE) based loss  
✅ Visualization of training loss per epoch  
✅ Boston Housing dataset preprocessed and normalized  
✅ Detailed result comparison table  

---

## 🧠 Neural Network Configuration

| Configuration | Hidden Neurons | Learning Rate | Epochs | Cross-Validation |
|----------------|----------------|----------------|---------|------------------|
| Case (a) | 3 | 0.01 | 1000 | 5-fold / 10-fold |
| Case (b) | 4 | 0.001 | 1000 | 5-fold / 10-fold |
| Case (c) | 5 | 0.0001 | 1000 | 5-fold / 10-fold |

---

## 📊 Results Summary

| Hidden Neurons | Learning Rate | 5-Fold Loss | 10-Fold Loss |
|----------------|----------------|--------------|---------------|
| 3 | 0.01 | 0.3919 | 0.3909 |
| 4 | 0.001 | 0.4505 | 0.4523 |
| 5 | 0.0001 | 3.5754 | 3.6033 |

> The results show that higher learning rates with fewer hidden neurons yielded lower loss, indicating faster convergence and better generalization.

---

## 🧩 Project Structure

Boston_Housing_Price_Prediction_BPNN/
│
├── housing.csv
├── BPNN.ipynb
├── model.py
├── utils.py
├── evaluate.py          # (contains evaluate_model)
└── README.md



---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/Boston_Housing_Price_Prediction_BPNN.git
   cd Boston_Housing_Price_Prediction_BPNN



## Author
**Roll Number**: 23IE10006  
**Course**: ES60011 - Application of Machine Learning in Biological Systems  
**Institution**: IIT Kharagpur

## License
This project is for educational purposes as part of coursework.
