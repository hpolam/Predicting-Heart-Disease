# 🫀 Heart Disease Prediction Using Machine Learning

This project uses supervised machine learning models to predict the presence of heart disease based on clinical data. It was developed as a final project for the DSA460 course.

---

## 📂 Project Overview

Heart disease is a leading cause of death globally. Early prediction using patient metrics can improve clinical decision-making. This project builds predictive models using the UCI Heart Disease dataset.

---
## 📁 Dataset

- UCI Heart Disease dataset
- Includes age, sex, chest pain type, blood pressure, cholesterol, ECG results, etc.

---

## 🔧 Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
## 🧠 Models Used

| Model           | Notes |
|----------------|-------|
| Decision Tree   | Simple and interpretable, but overfit on small data |
| Random Forest   | Reduced variance via ensemble learning |
| K-Nearest Neighbors (KNN) | Best performing model in accuracy and F1-score |

---

## 📊 Tools & Technologies

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🧹 Data Preprocessing

- One-hot encoding for categorical variables
- Standard scaling for continuous features
- Train-test split (80/20)
- 10-Fold Cross-Validation

---

## ✅ Results

| Model          | Accuracy | Notes |
|----------------|----------|-------|
| Decision Tree  | ~73%   | Overfit on training data |
| Random Forest  | ~85%   | Balanced performance |
| **KNN**        | **~87%** | Highest performance overall |


---

## 📈 Visualizations

<p float="left">
  <img src="visuals/age_before.png" width="45%" />
  <img src="visuals/age_after.png" width="45%" />
</p>

---

## 🚀 Future Work

- Add larger datasets
- Experiment with logistic regression, SVM, or neural networks
- Continue experimenting with hyperparameterization
- Deploy as a web app 

---

## 🧪 How to Run

1. Clone this repo:
```bash
git clone https://github.com/hpolam/predicting-heart-disease.git
