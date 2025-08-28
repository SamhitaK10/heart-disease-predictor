# ❤️ Heart Disease Predictor

An end-to-end machine learning project that predicts the likelihood of heart disease from health data using Python and scikit-learn.  
Built as a portfolio project to demonstrate data preprocessing, model training, and evaluation.

---

## 🚀 Features
- Data cleaning and preprocessing:
  - Label encoding (Yes/No categories → 0/1)
  - One-hot encoding for multi-class categorical features
- Trains two models:
  - **Decision Tree** (interpretable rules)
  - **Random Forest** (higher performance)
- Handles class imbalance with `class_weight="balanced"`
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion matrices (side-by-side)
  - Feature importances (Random Forest)

---

## 📂 Dataset
This project uses the **[Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)** dataset from the CDC.  

To run locally:
1. Download the dataset CSV from Kaggle  
2. Save it as `heart.csv` in the project folder  

---

## 🛠 Tech Stack
- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  

---

## ▶️ How to Run
Clone this repo and install requirements:

```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
python main.py

📊 Example Results

Decision Tree vs Random Forest on test data:

Model	Accuracy	Precision	Recall	F1
Decision Tree	0.91	0.15	0.10	0.12
Random Forest	0.94	0.28	0.22	0.25

(values are illustrative; your results may differ depending on dataset split)

🌟 What I Learned
- How to preprocess categorical health data for ML
- Trade-offs between model interpretability (Decision Trees) and performance (Random Forests)
- The impact of class imbalance on recall in medical datasets
