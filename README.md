# 📱 Spam SMS / Email Classifier

> A machine learning classifier that detects spam SMS messages using TF-IDF features and classical NLP models, achieving 98.12% accuracy and 0.9861 AUC-ROC on the UCI SMS Spam Collection dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

---

## 📊 Model Performance

### Best Model: Logistic Regression

| Metric | Score |
|--------|-------|
| **Accuracy** | **98.12%** |
| **Precision** | 0.9776 |
| **Recall** | 0.8792 |
| **F1-Score** | 0.9258 |
| **AUC-ROC** | 0.9861 |
| **5-Fold CV F1** | 0.9237 (± 0.0158) |

### Confusion Matrix (Test Set — 1,115 messages)

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 963 | 3 |
| **Actual Spam** | 18 | 131 |

- **True Negatives (Ham correct):** 963
- **False Positives (Ham marked as Spam):** 3
- **False Negatives (Spam missed):** 18
- **True Positives (Spam caught):** 131

The model is highly precise — only 3 legitimate messages were wrongly flagged as spam. Recall is 87.9%, meaning it misses roughly 12% of spam, which is the main trade-off to address in future iterations.

---

## 🎯 Features

- TF-IDF vectorization with unigrams and bigrams (5,000 features)
- Text preprocessing: lowercasing, punctuation removal, stopword removal
- Four models compared: Naive Bayes, Logistic Regression, SVM, Random Forest
- Stratified 80/20 train-test split
- 5-fold cross-validation for robustness
- Full evaluation: accuracy, precision, recall, F1, AUC-ROC, confusion matrix

---

## 🚀 Quick Start

### Run in Google Colab

Open the notebook in Colab and run all cells. No GPU required — CPU finishes in under 2 minutes.

### 🛠️ Technologies Used
Technology	Purpose
scikit-learn	Modeling, TF-IDF, evaluation
NLTK	Stopword removal
Pandas / NumPy	Data handling
Matplotlib / Seaborn	Visualization
Joblib	Model serialization
Google Colab	Development environment
### 📄 Dataset
Source: UCI SMS Spam Collection

Metric	Value
Total messages	5,572
Spam messages	747 (13.4%)
Ham messages	4,825 (86.6%)
Train split	4,457 (80%)
Test split	1,115 (20%)
### 📈 Training Configuration
Parameter	Value
Vectorizer	TF-IDF (max 5,000 features, n-gram 1–2)
Train/test split	80/20 stratified
Cross-validation	5-fold
Models compared	4 (NB, LR, SVM, RF)
### 📊 Model Comparison
Model	Accuracy	Precision	Recall	F1
(from your notebook — fill in the table)				
Replace this table with the comparison from your notebook's comparison DataFrame.

### 🔍 How It Works
text
Raw SMS → Lowercase → Remove punctuation/numbers → Remove stopwords → TF-IDF → Classifier → Spam / Ham
### ⚠️ Limitations
Recall (0.879) is lower than precision (0.977) — the model misses roughly 12% of spam messages. For a production system, this trade-off could be tuned by adjusting the decision threshold or using class weighting.

Dataset is small (5,572 messages) and heavily imbalanced (13.4% spam).

Trained on English SMS only; not evaluated on emails or other languages.

### 🚀 Future Improvements
Tune classification threshold to improve recall

Add class weighting or SMOTE for imbalance

Try deep learning approaches (LSTM, BERT)

Extend to email spam classification

Deploy as a REST API with FastAPI

👩‍💻 Author
Alishba

https://img.shields.io/badge/GitHub-Alishba--12-blue?style=flat&logo=github


⭐ If you find this useful, please give it a star.
