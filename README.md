# 📱 Spam SMS / Email Classifier

> A machine learning system that detects spam messages with 98%+ accuracy using Natural Language Processing and multiple classification algorithms.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NVDHVG4sv64O70L_K432SSNAkGK-7IXs?hl=en-GB)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Model Performance

### Best Model: **Logistic Regression**

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.2% |
| **Precision (Spam)** | 0.97 |
| **Recall (Spam)** | 0.96 |
| **F1-Score (Spam)** | 0.97 |
| **AUC-ROC** | 0.99 |

### Confusion Matrix
<div align="center">
  <img src="outputs/confusion_matrix/confusion_matrix.png" width="70%">
</div>

### Classification Report

precision recall f1-score support

Ham 0.99 0.99 0.99 966
Spam 0.97 0.96 0.97 149

accuracy 0.98 1115
macro avg 0.98 0.98 0.98 1115
weighted avg 0.98 0.98 0.98 1115

---

## 🎯 Features

- ✅ **Text Preprocessing:** Cleaning, stopword removal, stemming, lemmatization
- ✅ **Feature Engineering:** TF-IDF vectorization with n-grams (unigrams + bigrams)
- ✅ **Multiple Models:** Naive Bayes, Logistic Regression, SVM, Random Forest
- ✅ **Comprehensive Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix
- ✅ **Confidence Scoring:** Each prediction includes confidence level
- ✅ **Batch Inference:** Test multiple messages at once
- ✅ **Model Persistence:** Saved model and vectorizer for deployment

---

## 📁 Project Structure

spam-sms-classifier/
├── models/
│ ├── spam_classifier.pkl # Trained model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── outputs/
│ ├── confusion_matrix/
│ │ └── confusion_matrix.png # Confusion matrix visualization
│ ├── metrics/
│ │ ├── classification_report.txt
│ │ └── training_metrics.png
│ └── sample_predictions/
│ └── predictions.png # 10+ sample predictions
├── data/
│ └── spam.csv # Dataset
├── notebooks/
│ └── Spam_SMS_Classifier.ipynb # Complete Colab notebook
├── requirements.txt
├── README.md
└── LICENSE

---

## 🚀 Quick Start

### Option 1: Try it in Google Colab (Easiest)

Click the badge above to open the notebook in Colab - no setup required!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/Alishba-12/spam-sms-classifier.git
cd spam-sms-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook notebooks/Spam_SMS_Classifier.ipynb
Option 3: Use the Trained Model
python
import joblib

# Load the model and vectorizer
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Predict a new message
def predict_spam(text):
    # Clean and vectorize
    cleaned = clean_text(text)  # Use preprocessing function
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    return {
        'text': text,
        'is_spam': bool(prediction),
        'confidence': max(probability)
    }

# Test it
result = predict_spam("WINNER! You've won a free iPhone!")
print(f"Spam: {result['is_spam']} (Confidence: {result['confidence']:.2%})")
📊 Sample Predictions
10+ Example Messages with Confidence Scores
<div align="center"> <img src="outputs/sample_predictions/predictions.png" width="90%"> </div>
#	Message	Prediction	Confidence
1	"Hey, are we still meeting for lunch tomorrow?"	✅ HAM	99.5%
2	"CONGRATULATIONS! You've won a free iPhone! Click here to claim now"	🚨 SPAM	98.7%
3	"Can you please send me the report by 5 PM?"	✅ HAM	98.9%
4	"URGENT: Your bank account has been compromised. Call immediately"	🚨 SPAM	99.2%
5	"I'll call you later tonight"	✅ HAM	99.1%
6	"Free tickets to Disneyland! Reply with your email address"	🚨 SPAM	97.8%
7	"Thanks for your help! I really appreciate it"	✅ HAM	98.5%
8	"Earn $5000 per day working from home. No experience needed"	🚨 SPAM	99.0%
9	"Don't forget to bring the documents to the meeting"	✅ HAM	97.5%
10	"Investment opportunity: Double your money in 30 days!"	🚨 SPAM	98.9%
11	"Your account has been suspended. Verify your identity now"	🚨 SPAM	97.2%
12	"Happy Birthday! Hope you have a wonderful day"	✅ HAM	98.8%
🔍 EDA Visualizations
Spam Messages Word Cloud
<div align="center"> <img src="outputs/spam_wordcloud.png" width="60%"> </div>
Message Length Distribution
<div align="center"> <img src="outputs/length_distribution.png" width="70%"> </div>
Key Insights from EDA:
📏 Spam messages are on average 5.2 words longer than ham messages

🔝 Top spam indicators: "free", "win", "claim", "urgent", "congratulations", "cash"

📊 Spam percentage: 13.4% of total messages

🎯 Best features: TF-IDF with bigrams (two-word phrases)

🛠️ Technologies Used
Technology	Purpose
Python 3.8+	Core programming language
scikit-learn	ML algorithms and preprocessing
NLTK	Natural Language Processing (tokenization, stopwords)
Pandas/NumPy	Data manipulation and analysis
Matplotlib/Seaborn	Data visualization
WordCloud	Text visualization
Joblib	Model serialization (save/load)
Google Colab	Cloud-based development environment
📈 Model Comparison
Model	Accuracy	Precision (Spam)	Recall (Spam)	F1-Score (Spam)
Naive Bayes	97.8%	0.96	0.94	0.95
Logistic Regression	98.2%	0.97	0.96	0.97
SVM (Linear)	98.0%	0.96	0.95	0.96
Random Forest	97.5%	0.95	0.93	0.94
🎯 How It Works
1. Data Preprocessing
text
Original: "WINNER!! You've won $1000!!!"
↓
Cleaned: "winner youve won"
↓
Stopwords Removed: "winner won"
↓
Vectorized: [0.5, 0.3, 0.8, ...] (TF-IDF features)
2. Feature Engineering
TF-IDF Vectorization with parameters:

max_features=5000

ngram_range=(1, 2) (unigrams + bigrams)

stop_words='english'

3. Model Training
80% training / 20% testing split

5-fold cross-validation

Hyperparameter tuning (GridSearchCV)

4. Inference Pipeline
text
New Message → Preprocess → Vectorize → Model → Prediction + Confidence
📄 Dataset
Source: UCI SMS Spam Collection

Statistics:

Metric	Value
Total messages	5,574
Spam messages	747 (13.4%)
Ham messages	4,827 (86.6%)
Avg words (spam)	15.6
Avg words (ham)	10.4
🚀 Future Improvements
□ Add deep learning models (LSTM, Transformers/BERT)
□ Support for email classification (longer texts)
□ Build real-time API using FastAPI
□ Create web interface with Streamlit/Gradio
□ Add multi-language support
□ Implement active learning for continuous improvement
□ Add explainable AI (SHAP/LIME for feature importance)
🤝 Contributing
Contributions are welcome! Here's how you can help:

Fork the repository

Create a new branch (git checkout -b feature/improvement)

Make your changes

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/improvement)

Open a Pull Request

👩‍💻 Author
Alishba
AI Enthusiast | Machine Learning Developer

https://img.shields.io/badge/GitHub-Alishba--12-blue?style=flat&logo=github
https://img.shields.io/badge/LinkedIn-Alishba-blue?style=flat&logo=linkedin
https://img.shields.io/badge/Twitter-@Alishba-blue?style=flat&logo=twitter

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
UCI Machine Learning Repository for the SMS Spam Collection dataset

scikit-learn community for excellent ML tools

NLTK for NLP capabilities

Google Colab for free GPU resources

Open-source community for inspiration and tools

⭐ Show Your Support
If you found this project helpful, please give it a ⭐ on GitHub!

Built with ❤️ and Python
