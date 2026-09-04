SMS Spam Classifier

A machine learning model that classifies SMS text messages as Spam or Ham (not spam), built with TF-IDF feature extraction and a linear Support Vector Machine (SVM).

OVERVIEW

This project trains a text classifier to detect spam SMS messages. Raw message text is converted into numerical features using TF-IDF vectorization, and an SVM classifier is trained on those features to distinguish spam from legitimate messages.

MODEL

Vectorizer: TfidfVectorizer (scikit-learn)
- max_features = 5000
- ngram_range = (1, 2) — unigrams and bigrams
- stop_words = 'english'

Classifier: SVC (Support Vector Classifier)
- kernel = 'linear'
- probability = True
- random_state = 42

RESULTS

Evaluated on a held-out test set of 1,115 messages:

Metric        Ham     Spam
Precision     0.98    0.98
Recall        1.00    0.88
F1-score      0.99    0.93

Overall accuracy: 98%

Confusion Matrix:
                  Predicted Ham   Predicted Spam
Actual Ham        963             3
Actual Spam       18              131

The model rarely misclassifies real messages as spam (only 3 false positives), while catching the large majority of actual spam messages (131 of 149, about 88% recall).

See outputs/confusion_matrix/confusion_matrix.png for the visualized confusion matrix and outputs/sample_predictions/predictions.png for example predictions with confidence scores.

PROJECT STRUCTURE

.
├── models/
│   ├── spam_classifier.pkl      # Trained SVM classifier
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── outputs/
│   ├── confusion_matrix/
│   │   └── confusion_matrix.png
│   ├── metrics/
│   │   └── classification_report.txt
│   └── sample_predictions/
│       └── predictions.png
└── README.md

USAGE

Load the saved vectorizer and model to classify new messages:

import joblib

# Load trained artifacts
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/spam_classifier.pkl")

def predict(message: str) -> tuple[str, float]:
    """Classify an SMS message as Ham or Spam, with confidence."""
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    label = "Spam" if pred == 1 else "Ham"
    return label, proba

label, confidence = predict("Congratulations! You've won a free prize, claim now!")
print(f"{label} ({confidence:.1%} confidence)")

REQUIREMENTS

- Python 3.x
- scikit-learn
- joblib

Install dependencies:
pip install scikit-learn joblib

NOTES
The pickled artifacts were saved with scikit-learn 1.6.1. If you load them with a different scikit-learn version, you may see an InconsistentVersionWarning — for best results, use a matching version or re-train and re-save the artifacts in your current environment.
