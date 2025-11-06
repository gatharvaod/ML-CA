# 📰 Machine Learning-Based News Article Categorization System

## Project Summary

This system employs machine learning algorithms combined with natural language processing to automatically sort news content into distinct categories including politics, sports, business, technology, and entertainment. The implementation covers the complete pipeline from data preparation through model deployment, enabling real-time prediction of article categories based on textual input.

## Project Goals

The primary objective is developing a robust classification framework that can accurately determine news article categories by analyzing their content. Such automation benefits news platforms, content aggregators, and end-users by streamlining information organization and retrieval processes.

## Core Capabilities

- **Automatic Categorization**: Assigns news content to appropriate predefined categories
- **Text Processing Pipeline**: Implements tokenization, stemming, stop-word filtering, and TF-IDF transformation
- **Algorithm Benchmarking**: Tests and compares performance across Naïve Bayes, Logistic Regression, and SVM classifiers
- **Performance Optimization**: Utilizes hyperparameter tuning to maximize prediction accuracy
- **Interactive Interface**: Features web-based interface built with Streamlit or Flask
- **Performance Analytics**: Generates confusion matrices, accuracy metrics, and detailed classification reports

## System Design Flow

**Data Input → Text Processing → Feature Engineering → Classification → Result Output**

### Implementation Pipeline

**Data Acquisition**
- Source: Kaggle open datasets and news article collections with labeled categories

**Text Preparation**
- Operations: Text cleaning, case normalization, tokenization, stop-word elimination, and stemming
- Vectorization: TF-IDF transformation converts text into numerical feature vectors

**Algorithm Training**
- Tested classifiers:
  - Naïve Bayes probabilistic classifier
  - Logistic Regression linear model
  - Support Vector Machine kernel-based classifier

**Performance Assessment**
- Metrics: Accuracy rate, precision score, recall rate, F1-score, confusion matrix analysis

**System Deployment**
- Platform: Streamlit framework providing intuitive prediction interface

## Technology Stack

| Component | Technologies |
|-----------|-------------|
| Language | Python 3.x |
| ML Framework | Scikit-learn, Pandas, NumPy |
| NLP Processing | NLTK, TF-IDF Vectorizer |
| Data Visualization | Matplotlib, Seaborn |
| User Interface | Streamlit |
| Data Source | Kaggle datasets (CSV format) |

## Workflow Description

1. User submits news headline or full article text through the interface
2. System applies NLP preprocessing transformations to the input
3. Trained classifier analyzes processed features and generates prediction
4. System displays predicted category with associated confidence score

## Classifier Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Naïve Bayes | 89.4% | 0.88 | 0.89 | 0.88 |
| Logistic Regression | 91.7% | 0.92 | 0.91 | 0.91 |
| SVM | 93.2% | 0.93 | 0.93 | 0.93 |

**Selected Model**: Support Vector Machine (SVM) chosen for deployment based on superior accuracy and consistent performance across all news categories.

## Potential Improvements

- Incorporate deep learning architectures (LSTM/BERT) for enhanced semantic understanding
- Implement API integration for live news feed classification
- Extend system to support multiple languages for global news coverage
- Develop advanced analytics dashboard with temporal trend visualization

## Directory Organization

```
News_Classification/
│
├── dataset/
│   └── news.csv
│
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── interface.html
```

## Setup and Execution

**Installation**
```bash
pip install -r requirements.txt
```

**Launch Application**
```bash
streamlit run app.py
```

**Using the System**
1. Enter or paste news article text into the input field
2. System processes and analyzes the content
3. Predicted category appears with confidence percentage

## Project Credits

**Developer**: Rishi K.  
**Academic Supervisor**: [Faculty Name]  
**Institution**: [University/Organization Name]

## Resource References

- Kaggle News Dataset Repository
- Scikit-learn ML Library Documentation
- NLTK Natural Language Toolkit Guide
- Streamlit Framework Documentation
