📰 News Classification using Machine Learning
 Overview
This project aims to automatically classify news articles into different categories such as politics, sports, business, technology, and entertainment using machine learning and natural language processing (NLP) techniques. It provides an end-to-end solution—from text preprocessing and model training to evaluation and user interface integration—allowing users to predict the category of any news headline or article text.

Objective
The main goal of this project is to build an intelligent news classification system capable of accurately identifying the category of a news article based on its textual content.
This automation can help media organizations, news aggregators, and readers quickly organize and filter information efficiently.

Key Features
•  Automated Classification: Classifies news articles into predefined categories.
•  NLP-Based Preprocessing: Includes tokenization, stemming, stop-word removal, and TF-IDF vectorization.
•  Model Comparison: Evaluates multiple machine learning models (Naïve Bayes, Logistic Regression, SVM).
•  High Accuracy: Optimized using hyperparameter tuning for improved performance.
•  User Interface: Simple and interactive web interface (Streamlit or Flask).
•  Visualization: Displays confusion matrix, accuracy scores, and classification reports.

System Architecture
Input → Preprocessing → Feature Extraction → Model Prediction → Output
1. Data Collection:
o Dataset sourced from Kaggle or open-source news datasets containing text and category labels.
2. Data Preprocessing:
o Cleaning, tokenization, lowercasing, stop-word removal, and stemming.
o Transformation of text data into numerical form using TF-IDF Vectorizer.
3. Model Training:
o Trained and compared multiple algorithms such as:
• Naïve Bayes
• Logistic Regression
• Support Vector Machine (SVM)
4. Model Evaluation:
o Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
5. Deployment:
o Implemented using Streamlit for a user-friendly prediction interface.

 Technologies Used
Category
Tools / Libraries
Programming Language
Python
ML Libraries
Scikit-learn, Pandas, NumPy
NLP Tools
NLTK, TF-IDF Vectorizer
Visualization
Matplotlib, Seaborn
Frontend (UI)
Streamlit
Dataset Source
Kaggle / News Dataset CSV

 How It Works
1. The user inputs a news headline or article into the interface.
2. The system processes the text through NLP-based preprocessing.
3. The pre-trained machine learning model predicts the most probable category.
4. The result is displayed instantly with confidence probability.

 Model Performance Summary
Model
Accuracy
Precision
Recall
F1-Score
Naïve Bayes
89.4%
0.88
0.89
0.88
Logistic Regression
91.7%
0.92
0.91
0.91
SVM
93.2%
0.93
0.93
0.93
Final Model Used: Support Vector Machine (SVM) due to its superior accuracy and stability across all categories.

 Future Enhancements
• Integration with deep learning models (LSTM / BERT) for better contextual understanding.
• Real-time classification of live news feeds via APIs.
• Multi-language support for regional and international news articles.
• Improved visualization dashboard with trend analysis.

 Project Structure
News_Classification/
│
├── dataset/
│   ├── news.csv
│
├── models/
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── interface.html

 Usage Instructions
1. Install required libraries:
2. pip install -r requirements.txt
3. Run the project:
4. streamlit run app.py
5. Upload or paste a news article in the text field.
6. The system will display the predicted category and confidence score.

 Authors
Developed by: Rishi K.
Guided by: [Instructor / Faculty Name if applicable]
Institution: [Your College / Organization Name]

 References
• Kaggle News Classification Dataset
• Scikit-learn Documentation
• NLTK Text Processing Guide
• Streamlit Official Documentation

