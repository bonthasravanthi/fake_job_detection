# Fake Job Posting Detection

## Project Overview
This project detects whether a job posting is Real or Fake using Machine Learning and Natural Language Processing (NLP). It uses TF-IDF Vectorization for text feature extraction and Logistic Regression for classification of job descriptions.
## Live Demo
🔗 Streamlit App:
https://fake-job-detection-sravanthi.streamlit.app/
## Features
- Predicts whether a job posting is real or fake.
- User-friendly interface built with Streamlit.
- Uses TF-IDF to convert text into numerical features.
- Fast and accurate predictions using NLP and Machine Learning.

## Technologies Used
Python – Programming language
Pandas – Data loading and preprocessing
NumPy – Numerical operations
Scikit-learn – Machine Learning and TF-IDF
TF-IDF Vectorizer – NLP feature extraction
SMOTE – Handling imbalanced data
Logistic Regression – Final Machine Learning model
Joblib – Model saving and loading
Streamlit – Web application deployment

## Project Structure

AI PROJECT/
│
├── dataset/
│ └── fake_job_postings.csv
│
├── models/
│ ├── fake_job_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md

## Machine Learning Model
- Algorithm: Logistic Regression
- Feature Extraction: TF-IDF Vectorizer

## How to Run

1. Install the required libraries:

pip install -r requirements.txt


2. Train the model:

python train_model.py

3. Run the Streamlit application:

streamlit run app.py


## Dataset
Dataset: Fake Job Postings Dataset

## Author
Sravanthi
