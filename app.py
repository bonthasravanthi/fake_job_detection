import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/fake_job_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
# st.write("Model and TF-IDF loaded successfully!")
# Sidebar
st.sidebar.title("About Project")

st.sidebar.write("""
### Fake Job Detection System

**Purpose:**
Detect whether a job posting is Real or Fake.

**Machine Learning Model:**
Logistic Regression

**Feature Extraction:**
TF-IDF Vectorizer

**Techniques Used:**
- Text Processing
- Machine Learning Classification
- SMOTE for Data Balancing
""")

# Title
st.title("🔍 Fake Job Detection System")
st.write("Enter a job description below to check whether it is real or fraudulent.")

# # Input from user
job_text = st.text_area("Enter Job Description")

# # Button
if st.button("Predict"):
    if job_text.strip() == "":
        st.warning("Please enter a job description")
    else:
         # Convert text into numerical features
        text_vector = tfidf.transform([job_text])
         # Display converted data
        # st.write(text_vector)
        # Predict
        prediction = model.predict(text_vector)
         # Get probability
        probability = model.predict_proba(text_vector)
        # Display result
        if prediction[0] == 1:
            st.error("Fake Job ❌")
        else:
            st.success("Real Job ✅")
            # Display both probabilities
            st.write( "Real Job Probability:", round(probability[0][0] * 100, 2), "%")
        st.progress(int(probability[0][0] * 100))
        st.write("Fake Job Probability:", round(probability[0][1]*100, 2), "%")
        st.progress(int(probability[0][1]*100))