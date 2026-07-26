import pandas as pd
#load dataset
df = pd.read_csv("dataset/fake_job_postings.csv")

# Display first 5 rows
# print(df.head())

# Display dataset information
# print(df.info())

# Display rows and columns
# print(df.shape)

# Display column names
# print(df.columns)

# Fill missing values in categorical columns
df["employment_type"] = df["employment_type"].fillna("Unknown")
df["required_experience"] = df["required_experience"].fillna("Unknown")
df["required_education"] = df["required_education"].fillna("Unknown")
df["industry"] = df["industry"].fillna("Unknown")
df["function"] = df["function"].fillna("Unknown")
df["location"] = df["location"].fillna("Unknown")
df["department"] = df["department"].fillna("Unknown")
df["salary_range"] = df["salary_range"].fillna("Unknown")
df["company_profile"] = df["company_profile"].fillna("Unknown")
df["description"] = df["description"].fillna("Unknown")
df["requirements"] = df["requirements"].fillna("Unknown")
df["benefits"] = df["benefits"].fillna("Unknown")

#checking missing values
# print(df.isnull().sum())

# Combine important text columns
df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# Display the first 5 rows of the new text column
# print(df["text"].head())



from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF object
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

# Convert text into numerical features
X = tfidf.fit_transform(df["text"]).toarray()

# Target variable
y = df["fraudulent"]

# print(X.shape)
# print(y.head())




from sklearn.model_selection import train_test_split

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display shapes
# print("Training Data:", X_train.shape)
# print("Testing Data:", X_test.shape)


from imblearn.over_sampling import SMOTE
#checking the data before smote
# print("Before SMOTE:")
# print(y_train.value_counts())

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)
#checking the data after smote
# print("After SMOTE:")
# print(y_train.value_counts())


from sklearn.linear_model import LogisticRegression

# # Create the model
model = LogisticRegression(max_iter=1000)

# # Train the model
model.fit(X_train, y_train)

# # print("Model trained successfully!")

# # Make predictions
y_pred = model.predict(X_test)
# print(y_pred[:10])



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)


# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC

# from sklearn.metrics import accuracy_score, f1_score

# # Create models
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Naive Bayes": MultinomialNB(),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM": LinearSVC()
# }

# # Variables to store the best model
# best_model = None
# best_f1 = 0

# print("\nModel Comparison")
# print("-" * 40)

# Train each model one by one
# for name, model in models.items():

#     # Train model
#     model.fit(X_train, y_train)

#     # Predict on test data
#     y_pred = model.predict(X_test)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)

    # Calculate F1-score
    # f1 = f1_score(y_test, y_pred)

    # Print results
    # print("Model :", name)
    # print("Accuracy :", round(accuracy, 4))
    # print("F1 Score :", round(f1, 4))
    # print("-" * 40)

    # Check if current model is the best
    # if f1 > best_f1:
    #     best_f1 = f1
    #     best_model = model
    #     best_model_name = name

# Use the best model
# model = best_model

# Final prediction using the best model
# y_pred = model.predict(X_test)

# print("\nBest Model :", best_model_name)
# print("Best F1 Score :", round(best_f1, 4))
# print("Best model selected successfully!")



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# print(cm)



from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

# print(report)
# print(y_train.value_counts())
# print(y_test.value_counts())


# Save model after evaluation
import joblib

# Save best trained model
joblib.dump(model, "models/fake_job_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# print("Model and TF-IDF saved successfully!")