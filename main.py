import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Combine description_x and description_y into a single text column
train_data['text'] = train_data['description_x'] + ' ' + train_data['description_y']
test_data['text'] = test_data['description_x'] + ' ' + test_data['description_y']

# Display top five lines of train data
st.write("Top five lines of train data:")
st.write(train_data.head())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['same_security'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the validation data
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Create a logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_tfidf, y_train)

# Create a random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)

# Create a support vector machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# Model selection
selected_model = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Support Vector Machine"])

if selected_model == "Logistic Regression":
    model = logreg_model
elif selected_model == "Random Forest":
    model = rf_model
else:
    model = svm_model

# Streamlit interface for testing
st.title("Text Classification App")

# Input text boxes for description_x and description_y
desc_x = st.text_area("Enter description_x:")
desc_y = st.text_area("Enter description_y:")

# Submit button
if st.button("Submit"):
    # Combine input texts
    combined_text = desc_x + ' ' + desc_y

    # Transform input using TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([combined_text])

    # Make prediction
    prediction = model.predict(input_tfidf)

    # Display result
    st.write(f"Predicted same_security: {prediction[0]}")

# Evaluate the efficiency of the selected model
val_predictions = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, val_predictions)
st.write(f"Efficiency of the model: {accuracy:.2%}")
