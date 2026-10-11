import streamlit as st
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prepare text preprocessing
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()
    words = [
        word for word in words
        if word not in stop_words or word == "not"
    ]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

st.title("Amazon Review Sentiment Analysis")
st.write("Enter a product review to predict its rating.")

review = st.text_area("Enter your review")

if st.button("Predict"):
    if review.strip():
        cleaned_review = preprocess_text(review)
        features = vectorizer.transform([cleaned_review])
        prediction = int(model.predict(features)[0])

        if prediction in [4, 5]:
            sentiment = "Positive"
        elif prediction in [1, 2]:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.success(f"Predicted Rating: {prediction}/5")
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review first.")
