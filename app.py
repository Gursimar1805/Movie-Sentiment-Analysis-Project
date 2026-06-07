import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Use WordNetLemmatizer

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize lemmatizer and stopwords globally
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))
stops = stops - {'not', 'no', 'nor'} # Keep sentiment-altering stopwords

# Load the trained model and vectorizer
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error("Error: model.pkl or vectorizer.pkl not found. Make sure they are in the same directory as app.py")
    st.stop()

# --- Preprocessing functions (consistent with notebook) ---

def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def convert_lower(text):
    return text.lower()

def remove_special(text):
    x = ''
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x

def clean_text(text):
    # 1. Clean HTML tags
    text = clean_html(text)

    # 2. Convert to lowercase
    text = convert_lower(text)

    # 3. Remove special characters
    text = remove_special(text)

    # 4. Remove stop words and lemmatize
    words = []
    for i in text.split():
        if i not in stops:
            words.append(lemmatizer.lemmatize(i)) # Use lemmatizer

    # 5. Join back into a string
    return " ".join(words)

# --- Streamlit App ---
st.title('Sentiment Analysis App')
st.write('Enter a movie review below to predict its sentiment (Positive/Negative).')

user_input = st.text_area('Enter your review here:', '', height=150)

if st.button('Predict Sentiment'):
    if user_input:
        # Preprocess the user input
        processed_input = clean_text(user_input)

        # Vectorize the preprocessed input
        vectorized_input = tfidf_vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(vectorized_input)

        # Interpret prediction
        sentiment_map = {1: 'Positive', 0: 'Negative'}
        predicted_sentiment = sentiment_map[prediction[0]]

        st.write(f"**Processed Review:** {processed_input}")
        st.write(f"**Predicted Sentiment:** {predicted_sentiment}")

        if predicted_sentiment == 'Positive':
            st.success('This review expresses a **Positive** sentiment! 🎉')
        else:
            st.error('This review expresses a **Negative** sentiment! 😞')
    else:
        st.warning('Please enter some text to predict.')

# Footer
st.markdown("""
---
Created with Streamlit and a trained sentiment analysis model.
""")
