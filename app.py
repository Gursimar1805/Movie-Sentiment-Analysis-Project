import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# MUST come after importing streamlit
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# --- Streamlit App UI ---

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #FF4B4B;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #B0B0B0;
    margin-bottom: 25px;
}

.info-box {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #FF4B4B;
    margin-bottom: 20px;
}

.footer {
    text-align:center;
    color: #888888;
    margin-top: 40px;
    font-size: 15px;
}

.author {
    text-align:center;
    color:#00D4FF;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🎬 Movie Review Sentiment Analyzer</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI-Powered Movie Review Classification using TF-IDF & Machine Learning
</div>
""", unsafe_allow_html=True)

# Project Info Card
st.markdown("""
<div class="info-box">
<b>About Project</b><br>
This application analyzes movie reviews and predicts whether the sentiment expressed is Positive 😊 or Negative 😞.
The model has been trained using Natural Language Processing (NLP) techniques and TF-IDF feature extraction.
</div>
""", unsafe_allow_html=True)

# Input Area
st.subheader("📝 Enter Movie Review")

user_input = st.text_area(
    "",
    placeholder="Example: This movie was absolutely fantastic. The acting and storyline were brilliant!",
    height=180
)

# Prediction Button
predict_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

if predict_btn:

    if user_input:

        with st.spinner("Analyzing Review..."):

            processed_input = clean_text(user_input)

            vectorized_input = tfidf_vectorizer.transform([processed_input])

            prediction = model.predict(vectorized_input)

            sentiment_map = {
                1: "Positive",
                0: "Negative"
            }

            predicted_sentiment = sentiment_map[prediction[0]]

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Review Length",
                      f"{len(user_input.split())} Words")

        with col2:
            st.metric("Processed Tokens",
                      f"{len(processed_input.split())}")

        st.subheader("🧹 Processed Review")
        st.info(processed_input)

        st.subheader("📊 Prediction Result")

        if predicted_sentiment == "Positive":
            st.success(
                "🎉 Positive Sentiment Detected\n\nThis review reflects a favorable opinion about the movie."
            )
        else:
            st.error(
                "😞 Negative Sentiment Detected\n\nThis review reflects an unfavorable opinion about the movie."
            )

        st.markdown(
            f"### 🎯 Predicted Sentiment: **{predicted_sentiment}**")

    else:
        st.warning("⚠️ Please enter a movie review first.")

# Sidebar
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4221/4221419.png",
        width=120
    )

    st.header("📌 Project Details")

    st.write("""
    **Model:** Machine Learning Classifier
    
    **Feature Extraction:** TF-IDF
    
    **Text Processing:**
    - HTML Removal
    - Lowercasing
    - Stopword Removal
    - Lemmatization
    
    **Task:** Binary Sentiment Classification
    """)

# Footer
st.markdown("---")

st.markdown(
    '<div class="author">👨‍💻 Developed By: Gursimar Singh Kohli</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer">B.Tech Computer Science Engineering (AI & ML)</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer">Built with Streamlit • NLP • Machine Learning</div>',
    unsafe_allow_html=True
)
