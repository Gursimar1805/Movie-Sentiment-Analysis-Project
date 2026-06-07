import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# PAGE CONFIG (Must be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# DOWNLOAD NLTK RESOURCES
# --------------------------------------------------
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --------------------------------------------------
# NLP SETUP
# --------------------------------------------------
lemmatizer = WordNetLemmatizer()

stops = set(stopwords.words('english'))
stops = stops - {'not', 'no', 'nor'}

# --------------------------------------------------
# LOAD MODEL & VECTORIZER
# --------------------------------------------------
try:
    with open('sentiment_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

except FileNotFoundError:
    st.error(
        "sentiment_model.pkl or tfidf_vectorizer.pkl not found."
    )
    st.stop()

# --------------------------------------------------
# PREPROCESSING FUNCTIONS
# --------------------------------------------------
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def convert_lower(text):
    return text.lower()

def remove_special(text):
    x = ''
    for i in text:
        if i.isalnum():
            x += i
        else:
            x += ' '
    return x

def clean_text(text):

    text = clean_html(text)
    text = convert_lower(text)
    text = remove_special(text)

    words = []

    for word in text.split():
        if word not in stops:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.main-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#FF4B4B;
    margin-bottom:5px;
}

.sub-title {
    text-align:center;
    color:#BDBDBD;
    font-size:18px;
    margin-bottom:20px;
}

.info-card {
    background: linear-gradient(135deg,#1F2937,#111827);
    padding:18px;
    border-radius:12px;
    border-left:6px solid #FF4B4B;
    margin-bottom:20px;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:20px;
}

.author {
    text-align:center;
    color:#00D4FF;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered Sentiment Classification using NLP & Machine Learning</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# PROJECT DESCRIPTION
# --------------------------------------------------
st.markdown("""
<div class="info-card">
<b>About Project</b><br><br>

This application predicts whether a movie review expresses a
<b>Positive</b> or <b>Negative</b> sentiment.

<b>Techniques Used:</b>

<ul>
<li>TF-IDF Vectorization</li>
<li>Text Cleaning & Lemmatization</li>
<li>Stopword Removal</li>
<li>Machine Learning Classification</li>
</ul>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("📌 Project Details")

    st.write("""
    **Domain:** Natural Language Processing
    
    **Dataset:** Movie Reviews
    
    **Feature Extraction:** TF-IDF
    
    **Text Processing**
    - HTML Removal
    - Lowercasing
    - Special Character Removal
    - Stopword Removal
    - Lemmatization
    
    **Output**
    - Positive Sentiment
    - Negative Sentiment
    """)

    st.markdown("---")

    st.success("Built using Streamlit")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.subheader("📝 Enter Movie Review")

user_input = st.text_area(
    "",
    placeholder="Example: This movie was absolutely fantastic. The acting, story and direction were brilliant.",
    height=180
)

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if user_input.strip():

        with st.spinner("Analyzing review..."):

            processed_input = clean_text(user_input)

            vectorized_input = tfidf_vectorizer.transform(
                [processed_input]
            )

            prediction = model.predict(vectorized_input)

            sentiment_map = {
                1: "Positive",
                0: "Negative"
            }

            predicted_sentiment = sentiment_map[prediction[0]]

            # Probability if model supports it
            confidence = None

            try:
                probs = model.predict_proba(vectorized_input)[0]
                confidence = max(probs)
            except:
                pass

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Review Length",
                f"{len(user_input.split())} words"
            )

        with col2:
            st.metric(
                "Processed Tokens",
                f"{len(processed_input.split())}"
            )

        st.subheader("🧹 Processed Review")

        st.info(processed_input)

        st.subheader("📊 Prediction Result")

        if predicted_sentiment == "Positive":

            st.success(
                "🎉 Positive Sentiment Detected"
            )

            st.markdown(
                "The review expresses a favorable opinion about the movie."
            )

        else:

            st.error(
                "😞 Negative Sentiment Detected"
            )

            st.markdown(
                "The review expresses an unfavorable opinion about the movie."
            )

        st.markdown(
            f"### 🎯 Predicted Sentiment: **{predicted_sentiment}**"
        )

        if confidence is not None:

            st.subheader("📈 Confidence Score")

            st.progress(float(confidence))

            st.write(
                f"Confidence: **{confidence*100:.2f}%**"
            )

    else:
        st.warning("Please enter a review first.")

# --------------------------------------------------
# SAMPLE REVIEWS
# --------------------------------------------------
st.markdown("---")

st.subheader("💡 Sample Reviews")

st.write(
    "Positive Example: "
    "*This movie was amazing. The acting and storyline were excellent.*"
)

st.write(
    "Negative Example: "
    "*The movie was boring and a complete waste of time.*"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    '<div class="author">👨‍💻 Developed By: Gursimar Singh Kohli</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer">B.Tech CSE-AIML</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer">Movie Review Sentiment Analysis Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer">Streamlit • NLP • Machine Learning</div>',
    unsafe_allow_html=True
)
