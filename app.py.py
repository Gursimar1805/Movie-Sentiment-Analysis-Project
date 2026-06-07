import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

@st.cache_resource
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="wide")
st.markdown("""<style>.positive {background-color: #d4edda; border-left: 5px solid #28a745; padding: 20px; border-radius: 5px; margin: 10px 0;} .negative {background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 20px; border-radius: 5px; margin: 10px 0;}</style>""", unsafe_allow_html=True)

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("### Analyze the sentiment of movie reviews using Machine Learning")

with st.sidebar:
    st.header("About")
    st.info("This application uses NLP and Machine Learning to predict whether a movie review is Positive or Negative.")
    if st.button("Try Positive Example"):
        st.session_state.sample_text = "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout."
    if st.button("Try Negative Example"):
        st.session_state.sample_text = "Terrible movie. Poor acting and boring plot. Complete waste of time."

col1, col2 = st.columns([2, 1])
with col1:
    default_text = st.session_state.get('sample_text', '')
    review_text = st.text_area("Enter a movie review:", value=default_text, height=200, placeholder="Type or paste a movie review here...")
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.markdown("### Quick Stats")
    if review_text:
        st.metric("Words", len(review_text.split()))
        st.metric("Characters", len(review_text))

if analyze_button:
    if not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        model, vectorizer = load_model()
        if model is None or vectorizer is None:
            st.error("❌ Model files not found!")
        else:
            with st.spinner("Analyzing sentiment..."):
                cleaned_text = clean_text(review_text)
                text_vector = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vector)[0]
                prediction_proba = model.predict_proba(text_vector)[0]
                st.markdown("---")
                st.subheader("Analysis Results")
                if prediction == 1:
                    st.markdown("<div class='positive'><h2>😊 Positive Sentiment</h2><p>This review expresses a positive opinion about the movie.</p></div>", unsafe_allow_html=True)
                    confidence = prediction_proba[1] * 100
                else:
                    st.markdown("<div class='negative'><h2>😞 Negative Sentiment</h2><p>This review expresses a negative opinion about the movie.</p></div>", unsafe_allow_html=True)
                    confidence = prediction_proba[0] * 100
                st.markdown(f"**Confidence Score:** {confidence:.2f}%")
                st.progress(confidence / 100)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>Built with ❤️ using Streamlit | Gursimar Singh Kohli | CSE-AIML</p></div>", unsafe_allow_html=True)