import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Simple title
st.title("📰 Fake News Detector")

# Single text area (exactly like your friend's)
news_text = st.text_area(
    "Enter news headline or text:",
    placeholder="Paste any news text here...",
    height=250
)

# Single predict button
if st.button("🔍 Check Authenticity", type="primary"):
    if news_text:
        with st.spinner("Analyzing..."):
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X = vectorizer.fit_transform([news_text])
            
            # Predict
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            
            # Results
            if prediction == 0:
                st.error("🚨 **FAKE NEWS**")
                st.info(f"Confidence: **{prob[0]*100:.1f}%**")
            else:
                st.success("✅ **REAL NEWS**")
                st.info(f"Confidence: **{prob[1]*100:.1f}%**")
    else:
        st.warning("Please enter some text!")

st.markdown("---")
st.caption("Powered by TF-IDF + Logistic Regression")
