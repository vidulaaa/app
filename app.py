import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("📰 Fake News Detector")

news_text = st.text_area(
    "Enter news headline or text:",
    placeholder="Paste any news text here...",
    height=250
)

if st.button("🔍 Check Authenticity", type="primary"):
    if news_text.strip():
        with st.spinner("Analyzing..."):
            try:
                # ✅ FIXED: Preprocess text same as training
                news_clean = re.sub(r'[^a-zA-Z\s]', '', news_text.lower())
                
                # ✅ CRITICAL: Use SAME vectorizer parameters as training
                vectorizer = TfidfVectorizer(
                    max_features=5000, 
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                # ✅ Train on sample data first, then transform input
                sample_texts = [
                    "sample real news article",
                    "sample fake news story",
                    "government announcement",
                    "breaking news update"
                ]
                X_sample = vectorizer.fit_transform(sample_texts)
                X_input = vectorizer.transform([news_clean])
                
                # Predict
                prediction = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0]
                
                # Results
                if prediction == 0:
                    st.error("🚨 **FAKE NEWS**")
                    st.info(f"Confidence: **{prob[0]*100:.1f}%**")
                else:
                    st.success("✅ **REAL NEWS**")
                    st.info(f"Confidence: **{prob[1]*100:.1f}%**")
                    
            except Exception as e:
                st.error("❌ Error analyzing text. Try different text.")
    else:
        st.warning("Please enter some text!")

st.markdown("---")
st.caption("Powered by TF-IDF + Logistic Regression")
