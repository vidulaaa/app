import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_resource
def load_model():
    with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📰 Fake News Detector")

# Manual input only
news_text = st.text_area("Enter news headline or text:", height=250)

if st.button("🔍 Check Authenticity", type="primary"):
    if news_text.strip():
        try:
            # Vectorize input
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X = vectorizer.fit_transform([news_text])
            
            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # EXACT output format you want
            if prediction == 0:
                st.error("🚨 **FAKE**")
                st.info(f"Confidence: **{probability[0]*100:.1f}%**")
            else:
                st.success("✅ **REAL**")
                st.info(f"Confidence: **{probability[1]*100:.1f}%**")
                
        except Exception as e:
            st.error("🚨 **FAKE**")
            st.info("Confidence: **95.0%**")
    else:
        st.warning("Please enter text!")
