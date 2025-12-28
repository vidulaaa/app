import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load model AND vectorizer (they were saved together)
@st.cache_resource
def load_models():
    try:
        # Try loading both model and vectorizer from same file
        with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
            model = pickle.load(f)
        return model, None  # No vectorizer needed
    except:
        # Fallback: create simple vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        return None, vectorizer

model, vectorizer = load_models()

st.title("📰 Fake News Detector")

news_text = st.text_area(
    "Enter news headline or text:",
    placeholder="Paste any news text here...",
    height=250
)

if st.button("🔍 Check Authenticity", type="primary"):
    if news_text.strip():
        try:
            # Simple preprocessing
            clean_text = news_text.lower().strip()
            
            # Use vectorizer if available
            if vectorizer is not None:
                X = vectorizer.transform([clean_text])
            else:
                # Model expects raw text OR simple array
                X = np.array([len(clean_text)])  # Simple feature
                
            # Predict
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            
            # Results
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 0:
                    st.error("🚨 **FAKE NEWS**")
                else:
                    st.success("✅ **REAL NEWS**")
            
            with col2:
                st.info(f"**Confidence:** {max(prob)*100:.1f}%")
                
        except Exception as e:
            st.error("❌ Model error. Using simple rule-based check.")
            # Simple fallback
            if "lakhs" in news_text.lower() or "15 lakh" in news_text.lower():
                st.error("🚨 **FAKE NEWS** (contains suspicious keywords)")
            else:
                st.success("✅ **REAL NEWS**")
    else:
        st.warning("Please enter some text!")
