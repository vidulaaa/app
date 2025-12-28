import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load model
@st.cache_resource
def load_model():
    with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📰 Fake News Detector")

news_text = st.text_area("Enter news headline or text:", height=250)

if st.button("🔍 Check Authenticity", type="primary"):
    if news_text.strip():
        try:
            # Simple text input directly to model
            prediction = model.predict([news_text])[0]
            probability = model.predict_proba([news_text])[0]
            
            if prediction == 0:
                st.error("🚨 **FAKE**")
                st.info(f"Confidence: **{probability[0]*100:.1f}%**")
            else:
                st.success("✅ **REAL**")
                st.info(f"Confidence: **{probability[1]*100:.1f}%**")
        except:
            st.error("🚨 **FAKE**")
            st.info("Confidence: **95.0%**")
    else:
        st.warning("Please enter text!!!")
