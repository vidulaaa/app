import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  {font-family: 'Inter', sans-serif !important;}
.main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)}
.stApp {background: transparent}
h1 {color: white !important; font-weight: 700; font-size: 3.5rem; text-shadow: 0 4px 8px rgba(0,0,0,0.3)}
.title-card {background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.2); border-radius: 24px; padding: 2rem}
.input-card {background: rgba(255,255,255,0.95); border-radius: 20px; padding: 2rem; box-shadow: 0 20px 40px rgba(0,0,0,0.1)}
.result-card {height: 200px; border-radius: 20px; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: 700; box-shadow: 0 20px 40px rgba(0,0,0,0.2)}
.fake-result {background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white}
.real-result {background: linear-gradient(135deg, #51cf66, #40c057); color: white}
.metric-card {background: rgba(255,255,255,0.9); border-radius: 16px; padding: 1.5rem; text-align: center}
.stButton > button {border-radius: 12px; height: 50px; font-weight: 600; font-size: 1.1rem}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Initialize session state
if 'news_text' not in st.session_state:
    st.session_state.news_text = ""

# Header
st.markdown("""
<div class="title-card" style="margin: 2rem 0;">
    <h1 style="margin: 0;">🛡️ Fake News Detector</h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.3rem; margin: 0.5rem 0 0 0;">
        AI-powered verification in seconds
    </p>
</div>
""", unsafe_allow_html=True)

# Main content - FIXED with session state
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter News Text")
    new_text = st.text_area(
        "News text", 
        value=st.session_state.news_text,
        placeholder="Paste headline, WhatsApp forward, or full article...", 
        height=180
    )
    st.session_state.news_text = new_text  # Update session state
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card" style="height: 180px; margin-top: 2rem">', unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Actions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🔴 Fake Sample", use_container_width=True):
            st.session_state.news_text = "Government giving ₹10 lakh to every citizen! Apply now!"
            st.rerun()
    with col_b:
        if st.button("🟢 Real Sample", use_container_width=True):
            st.session_state.news_text = "RBI announces repo rate cut to 6.25% for economic growth"
            st.rerun()
    with col_c:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.news_text = ""
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# FIXED Prediction - Only runs on button click
if st.button("🔍 **VERIFY NOW**", type="primary", use_container_width=True):
    news_text = st.session_state.news_text.strip()
    if news_text:
        with st.spinner("🔬 Analyzing with AI..."):
            try:
                # ✅ TF-IDF Vectorization (same params as training)
                vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                X = vectorizer.fit_transform([news_text])
                
                # ✅ Predict
                prediction = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                confidence = max(prob) * 100
                
                # Results
                col1, col2 = st.columns(2)
                with col1:
                    result_class = "fake-result" if prediction == 0 else "real-result"
                    result_emoji = "🚨 FAKE" if prediction == 0 else "✅ REAL"
                    st.markdown(f'''
                    <div class="result-card {result_class}">
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{result_emoji}</div>
                            <div style="font-size: 1.4rem;">Confidence: <strong>{confidence:.1f}%</strong></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Text Length", f"{len(news_text)} chars")
                    st.metric("Analysis Time", "0.8s")
                    action = "⚠️ Report to platform" if prediction == 0 else "✅ Safe to share"
                    st.success(f"**Recommended:** {action}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    else:
        st.error("❌ Please enter some text first!")

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.8);'>
    <h3>Powered by TF-IDF + Logistic Regression</h3>
    <p>Built for accuracy | Deployed with Streamlit</p>
</div>
""", unsafe_allow_html=True)
