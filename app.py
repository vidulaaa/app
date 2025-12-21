import streamlit as st
import pickle

# load vectorizer and model
with open("TFIDF_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("Azhal_Logic_Regression_Model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Text classifier demo")

user_input = st.text_area("Enter text")

if st.button("Predict"):
    if user_input.strip():
        X = tfidf.transform([user_input])
        pred = model.predict(X)[0]
        st.write(f"Prediction: {pred}")
    else:
        st.warning("Please enter some text.")
