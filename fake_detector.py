import streamlit as st
import joblib
import os

if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    st.error("Model or vectorizer not found. Please train the model first.")
    st.stop()

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to detect whether it is **Fake** or **Real**.")

user_input = st.text_area("Enter news article text:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == "FAKE":
            st.error("This news is likely **FAKE** ❌")
        else:
            st.success("This news is likely **REAL** ✅")
