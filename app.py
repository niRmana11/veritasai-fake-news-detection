import streamlit as st
import pickle
import re

# ---------------------------
# Load trained artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    with open("models/veritasai_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/veritasai_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("models/label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    inverse_label_map = {v: k for k, v in label_map.items()}
    return model, vectorizer, inverse_label_map

model, vectorizer, inverse_label_map = load_artifacts()

# ---------------------------
# Text cleaning (same as training)
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="VeritasAI", page_icon="📰", layout="centered")

st.title("📰 VeritasAI")
st.subheader("AI-Based Fake News Detection")
st.write("Paste a news headline and article to check credibility.")

# Input fields
title = st.text_input("News Title")
text = st.text_area("News Content", height=200)

# Predict button
if st.button("Analyze News"):
    if not title or not text:
        st.warning("Please enter both title and content.")
    else:
        content = clean_text(title + " " + text)
        features = vectorizer.transform([content])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        real_conf = probabilities[1]

        if features.nnz < 30:
            verdict = "⚠️ Outside Training Domain"
        elif real_conf > 0.7:
            verdict = "✅ Likely REAL"
        elif real_conf < 0.3:
            verdict = "❌ Likely FAKE"
        else:
            verdict = "⚠️ UNCERTAIN"

        st.markdown("### 🔍 Result")
        st.write("**Prediction:**", inverse_label_map[prediction])
        st.write("**Verdict:**", verdict)

        st.markdown("### 📊 Confidence")
        st.progress(real_conf)
        st.write(f"REAL: {real_conf:.2f}")
        st.write(f"FAKE: {probabilities[0]:.2f}")

        st.caption(f"Non-zero features: {features.nnz}")
