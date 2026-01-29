import streamlit as st
import pickle

# =========================
# LOAD MODEL & VECTORIZER
# =========================
@st.cache_resource
def load_artifacts():
    with open("veritasai_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("veritasai_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    inverse_label_map = {v: k for k, v in label_map.items()}
    return model, vectorizer, inverse_label_map

model, vectorizer, inverse_label_map = load_artifacts()

# =========================
# UI
# =========================
st.set_page_config(
    page_title="VeritasAI – Fake News Detection",
    layout="centered"
)

st.title("📰 VeritasAI")
st.subheader("AI-based Fake News Detection System")

st.write(
    "Paste a full news article below. "
    "VeritasAI analyzes writing patterns and provides a likelihood estimate."
)

# =========================
# INPUT
# =========================
news_text = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste the full news article here..."
)

# =========================
# PREDICTION
# =========================
if st.button("Check Authenticity"):

    if len(news_text.strip()) < 100:
        st.warning("Please enter a longer news article for better accuracy.")
    else:
        tfidf_vector = vectorizer.transform([news_text])
        non_zero = tfidf_vector.nnz

        prediction = model.predict(tfidf_vector)
        probabilities = model.predict_proba(tfidf_vector)

        real_conf = probabilities[0][1]

        # Decision logic (Layer 1)
        if real_conf >= 0.7:
            verdict = "Likely REAL"
            st.success(verdict)
        elif real_conf <= 0.3:
            verdict = "Likely FAKE"
            st.error(verdict)
        else:
            verdict = "UNCERTAIN"
            st.warning(verdict)

        # Confidence display
        st.write(f"**Confidence (REAL):** {real_conf * 100:.2f}%")

        # Out-of-domain warning
        if non_zero < 40:
            st.info(
                "⚠️ The text may be outside the training domain. "
                "Prediction confidence may be reduced."
            )
