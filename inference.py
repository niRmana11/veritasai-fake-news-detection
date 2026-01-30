import pickle
import re

# ---------------------------
# Load trained artifacts
# ---------------------------
with open("models/veritasai_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/veritasai_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

inverse_label_map = {v: k for k, v in label_map.items()}

print("VeritasAI loaded successfully\n")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Function for prediction
# ---------------------------
def predict_news(title: str, text: str):
    content = title + " " + text
    features = vectorizer.transform([content])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    real_conf = probabilities[1]





    if real_conf > 0.7:
        verdict = "Likely REAL"
    elif real_conf < 0.3:
        verdict = "Likely FAKE"
    else:
        verdict = "UNCERTAIN"


    if features.nnz < 30:
        verdict = "OUT OF DOMAIN / INSUFFICIENT DATA"

    return {
        "prediction": inverse_label_map[prediction],
        "confidence": {
            "FAKE": round(probabilities[0], 3),
            "REAL": round(probabilities[1], 3)
        },
        "verdict": verdict,
        "non_zero_features": features.nnz
    }

# ---------------------------
# Example test
# ---------------------------
if __name__ == "__main__":
    title = "NASA collects new rock samples from Mars"
    text = (
        "NASA said its Perseverance rover successfully collected new rock "
        "samples from the Martian surface. Scientists believe the samples "
        "could provide insight into the planet’s geological history."
    )

    result = predict_news(title, text)

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Verdict:", result["verdict"])
    print("Non-zero features:", result["non_zero_features"])
