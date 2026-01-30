import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load datasets
old = pd.read_csv("data/news.csv", low_memory=False)
latest = pd.read_csv("data/latest_news.csv")

data = pd.concat([old, latest], ignore_index=True)

data = data[["title", "text", "label"]]
data['label'] = data['label'].astype(str).str.upper().str.strip()

label_map = {"FAKE": 0, "REAL": 1}
data["label"] = data["label"].map(label_map)

before = data.shape[0]

data = data.dropna(subset=['title', 'text', 'label'])

after = data.shape[0]
print(f"Removed {before - after} invalid rows")



data["content"] = data["title"] + " " + data["text"]
data = data.drop_duplicates(subset=["content"])
data = data.sample(frac=1, random_state=42)

print(data['label'].value_counts())


X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,1),
    min_df=5,
    max_df=0.8
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    C=0.3,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

print("Train:", accuracy_score(y_train, model.predict(X_train_vec)))
print("Test:", accuracy_score(y_test, model.predict(X_test_vec)))

# save
with open("models/veritasai_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/veritasai_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)
