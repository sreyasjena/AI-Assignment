
import streamlit as st
import joblib
from transformers import pipeline
import torch

# Load baseline model
model = joblib.load("artifacts/baseline_tfidf_logreg.joblib")
vectorizer = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
feature_names = vectorizer.get_feature_names_out()

def explain_instance_fast(text: str, num_features: int = 6):
    X_vec = vectorizer.transform([text])
    contrib = X_vec.multiply(clf.coef_[0]).toarray()[0]
    top_idx = contrib.argsort()[-num_features:][::-1]
    return [(feature_names[i], contrib[i]) for i in top_idx], model.predict_proba([text])[0]

# PyTorch-only generator
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=device, framework="pt")

def generate_justification(text: str):
    probs = model.predict_proba([text])[0]
    pred = model.predict([text])[0]
    label = "fake" if pred == 1 else "real"
    feats, _ = explain_instance_fast(text, 6)
    feat_lines = "\\n".join([f"- {f} ({w:+.3f})" for f,w in feats])
    prompt = f"Article: {text}\\nPrediction: {label} (conf {max(probs):.2f})\\nFeatures:\\n{feat_lines}\\nWrite a short justification."
    out = generator(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"]
    return label, out, feats, probs

# Streamlit UI
st.title("📰 Fake News Detector + Justification")
text = st.text_area("Paste a news article here:")

if st.button("Check") and text.strip():
    label, justification, feats, probs = generate_justification(text)
    st.write(f"Prediction: **{label.upper()}** (Confidence: {max(probs):.2f})")
    st.subheader("Justification")
    st.write(justification)
    st.subheader("Top Features")
    for f,w in feats:
        st.write(f"- {f} ({w:+.3f})")
