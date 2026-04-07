import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import glob
import os
from transformers import AutoModel, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral",
]

# ── Model definition (must exactly match notebook Cell 5) ─────────────────────
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.checkpoint = "microsoft/deberta-v3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.basemodel = AutoModel.from_pretrained(self.checkpoint)
        self.head = nn.Sequential(
            nn.Linear(self.basemodel.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 28),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.basemodel(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Mean pooling (same as training)
        pooled = (
            (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            / attention_mask.sum(1, keepdim=True)
        )
        return self.head(pooled)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = EmotionClassifier().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.float()
    model.eval()
    return model

# ── Load thresholds from JSON (matches save_thresholds() in notebook) ──────────
@st.cache_resource
def load_thresholds():
    # Find the most recently saved threshold file
    files = sorted(glob.glob("optim_thresholds_*.json"), reverse=True)
    if not files:
        st.warning("No optim_thresholds_*.json file found. Using default threshold of 0.5.")
        return np.full(len(EMOTION_LABELS), 0.5)
    
    with open(files[0], "r") as f:
        data = json.load(f)
    
    thresholds_dict = data["thresholds"]
    return np.array([thresholds_dict[label] for label in EMOTION_LABELS])

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Emotion Detection App")
st.write("Enter a sentence to detect emotions")

model = load_model()
best_thresholds = load_thresholds()

text = st.text_area("Your text here:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        preds = (probs > best_thresholds).astype(int)
        predicted_labels = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS)) if preds[i] == 1]

        st.subheader("Predicted Emotions:")
        if predicted_labels:
            for label in predicted_labels:
                st.write(f"• {label}")
        else:
            st.write("No strong emotion detected")

        st.subheader("Confidence Scores:")
        for i, label in enumerate(EMOTION_LABELS):
            st.progress(float(probs[i]), text=f"{label}: {probs[i]:.2f}")