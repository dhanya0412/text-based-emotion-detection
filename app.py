import streamlit as st
import torch
import numpy as np

# Title
st.title("Emotion Detection App")
st.write("Enter a sentence to detect emotions")

# Input box
text = st.text_area("Your text here:")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        model.eval()

        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        preds = (probs > best_thresholds).astype(int)

        predicted_labels = [emotion_labels[i] for i in range(len(emotion_labels)) if preds[i] == 1]

        # Display results
        st.subheader("Predicted Emotions:")
        if predicted_labels:
            for label in predicted_labels:
                st.write(f"• {label}")
        else:
            st.write("No strong emotion detected")

        # Show confidence
        st.subheader("Confidence Scores:")
        for i, label in enumerate(emotion_labels):
            st.write(f"{label}: {probs[i]:.2f}")