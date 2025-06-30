import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import numpy as np

# --- Custom CSS for modern light glassmorphism UI (centered text, no icon) ---
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        background: #fff !important;
        font-family: 'Roboto', sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #e3f0fc 0%, #f8fafc 100%);
        color: #1a232f;
        padding: 1.5rem 0 1rem 0;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 1px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0px;
    }
    .glass-card {
        background: rgba(240, 247, 255, 0.85);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border: 1px solid rgba(30,41,59,0.08);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 2rem;
        color: #1a232f;
        transition: box-shadow 0.3s;
        text-align: center;
    }
    .glass-card:hover {
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
    }
    .footer {
        text-align: center;
        color: #6c7a89;
        font-size: 1rem;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2E86C1 0%, #117A65 100%);
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        font-size: 1.1rem;
        transition: box-shadow 0.2s, background 0.2s;
        box-shadow: 0 2px 8px rgba(46,134,193,0.10);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #117A65 0%, #2E86C1 100%);
        box-shadow: 0 4px 16px rgba(46,134,193,0.15);
    }
    .custom-progress {
        width: 100%;
        height: 22px;
        background: #e3f0fc;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0 0.5rem 0;
        box-shadow: 0 2px 8px rgba(46,134,193,0.08);
    }
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #2E86C1 0%, #117A65 100%);
        border-radius: 12px 0 0 12px;
        transition: width 0.7s cubic-bezier(.4,2.3,.3,1);
        text-align: right;
        color: #fff;
        font-weight: 600;
        font-size: 1rem;
        padding-right: 12px;
        line-height: 22px;
    }
    .emoji {
        font-size: 2.1rem;
        vertical-align: middle;
        margin-right: 0.5rem;
    }
    .prediction-label {
        font-size: 1.6rem;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        text-align: center;
        color: #1a232f;
    }
    .confidence-label {
        font-size: 1.1rem;
        color: #6c7a89;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .class-prob {
        font-size: 1.05rem;
        color: #6c7a89;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .alert {
        font-size: 1.15rem;
        text-align: center;
        margin-top: 1rem;
        border-radius: 8px;
        padding: 0.7rem 0.5rem;
        font-weight: 600;
    }
    .alert-pneumonia {
        background: rgba(192,57,43,0.08);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    .alert-normal {
        background: rgba(26,188,156,0.08);
        color: #1abc9c;
        border: 1px solid #1abc9c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header without icon
st.markdown(
    """
    <div class='main-header'>
        <span style='font-size:2.2rem; vertical-align: middle;'>Pneumonia Detection from Chest X-ray</span>
        <div style='font-size:1.1rem; color:#6c7a89; margin-top:0.5rem; text-align:center;'>Powered by Vision Transformer (ViT)</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align:center; font-size:1.15rem; color:#6c7a89;'>🩺 Upload a chest X-ray image to predict if it shows signs of pneumonia.</div>",
    unsafe_allow_html=True
)

# Set up model and processor
@st.cache_resource

def load_model_and_processor():
    model_name = 'google/vit-base-patch16-224'
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    # Load trained weights
    model.load_state_dict(torch.load(os.path.join('Models', 'vit_chest_model.pkl'), map_location=torch.device('cpu')))
    model.eval()
    return model, processor

model, processor = load_model_and_processor()
class_names = ['NORMAL', 'PNEUMONIA']

@torch.no_grad()
def predict_image(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence, probs

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image in JPG or PNG format."
)

if uploaded_file is not None:
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(Image.open(uploaded_file).convert("RGB"), caption="Uploaded X-ray", use_container_width=True)
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='prediction-label'>🔎 Prediction Result</div>", unsafe_allow_html=True)
        with st.spinner('Predicting...'):
            pred_class, confidence, probs = predict_image(Image.open(uploaded_file).convert("RGB"), model, processor)
        emoji = "⚠️" if pred_class == 1 else "✅"
        st.markdown(f"<div class='prediction-label'>{emoji} {class_names[pred_class]}</div>", unsafe_allow_html=True)
        # Custom animated progress bar
        st.markdown(f"""
        <div class='confidence-label'>Confidence</div>
        <div class='custom-progress'>
            <div class='custom-progress-bar' style='width: {confidence*100:.1f}%;'>
                {confidence*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='class-prob'>Class probabilities:</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'><span style='color:#1abc9c;'>NORMAL</span>: {probs[0]*100:.2f}% &nbsp; | &nbsp; <span style='color:#e74c3c;'>PNEUMONIA</span>: {probs[1]*100:.2f}%</div>", unsafe_allow_html=True)
        if pred_class == 1:
            st.markdown("<div class='alert alert-pneumonia'>⚠️ <b>Signs of Pneumonia detected.</b> Please consult a medical professional.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert alert-normal'>✔️ <b>No signs of Pneumonia detected.</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer with social links
st.markdown(
    """
    <div class='footer'>
        &copy; 2024 Pneumonia Detection | Made with ❤️ using Streamlit & ViT &nbsp;·&nbsp;
        <a href='https://github.com/' style='color:#2E86C1; text-decoration:none;' target='_blank'>GitHub</a> |
        <a href='mailto:your@email.com' style='color:#2E86C1; text-decoration:none;' target='_blank'>Contact</a>
    </div>
    """,
    unsafe_allow_html=True
) 