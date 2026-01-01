import streamlit as st
import torch
from ngram_nn.predictor import NGramNNPredictor
from LSTM.predictor import LSTMPredictor
from LSTM.model import LSTM_Model, ModelConfig
from ngram_model.NGramPredictor import NgramsPredictor
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. Page Configuration (English UI)
# ==========================================
st.set_page_config(
    page_title="Next-Word Predictor", 
    page_icon="🇫🇷",
    layout="wide"
)

st.title("🇫🇷 French Mobile Keyboard Predictor")
st.markdown("""
> This interactive demo showcases the evolution of next-word prediction models, 
> from statistical **N-Gram Count** to Deep Learning **N-Gram Neural Network** and **LSTM**.
""")

# ==========================================
# 2. Load Models
# ==========================================
@st.cache_resource
def load_models():
    
    lstm = LSTMPredictor(
        model_class=LSTM_Model, 
        checkpoint_path=r"C:\Users\Agnes\Desktop\Type Predictor\LSTM\lstm_best_model.pth", 
        device='cpu',
    )
    
    ngramnn = NGramNNPredictor(model_path=r"C:\Users\Agnes\Desktop\Type Predictor\ngram_nn\ngramnn_best_model.pth")
    
    ngram = NgramsPredictor(model_path=r"C:\Users\Agnes\Desktop\Type Predictor\ngram_model\ngram_model.pkl")
    
    return ngram, ngramnn, lstm

ngram_model, ngramnn_model, lstm_model = load_models()

# ==========================================
# 3. Sidebar: Settings & Personalization
# ==========================================
st.sidebar.header("⚙️ Configuration")

# Model Selection
model_option = st.sidebar.radio(
    "Select Model:",
    ("N-Gram (Baseline)", "N-Gram Neural Network", "LSTM with Personalization")
)

# Personalization Control (Only for LSTM)
alpha = 0.0

if "LSTM" in model_option:
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Personalization ")
    
    alpha = st.sidebar.slider("User Habit Weight (Alpha)", 0.0, 1.0, 0.4)
    
    lstm_model.alpha = alpha
    
    with st.sidebar.expander("Simulation Area"):
        st.write("Simulate a user who loves eating **'Croissant'**.")
        
        if st.button("Simulate typing 'Je veux un croissant' 20 times"):
            target_sentence = ["je", "veux", "un", "croissant"]
            for _ in range(20):
                lstm_model.update_user_history(target_sentence)
            
            st.toast("User history updated! 'Croissant' frequency boosted.", icon="🥐")
            st.rerun()
# ==========================================
# 4. Main Interface
# ==========================================

# Input Section
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("⌨️ User Input")
    user_input = st.text_input(
        "Type a French sentence fragment:", 
        value="je veux",
        placeholder="e.g., bonjour, je suis..."
    )

if user_input:
    # --- Prediction Logic ---
    start_time = time.time()
    
    if "LSTM" in model_option:
        results = lstm_model.predict_next(user_input)
        words, probs = results
        model_color = "#1E88E5" # Blue for LSTM
        
    elif "Neural Network" in model_option:
        results = ngramnn_model.predict_next(user_input)
        words = [w for w, _ in results]
        probs = [p for _, p in results]
        model_color = "#00C853" # Green for NGramNN
    
    else: #baseline N-Gram
        results = ngram_model.predict_next(user_input)
        words = [w for w, _ in results]
        probs = [p for _, p in results]
        model_color = "#FF4B4B" # Red
        
    latency = (time.time() - start_time) * 1000 # ms

    # --- Results Display ---
    st.markdown("### 💡 Top-5 Predictions")
    
    # Display buttons with French words
    if words:
        btn_cols = st.columns(5)
        
        for i, (word, prob) in enumerate(zip(words[:5], probs[:5])):
            with btn_cols[i]:
                st.button(
                        f"{word}\n({prob:.1%})", 
                        key=f"btn_{i}", 
                        use_container_width=True
                    )
    else:
        st.warning("No prediction available.")

    # --- Metrics & Charts ---
    st.divider()
    
    # 1. Latency Metric
    st.caption(f"⚡ Inference Latency: **{latency:.2f} ms**")

    # 2. Probability Chart
    st.subheader("📊 Confidence Distribution")
    if words:
        chart_data = pd.DataFrame({
            "French Word": words,
            "Probability": probs
        })
        
        st.bar_chart(
            chart_data, 
            x="French Word", 
            y="Probability",
            color=model_color
        )

# ==========================================
# 5. Footer
# ==========================================

st.markdown("---")
st.markdown("#### 📝 Technical Details")
st.markdown("""
- **Dataset**: 88k French SMS messages.
- **Preprocessing**: NLTK Tokenization, Dynamic Vocabulary (Top 15k words).
- **Architecture**: 
    - *Baseline*: 3-Gram Backoff.
    - *Deep Learning* : N-Gram Neural Network with Embeddings (Dim=128).
    - *Deep Learning*: 2-Layer LSTM with Embedding (Dim=256).
""")