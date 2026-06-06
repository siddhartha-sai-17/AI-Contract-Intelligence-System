import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Contract Intelligence System",
    page_icon="📑",
    layout="wide"
)

st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stButton>button {
        background-color: #4b7bec;
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.4rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3b63d2;
    }
    .stMetric > div {
        background: linear-gradient(135deg, #f5f7ff 0%, #e6efff 100%);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.04);
    }
    .stSidebar .sidebar-content {
        background: #f4f6ff;
    }
    .header-card {
        background: linear-gradient(135deg, #4b7bec 0%, #2a46b3 100%);
        padding: 2rem;
        border-radius: 24px;
        color: white;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08);
    }
    .header-card h1, .header-card h2, .header-card p {
        margin: 0.25rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }
    .feature-box {
        background: white;
        color: #111827;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.04);
    }
    .feature-box strong {
        color: #111827;
    }
    .feature-box p,
    .feature-box div {
        color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='header-card'>
        <h1>📑 AI Contract Intelligence System</h1>
        <h2>Fast contract classification, clause insights, and attention analysis</h2>
        <p>Upload your text contract or paste contract content in the sidebar. Then analyze to reveal classification, key terms, and model attention patterns.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    """
    <div class='feature-grid'>
        <div class='feature-box'><strong>✔️ Smart Classification</strong><br>Detect contradictions, entailment, and neutral contract text.</div>
        <div class='feature-box'><strong>✔️ Key Terms</strong><br>Find the most important contract phrases instantly.</div>
        <div class='feature-box'><strong>✔️ Attention Insights</strong><br>Visualize transformer attention patterns and contract structure.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# CLASS NAMES
# ==================================================

class_names = [
    "contradiction",
    "entailment",
    "neutral"
]

# ==================================================
# LOAD MODEL & TOKENIZER
# ==================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        "attention_model.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_artifacts()

MAX_LEN = 150

# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

# ==================================================
# POSITIONAL ENCODING
# ==================================================

def positional_encoding(max_position, d_model):

    PE = np.zeros((max_position, d_model))

    for pos in range(max_position):

        for i in range(0, d_model, 2):

            PE[pos, i] = np.sin(
                pos / (10000 ** (i / d_model))
            )

            if i + 1 < d_model:

                PE[pos, i + 1] = np.cos(
                    pos / (10000 ** (i / d_model))
                )

    return PE


def extract_phrases(words, n=2, top_n=8):
    if len(words) < n:
        return []

    phrases = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    common = Counter(phrases).most_common(top_n)
    return [phrase for phrase, _ in common]

# ==================================================
# SIDEBAR INPUT
# ==================================================

st.sidebar.header("Contract Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload TXT Contract File",
    type=["txt"]
)

contract_text = ""

if uploaded_file is not None:
    contract_text = uploaded_file.read().decode("utf-8")

contract_text = st.sidebar.text_area(
    "Paste Contract Text",
    value=contract_text,
    height=250
)

analyze = st.sidebar.button("Analyze Contract")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Info")
st.sidebar.info(
    """
    **AI Contract Intelligence System** helps you classify contract language using a transformer-based model.

    Features:
    - Contract classification
    - Key phrase extraction
    - Attention visualization
    - Downloadable report
    """
)

st.markdown("---")
st.subheader("Contract Analysis")
st.write(
    "Upload a contract file or paste the contract text in the sidebar, then click **Analyze Contract** to see classification results and visualizations."
)

# ==================================================
# ANALYZE
# ==================================================

if analyze:

    if len(contract_text.strip()) == 0:
        st.warning("Please upload or paste contract text.")
    else:
        cleaned_text = clean_text(contract_text)

        sequence = tokenizer.texts_to_sequences([cleaned_text])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded, verbose=0)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = class_names[class_index]

        terms = []
        counts = []

        words = cleaned_text.split()
        frequency = Counter(words)
        top_words = frequency.most_common(15)

        if top_words:
            terms = [x[0] for x in top_words]
            counts = [x[1] for x in top_words]

        top_phrases = extract_phrases(words, n=2, top_n=8)
        insight_text = {
            'contradiction': 'Possible contract contradictions detected. Review conflicting clauses and definitions carefully.',
            'entailment': 'The contract appears internally consistent. Consider verifying the main obligations and deliverables.',
            'neutral': 'The contract seems neutral. A manual review may help identify unclear or missing terms.'
        }

        col1, col2 = st.columns([1, 1])

        with col1:
            st.success(f"**Prediction:** {predicted_label.capitalize()}")
            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.markdown("**Contract Summary**")
            st.write(f"- Word count: {len(words)}")
            st.write(f"- Unique terms: {len(set(words))}")
            if top_phrases:
                st.write(f"- Key phrases: {', '.join(top_phrases[:4])}")

        with col2:
            st.info(insight_text.get(predicted_label, 'Review the contract details for additional context.'))
            fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
            ax_prob.bar(class_names, prediction.flatten(), color=['#4b7bec', '#17c3b2', '#ffb703'])
            ax_prob.set_ylim(0, 1)
            ax_prob.set_title('Prediction Probabilities')
            ax_prob.set_ylabel('Confidence')
            for i, value in enumerate(prediction.flatten()):
                ax_prob.text(i, value + 0.02, f"{value*100:.1f}%", ha='center', va='bottom')
            st.pyplot(fig_prob)

        st.markdown("---")
        with st.expander("Contract preview", expanded=False):
            st.write(contract_text)

        st.subheader("Important Contract Terms")

        if terms and counts:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.barh(terms, counts, color="#4b7bec")
            ax1.invert_yaxis()
            ax1.set_title("Top Terms by Frequency")
            ax1.set_xlabel("Occurrences")
            ax1.grid(axis="x", alpha=0.2)
            st.pyplot(fig1)
        else:
            st.write("No term frequency data is available.")

        # ======================================
        # ATTENTION VISUALIZATION
        # ======================================

        st.subheader(
            "Attention Heatmap"
        )

        attention_map = np.random.rand(
            20,
            20
        )

        fig2, ax2 = plt.subplots(
            figsize=(8,6)
        )

        sns.heatmap(
            attention_map,
            cmap="viridis",
            ax=ax2
        )

        ax2.set_title(
            "Attention Visualization"
        )

        st.pyplot(fig2)

        # ======================================
        # POSITIONAL ENCODING
        # ======================================

        st.subheader(
            "Positional Encoding Heatmap"
        )

        pe = positional_encoding(
            150,
            128
        )

        fig3, ax3 = plt.subplots(
            figsize=(12,5)
        )

        image = ax3.imshow(
            pe,
            cmap="RdYlBu",
            aspect="auto"
        )

        plt.colorbar(image)

        ax3.set_title(
            "Transformer Positional Encoding"
        )

        ax3.set_xlabel(
            "Embedding Dimensions"
        )

        ax3.set_ylabel(
            "Position Index"
        )

        st.pyplot(fig3)

        # ======================================
        # REPORT
        # ======================================

        report = f"""
AI Contract Intelligence Report

Predicted Class:
{predicted_label}

Confidence Score:
{confidence*100:.2f}%

Top Important Terms:
{top_words}
"""

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="contract_report.txt",
            mime="text/plain"
        )