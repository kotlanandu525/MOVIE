import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    simple_rnn = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "simple_rnn_model.h5")
    )

    lstm = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "lstm_model.h5")
    )

    gru = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "gru_model.h5")
    )

    return simple_rnn, lstm, gru


@st.cache_resource
def load_tokenizer():

    with open(
        os.path.join(BASE_DIR, "tokenizer.pkl"),
        "rb"
    ) as f:
        tokenizer = pickle.load(f)

    return tokenizer


simple_rnn_model, lstm_model, gru_model = load_models()
tokenizer = load_tokenizer()

MAX_LEN = 250

# -----------------------------
# Header
# -----------------------------
st.title("🎬 Movie Review Sentiment Analysis System")

st.markdown(
    "### Deep Learning Based Sentiment Classification"
)

st.divider()

# -----------------------------
# Model Selection
# -----------------------------
selected_model = st.radio(
    "Choose Model",
    ["SimpleRNN", "LSTM", "GRU"],
    horizontal=True
)

# -----------------------------
# Input
# -----------------------------
review = st.text_area(
    "Enter your movie review here...",
    height=180
)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_review(model, text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    prob = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    sentiment = (
        "Positive"
        if prob >= 0.5
        else "Negative"
    )

    confidence = (
        prob * 100
        if prob >= 0.5
        else (1 - prob) * 100
    )

    return sentiment, confidence, prob


# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    model_dict = {
        "SimpleRNN": simple_rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    model = model_dict[selected_model]

    sentiment, confidence, prob = predict_review(
        model,
        review
    )

    st.subheader("Prediction Result")

    if sentiment == "Positive":
        st.success(f"Sentiment: {sentiment}")
    else:
        st.error(f"Sentiment: {sentiment}")

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # -----------------------------
    # Probability Chart
    # -----------------------------
    positive_prob = prob * 100
    negative_prob = (1 - prob) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Positive"],
            y=[positive_prob],
            name="Positive"
        )
    )

    fig.add_trace(
        go.Bar(
            x=["Negative"],
            y=[negative_prob],
            name="Negative"
        )
    )

    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability (%)",
        barmode="group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------
    # Confidence Gauge
    # -----------------------------
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Confidence"},
            gauge={
                "axis": {
                    "range": [0, 100]
                }
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

# ====================================================
# Compare All Models
# ====================================================

st.divider()

st.header("Compare All Models")

if st.button("Compare Models"):

    if review.strip() == "":
        st.warning("Enter a review first.")
        st.stop()

    results = []

    for name, model in {
        "SimpleRNN": simple_rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }.items():

        sentiment, confidence, prob = predict_review(
            model,
            review
        )

        results.append({
            "Model": name,
            "Sentiment": sentiment,
            "Confidence (%)": round(confidence, 2)
        })

    st.dataframe(
        results,
        use_container_width=True
    )

    comparison_fig = go.Figure()

    comparison_fig.add_trace(
        go.Bar(
            x=[r["Model"] for r in results],
            y=[r["Confidence (%)"] for r in results]
        )
    )

    comparison_fig.update_layout(
        title="Model Confidence Comparison",
        yaxis_title="Confidence (%)"
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )
