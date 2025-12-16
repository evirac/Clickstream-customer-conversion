import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from helpers import (
    load_models,
    load_and_clean_raw_clickstream,
    prepare_session_level,
    predict_all,
)

# ------------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Clickstream Customer Conversion",
    layout="wide"
)

st.title("🛒 Clickstream Customer Conversion Analysis")

st.markdown("""
This app predicts:
- **Conversion probability**
- **Estimated revenue**
- **Customer behavior cluster**

from **raw clickstream data** (`e-shop clothing 2008.csv`).
""")

# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------
with st.spinner("Loading models..."):
    clf, reg, kmeans, scaler = load_models()

st.success("Models loaded successfully")

# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload raw clickstream CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # ✅ Clean raw data FIRST
    raw_df = load_and_clean_raw_clickstream(uploaded_file)

    st.subheader("Raw Data Preview (cleaned)")
    st.dataframe(raw_df.head())

    with st.spinner("Creating session-level features..."):
        session_df = prepare_session_level(raw_df)

    st.subheader("Session-Level Features")
    st.dataframe(session_df.head())

    with st.spinner("Running predictions..."):
        results_df = predict_all(session_df, clf, reg, kmeans, scaler)

    st.subheader("Predictions")
    st.dataframe(
        results_df[
            [
                "session_id",
                "conversion_probability",
                "will_convert",
                "predicted_revenue",
                "cluster"
            ]
        ].head(20)
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Conversion Probability",
            f"{results_df['conversion_probability'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Avg Predicted Revenue",
            f"{results_df['predicted_revenue'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Most Common Cluster",
            int(results_df["cluster"].mode()[0])
        )

    st.subheader("Cluster Distribution")
    st.bar_chart(results_df["cluster"].value_counts())

else:
    st.info("Please upload `e-shop clothing 2008.csv` to begin.")
