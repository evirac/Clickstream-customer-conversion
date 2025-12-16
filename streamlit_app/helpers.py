import sys
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.feature_engineering import create_session_features

MODEL_DIR = ROOT_DIR / "model_artifacts"

# ------------------------------------------------------------------
# CLUSTER FEATURES (MUST MATCH TRAINING)
# ------------------------------------------------------------------
CLUSTER_FEATURES = [
    "session_length",
    "n_unique_models",
    "n_unique_categories",
    "category_entropy",
    "mean_price",
    "max_price",
    "min_price",
    "std_price",
    "last_price",
    "last_price_2",
    "num_high_price_views",
    "pct_price2_views",
    "first_page",
    "last_page",
    "first_eq_last_category",
    "models_per_click",
    "last_order",
]

# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load(MODEL_DIR / "classification.pkl")
    reg = joblib.load(MODEL_DIR / "regression.pkl")
    km_bundle = joblib.load(MODEL_DIR / "kmeans.pkl")
    return clf, reg, km_bundle["model"], km_bundle["scaler"]


# ------------------------------------------------------------------
# Raw CSV loading + cleaning
# ------------------------------------------------------------------
def load_and_clean_raw_clickstream(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=";")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    df = df.rename(columns={
        "session_id": "session_id",
        "page_1_main_category": "page_1_main_category",
        "page_2_clothing_model": "page_2_clothing_model",
        "price_2": "price_2",
    })

    return df


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
def prepare_session_level(raw_df: pd.DataFrame) -> pd.DataFrame:
    return create_session_features(raw_df)


# ------------------------------------------------------------------
# Prediction wrapper (FIXED)
# ------------------------------------------------------------------
def predict_all(session_df, clf, reg, kmeans, scaler):
    df = session_df.copy()

    # ---- Classification ----
    X_clf = df[clf.named_steps["preproc"].feature_names_in_]
    df["conversion_probability"] = clf.predict_proba(X_clf)[:, 1]
    df["will_convert"] = clf.predict(X_clf)

    # ---- Regression ----
    X_reg = df[reg.named_steps["preprocess"].feature_names_in_]
    df["predicted_revenue"] = reg.predict(X_reg)

    # ---- Clustering (FIX: strict feature selection) ----
    X_cluster = df[CLUSTER_FEATURES].copy()
    X_cluster_scaled = scaler.transform(X_cluster)
    df["cluster"] = kmeans.predict(X_cluster_scaled)

    return df
