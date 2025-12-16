"""
Customer segmentation using KMeans clustering on session-level features.

Algorithms:
- KMeans

Evaluation:
- Silhouette Score
- Davies-Bouldin Index
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_OUT = Path("model_artifacts/kmeans.pkl")


def load_session_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop non-feature columns
    drop_cols = [
        "session_id",
        "first_category",
        "last_category"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)


def scale_features(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_kmeans(X_scaled: np.ndarray, k: int):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
    return model, labels


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray):
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    return sil, dbi


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 8)):
    results = []
    for k in k_range:
        model, labels = run_kmeans(X_scaled, k)
        sil, dbi = evaluate_clustering(X_scaled, labels)
        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": dbi
        })
        logger.info(
            "k=%d | silhouette=%.3f | davies_bouldin=%.3f",
            k, sil, dbi
        )
    return pd.DataFrame(results)


def main():
    # Load session-level features
    df = load_session_features("data/processed/session_features.csv")
    logger.info("Session features shape: %s", df.shape)

    # Scale
    X_scaled, scaler = scale_features(df)

    # Find optimal clusters
    metrics_df = find_optimal_k(X_scaled, range(2, 8))
    print("\nClustering evaluation:")
    print(metrics_df)

    # Choose best k (highest silhouette)
    best_k = metrics_df.sort_values("silhouette", ascending=False).iloc[0]["k"]
    best_k = int(best_k)
    logger.info("Selected optimal k = %d", best_k)

    # Train final model
    model, labels = run_kmeans(X_scaled, best_k)

    # Save model + scaler
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler},
        MODEL_OUT
    )
    logger.info("Saved KMeans model to %s", MODEL_OUT)

    # Attach labels for inspection
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out.to_csv("data/processed/session_clusters.csv", index=False)
    logger.info("Saved clustered sessions to data/processed/session_clusters.csv")


if __name__ == "__main__":
    main()