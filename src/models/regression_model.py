"""
Train a regression model to predict session-level revenue.

Models:
- Linear Regression (baseline)
- Gradient Boosting Regressor

Metrics:
- MAE
- RMSE
- R^2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_OUT = Path("model_artifacts/regression.pkl")


def build_regression_pipeline(numeric_features, categorical_features, model_type="gbr"):
    """Build a preprocessing + regression pipeline."""

    transformers = []

    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))

    if categorical_features:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers.append(("cat", ohe, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    if model_type == "linear":
        model = LinearRegression()
    else:
        model = GradientBoostingRegressor(random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_and_evaluate(train_csv: str, model_type="gbr"):
    df = pd.read_csv(train_csv)

    # One row per session (last click)
    df = (
        df.sort_values(["session_id", "order"])
          .drop_duplicates("session_id", keep="last")
          .reset_index(drop=True)
    )

    if "revenue" not in df.columns:
        raise ValueError("Target column 'revenue' not found")

    # Drop missing revenue
    df = df.dropna(subset=["revenue"]).copy()
    logger.info("Training rows after dropna revenue: %d", df.shape[0])

    # Feature selection
    numeric_features = [
        "session_length",
        "n_unique_models",
        "n_unique_categories",
        "category_entropy",
        "mean_price",
        "max_price",
        "min_price",
        "std_price",
        "num_high_price_views",
        "pct_price2_views",
        "models_per_click",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]

    categorical_features = [
        "first_category",
        "last_category",
        "first_page",
        "last_page",
    ]
    categorical_features = [c for c in categorical_features if c in df.columns]

    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    X = df[numeric_features + categorical_features]
    y = df["revenue"].astype(float)
    # y = np.log1p(df["revenue"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_regression_pipeline(
        numeric_features,
        categorical_features,
        model_type=model_type
    )

    # Hyperparameter tuning only for GBR
    if model_type == "gbr":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3]
        }
        gs = GridSearchCV(
            pipeline,
            param_grid,
            scoring="neg_root_mean_squared_error",
            cv=4,
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        logger.info("Best params: %s", gs.best_params_)
    else:
        model = pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_val, y_pred)

    print("\nRegression Metrics (Validation)")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    logger.info("Saved regression model to %s", MODEL_OUT)

    return model, {"MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    train_and_evaluate(
        "data/processed/train_final_with_features.csv",
        model_type="gbr"
    )