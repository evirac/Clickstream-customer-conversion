"""
src/models/classification_model.py

Train a classification model to predict `converted` (0/1) using session-level features.

Expectations:
- Input CSV: data/processed/train_final_with_features.csv  (click-level rows with session features merged)
- We will build a session-level dataset by taking one row per session (last click row per session is fine)
- Save model artifact to model_artifacts/classification.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_OUT = Path("model_artifacts/classification.pkl")
ENC_OUT = Path("model_artifacts/encoders_scalers.pkl")


def build_model_pipeline(numeric_features, categorical_features, use_scaler=True, model_type='rf'):
    """Return sklearn pipeline (ColumnTransformer + estimator)."""
    transformers = []

    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))

    if categorical_features:
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        transformers.append(("cat", ohe, categorical_features))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )

    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )

    pipeline = Pipeline([
        ('preproc', preproc),
        ('clf', clf)
    ])
    return pipeline


def train_and_evaluate(train_csv: str, features: list = None, target_col: str = 'converted', model_type='rf'):
    df = pd.read_csv(train_csv)
    # create session-level dataset by taking last row per session (since we stored session-level features merged on each click row)
    df = df.sort_values(['session_id','order']).drop_duplicates('session_id', keep='last').reset_index(drop=True)
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in {train_csv}")
    # drop rows without target
    df = df[~df[target_col].isna()].copy()
    logger.info("Training rows after dropna target: %d", df.shape[0])

    # If features not provided, use sensible defaults
    if features is None:
        # numeric session features
        features = [
            'session_length','n_unique_models','n_unique_categories','category_entropy',
            'mean_price','max_price','min_price','std_price','num_high_price_views','pct_price2_views','models_per_click','last_price'
        ]
        # keep only those that exist
        features = [f for f in features if f in df.columns]

    numeric_features = [c for c in features if df[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
    categorical_features = ['first_category','last_category','first_page','last_page']
    categorical_features = [c for c in categorical_features if c in df.columns]

    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    X = df[numeric_features + categorical_features]
    y = df[target_col].astype(int)

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_model_pipeline(numeric_features, categorical_features, model_type=model_type)

    # quick grid for RF depth / LR C
    if model_type == 'rf':
        param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10, 20]}
    else:
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    logger.info("Best params: %s", gs.best_params_)
    best = gs.best_estimator_
    y_pred = best.predict(X_val)
    y_proba = best.predict_proba(X_val)[:,1] if hasattr(best, "predict_proba") else None

    print("Classification report (validation):")
    print(classification_report(y_val, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", roc_auc_score(y_val, y_proba))

    # save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_OUT)
    logger.info("Saved model to %s", MODEL_OUT)
    return best, gs.best_params_


if __name__ == "__main__":
    # test usage
    model, best = train_and_evaluate("data/processed/train_final_with_features.csv", model_type='rf')
