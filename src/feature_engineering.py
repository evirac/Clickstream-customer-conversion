"""
src/feature_engineering.py

Functions:
- create_session_features(raw_df)
- add_session_features_to_dataset(dataset_df, session_features_df)
- save_session_features(session_features_df, path)

Notes:
- This module assumes `raw_df` is the cleaned raw clickstream (columns normalized:
  session_id, order, page1_main_category, page_2_clothing_model, colour,
  location, model_photography, price, price_2, page).
- Because the raw data doesn't contain timestamps, we cannot compute time deltas.
  We compute features from order / sequence and price info.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _category_entropy(series: pd.Series) -> float:
    """Shannon entropy for a categorical Series (na ignored)."""
    counts = series.value_counts(normalize=True)
    # Shannon entropy
    ent = -(counts * np.log2(counts + 1e-12)).sum()
    return float(ent)


def create_session_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    # ---- canonical column names (after normalization) ----
    CAT_COL = 'page_1_main_category'
    MODEL_COL = 'page_2_clothing_model'

    """Create session-level features aggregated from raw clickstream rows.

    Returns a DataFrame indexed by session_id with features:
        - session_length
        - n_unique_models
        - n_unique_categories
        - category_entropy
        - mean_price, max_price, min_price, std_price
        - last_price, last_price_2
        - pct_price2_views
        - num_high_price_views
        - first_page, last_page
        - first_category, last_category
        - last_order (index/position)
    """
    df = raw_df.copy()
    # ensure correct ordering
    df = df.sort_values(['session_id', 'order'])
    
    # safe casts
    for c in ['price','price_2','order','page']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    agg_rows = []
    grouped = df.groupby('session_id', sort=False)
    for session_id, group in grouped:
        row = {}
        row['session_id'] = session_id
        row['session_length'] = int(group['order'].count())
        # unique counts
        row['n_unique_models'] = int(group[MODEL_COL].nunique(dropna=True))
        row['n_unique_categories'] = int(group[CAT_COL].nunique(dropna=True))
        row['category_entropy'] = _category_entropy(group[CAT_COL].dropna())

        # price stats
        prices = pd.to_numeric(group['price'], errors='coerce').dropna()
        if len(prices) > 0:
            row['mean_price'] = float(prices.mean())
            row['max_price'] = float(prices.max())
            row['min_price'] = float(prices.min())
            row['std_price'] = float(prices.std(ddof=0)) if len(prices) > 1 else 0.0
        else:
            row['mean_price'] = np.nan
            row['max_price'] = np.nan
            row['min_price'] = np.nan
            row['std_price'] = np.nan

        # last-click info
        last_row = group.iloc[-1]
        row['last_price'] = float(last_row['price']) if (not pd.isna(last_row['price'])) else np.nan
        row['last_price_2'] = int(last_row['price_2']) if (not pd.isna(last_row['price_2'])) else np.nan
        row['last_page'] = int(last_row['page']) if ('page' in last_row and not pd.isna(last_row['page'])) else np.nan
        # row['last_category'] = last_row.get('page1_main_category', np.nan)
        # row['last_model'] = last_row.get('page_2_clothing_model', np.nan)
        row['last_model'] = last_row.get(MODEL_COL, np.nan)
        row['last_order'] = int(last_row['order']) if (not pd.isna(last_row['order'])) else np.nan

        # first-click info
        first_row = group.iloc[0]
        # row['first_category'] = first_row.get('page1_main_category', np.nan)
        row['first_page'] = int(first_row['page']) if ('page' in first_row and not pd.isna(first_row['page'])) else np.nan

# added check for CAT_COL existence
        if CAT_COL in group.columns:
            row['first_category'] = group.iloc[0][CAT_COL]
            row['last_category'] = group.iloc[-1][CAT_COL]
        else:
            row['first_category'] = np.nan
            row['last_category'] = np.nan

        # price2 related
        price2_views = group['price_2'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        row['num_high_price_views'] = int((price2_views == 1).sum())
        row['pct_price2_views'] = float((price2_views == 1).mean()) if len(price2_views) > 0 else 0.0

        # category transition features (first->last same?)
        try:
            row['first_eq_last_category'] = int(row['first_category'] == row['last_category'])
        except Exception:
            row['first_eq_last_category'] = 0

        # diversity ratio
        row['models_per_click'] = row['n_unique_models'] / row['session_length'] if row['session_length'] > 0 else 0.0

        agg_rows.append(row)

    features_df = pd.DataFrame(agg_rows)
    # set session_id as numeric if possible
    try:
        features_df['session_id'] = pd.to_numeric(features_df['session_id'], errors='coerce').astype(int)
    except Exception:
        pass

    # order columns for readability
    cols = [
        'session_id','session_length','n_unique_models','n_unique_categories','category_entropy',
        'mean_price','max_price','min_price','std_price','last_price','last_price_2','num_high_price_views','pct_price2_views',
        'first_page','last_page','first_category','last_category','first_eq_last_category','models_per_click','last_order'
    ]
    # keep only existing columns
    cols = [c for c in cols if c in features_df.columns]
    features_df = features_df[cols]
    logger.info("Created session features for %d sessions", features_df.shape[0])
    return features_df


def add_session_features_to_dataset(dataset_df: pd.DataFrame, session_features_df: pd.DataFrame) -> pd.DataFrame:
    """Merge session-level features onto a click-level dataset by session_id."""
    out = dataset_df.merge(session_features_df, on='session_id', how='left')
    return out


def save_session_features(features_df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(p, index=False)
    logger.info("Saved session features to %s", p)