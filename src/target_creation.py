"""
Create session-level targets:
- classification: converted (0/1)
- regression: revenue (float, last_price)

This module maps session aggregates to train/test by session_id.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_session_aggregates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create session-level aggregates needed for target derivation and features."""
    # Assumes raw_df has 'session_id', 'order', 'page', 'price', 'price_2'
    raw_df = raw_df.copy()
    # Ensure order is numeric for proper sorting
    raw_df = raw_df.sort_values(["session_id", "order"])
    agg = (
        raw_df.groupby("session_id", as_index=False)
        .agg(
            session_length=("order", "count"),
            first_order=("order", "min"),
            last_order=("order", "max"),
            first_page=("page", "first"),
            last_page=("page", "last"),
            last_price=("price", "last"),
            last_price_2=("price_2", "last"),
        )
    )
    return agg


def heuristic_conversion_label(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Simple heuristic:
       converted = 1 if last_price_2 == 1 OR last_page >= 4
       This is a starting point. We'll refine after inspecting raw behaviors.
    """
    df = agg_df.copy()
    df["converted"] = ((df["last_price_2"] == 1) | (df["last_page"] >= 4)).astype(int)
    # regression target: last_price (could be NaN if missing)
    df["revenue"] = df["last_price"].astype(float)
    return df


def map_to_dataset(dataset_df: pd.DataFrame, session_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Merge session-level targets onto the dataset by session_id.
    Returns dataset augmented with 'converted' and 'revenue' (session-level).
    """
    out = dataset_df.merge(session_targets[["session_id", "converted", "revenue"]], on="session_id", how="left")
    return out

if __name__ == "__main__":
    raw = pd.read_csv("data/processed/e-shop_clothing_2008_cleaned.csv")
    train = pd.read_csv("data/processed/train_data.csv")
    test = pd.read_csv("data/processed/test_data.csv")

    # 1. Create session-level aggregates
    agg = create_session_aggregates(raw)

    # 2. Create targets
    targets = heuristic_conversion_label(agg)

    # 3. Merge targets into train/test
    train_labeled = map_to_dataset(train, targets)
    test_labeled  = map_to_dataset(test, targets)

    train_labeled.to_csv("data/processed/train_with_targets.csv", index=False)
    test_labeled.to_csv("data/processed/test_with_targets.csv", index=False)

    logger.info("Saved train_with_targets.csv and test_with_targets.csv")

