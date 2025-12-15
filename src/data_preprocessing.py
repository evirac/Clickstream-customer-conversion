"""
src/data_preprocessing.py

Responsible for:
- Loading raw e-shop file (semicolon separated)
- Loading provided train/test CSVs (comma separated)
- Normalizing column names
- Basic cleaning (missing values placeholder handling)
- Saving standardized processed CSVs to data/processed/

Usage:
    python -m src.data_preprocessing --raw data/raw/e-shop_clothing_2008.csv --train data/processed/train.csv --test data/processed/test.csv --out_dir data/processed
"""

from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case and remove problematic characters."""
    df = df.copy()
    new_cols = []
    for c in df.columns:
        c2 = (
            c.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
            .replace("-", "_")
        )
        # unify multiple underscores
        c2 = "_".join([p for p in c2.split("_") if p != ""])
        new_cols.append(c2)
    df.columns = new_cols
    return df


def read_raw(raw_path: Path) -> pd.DataFrame:
    """Read the raw e-shop CSV which uses semicolon separators."""
    logger.info("Reading raw file: %s", raw_path)
    df = pd.read_csv(raw_path, sep=";", engine="python", dtype=str)
    df = normalize_columns(df)
    # convert numeric columns when possible
    for c in ["year", "month", "day", "order", "price", "price_2", "page"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # session_id sometimes has whitespace; strip and convert if numeric; keep as string if not
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str).str.strip()
        # attempt to convert purely numeric
        if df["session_id"].str.isnumeric().all():
            df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce").astype(int)
    return df


def read_commas(path: Path) -> pd.DataFrame:
    """Read a standard comma-separated CSV (train/test provided by the institute)."""
    logger.info("Reading comma-separated file: %s", path)
    df = pd.read_csv(path, dtype=str)
    df = normalize_columns(df)
    # numeric conversion: safe conversion for columns that should be numeric
    for c in ["year", "month", "day", "order", "country", "price", "price_2", "page"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str).str.strip()
        if df["session_id"].str.isnumeric().all():
            df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce").astype(int)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning:
    - Strip strings
    - Fill trivial missing values for categorical columns with 'unknown'
    - Leave numeric NaNs for imputation later
    """
    df = df.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None})
    # convert empty strings to NaN
    df = df.replace({"": np.nan})
    # For categorical object columns, we can fillna('unknown') to simplify EDA. We'll keep numeric columns NaN.
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("unknown")
    return df


def save_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows, %d cols)", out_path, df.shape[0], df.shape[1])


def main(raw_path: str, train_path: str, test_path: str, out_dir: str):
    raw = read_raw(Path(raw_path))
    train = read_commas(Path(train_path))
    test = read_commas(Path(test_path))

    # basic cleaning
    raw_clean = basic_cleaning(raw)
    train_clean = basic_cleaning(train)
    test_clean = basic_cleaning(test)

    # Save files
    out_dir = Path(out_dir)
    save_df(raw_clean, out_dir / "e-shop_clothing_2008_raw_cleaned.csv")
    save_df(train_clean, out_dir / "train_data.csv")
    save_df(test_clean, out_dir / "test_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Raw e-shop CSV (semicolon separated)")
    parser.add_argument("--train", required=True, help="Train CSV (comma separated)")
    parser.add_argument("--test", required=True, help="Test CSV (comma separated)")
    parser.add_argument("--out_dir", default="data/processed", help="Where to save processed CSVs")
    args = parser.parse_args()
    main(args.raw, args.train, args.test, args.out_dir)
