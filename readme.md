# Clickstream Customer Conversion Analysis

This repository implements the Clickstream Customer Conversion project:
- Raw dataset: `data/raw/e-shop_clothing_2008.csv`
- Train/test datasets: `data/processed/train_data.csv`, `data/processed/test_data.csv`

Project goals:
- EDA on raw clickstream data → derive targets & features.
- Train classification/regression/clustering models.
- Deploy a Streamlit app.

Files in this initial commit:
- `src/data_preprocessing.py` : load + normalize raw and processed CSVs
- `src/target_creation.py` : draft for creating conversion & revenue labels
- `notebooks/01_raw_eda.ipynb` : EDA following the EDA Guide.
- `requirements.txt`, `.gitignore`, `tests/`

