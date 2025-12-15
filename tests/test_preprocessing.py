import pandas as pd
from src import data_preprocessing as dp
from pathlib import Path

def test_normalize_columns():
    df = pd.DataFrame({' A (x) ': [1], 'B-C': [2]})
    df2 = dp.normalize_columns(df)
    assert 'a_x' in df2.columns or 'a_x' in "".join(df2.columns)  # flexible check

def test_read_raw_and_commas(tmp_path):
    # Create tiny sample files
    raw_file = tmp_path / "raw.csv"
    raw_file.write_text("year;month;day;order;session_id;price;price_2;page\n2008;4;1;1;1;22;1;1\n")
    train_file = tmp_path / "train.csv"
    train_file.write_text("year,month,day,order,session_id,price,price_2,page\n2008,4,1,1,1,22,1,1\n")
    raw = dp.read_raw(raw_file)
    train = dp.read_commas(train_file)
    assert 'session_id' in raw.columns
    assert 'session_id' in train.columns
