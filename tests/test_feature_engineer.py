import pandas as pd
from pathlib import Path

import sys
import os
project_root = os.path.abspath("..")
sys.path.append(project_root)

from src.feature_engineering import create_session_features
def test_create_session_features_simple(tmp_path):
    # build a tiny raw click log
    data = [
        {'session_id': 1, 'order': 1, 'page_1_main_category': '1', 'page_2_clothing_model': 'A1', 'price': 10, 'price_2': 1, 'page': 1},
        {'session_id': 1, 'order': 2, 'page_1_main_category': '1', 'page_2_clothing_model': 'A2', 'price': 20, 'price_2': 1, 'page': 2},
        {'session_id': 2, 'order': 1, 'page_1_main_category': '2', 'page_2_clothing_model': 'B1', 'price': 5, 'price_2': 2, 'page': 1},
    ]
    raw = pd.DataFrame(data)
    feats = create_session_features(raw)
    assert feats.shape[0] == 2
    # session_length check
    s1 = feats[feats['session_id']==1].iloc[0]
    assert s1['session_length'] == 2
    assert s1['num_high_price_views'] == 2
