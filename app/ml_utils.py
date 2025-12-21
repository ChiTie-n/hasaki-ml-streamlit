
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Setup paths to ensure src imports work if this utility is imported
# This block might be redundant if the importing page already sets paths, 
# but good for safety if used independently.
try:
    app_dir = Path(__file__).parent
    root_dir = app_dir.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
except:
    pass

from src.data_loader import (
    load_dim_product,
    load_fact_prices,
    load_fact_inventory,
    load_fact_reviews,
)
from src.features import (
    build_product_feature_table,
    clean_review_text,
    classify_price_sentiment_rule_based,
    aggregate_product_sentiment,
    get_pricing_action_candidates
)

@st.cache_data(ttl=3600)
def load_ml_data(limit=3000):
    """Load and prepare all data for ML models"""
    with st.spinner("Đang tải dữ liệu..."):
        products = load_dim_product(limit=limit)
        prices = load_fact_prices(limit=limit*3)
        inventory = load_fact_inventory(limit=limit*3)
        reviews = load_fact_reviews(limit=limit*3)
        
        # Build feature table with proper preprocessing
        df_feat, feature_cols = build_product_feature_table(
            products, prices, inventory, reviews,
            apply_log_transform=True,
            impute_missing_rating=True
        )
    return df_feat, feature_cols, reviews
