# src/features.py
"""
Feature Engineering module for ML Models.
Handles proper preprocessing, missing value treatment, and feature transformations.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def build_product_feature_table(
    products: pd.DataFrame,
    prices: pd.DataFrame,
    inventory: pd.DataFrame,
    reviews: pd.DataFrame,
    apply_log_transform: bool = True,
    impute_missing_rating: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build feature table for product clustering with proper preprocessing.
    
    Args:
        products: dim_product DataFrame
        prices: fact_prices DataFrame
        inventory: fact_inventory DataFrame
        reviews: fact_reviews DataFrame
        apply_log_transform: Apply log1p to skewed features (recommended)
        impute_missing_rating: Impute missing ratings with category median
        
    Returns:
        df: Feature DataFrame with product info + features
        feature_cols: List of feature column names for clustering
    """
    
    # --- 1. Price & Sales aggregation ---
    price_agg = (
        prices
        .groupby("product_id", as_index=False)
        .agg(
            avg_final_price=("final_price", "mean"),
            median_final_price=("final_price", "median"),
            avg_discount_percent=("discount_percent", "mean"),
            max_discount_percent=("discount_percent", "max"),
            avg_bought=("bought", "mean"),  # May be cumulative - handle with caution
            price_std=("final_price", "std"),  # Price volatility
        )
    )
    # Fill std NaN (single observation) with 0
    price_agg["price_std"] = price_agg["price_std"].fillna(0)
    
    # --- 2. Review aggregation with proper handling ---
    review_agg = (
        reviews
        .groupby("product_id", as_index=False)
        .agg(
            rating_mean=("rating_star", "mean"),
            rating_std=("rating_star", "std"),
            review_count=("review_id", "count"),
        )
    )
    review_agg["rating_std"] = review_agg["rating_std"].fillna(0)
    
    # --- 3. Inventory aggregation ---
    inv_agg = (
        inventory
        .groupby("product_id", as_index=False)
        .agg(
            avg_stock_available=("stock_available", "mean"),
            stockout_rate=("stock_available", lambda x: (x == 0).mean()),
            stock_std=("stock_available", "std"),
        )
    )
    inv_agg["stock_std"] = inv_agg["stock_std"].fillna(0)
    
    # --- 4. Merge all ---
    df = (
        products[["product_id", "product_name", "brand_name", "category_id"]]
        .merge(price_agg, on="product_id", how="left")
        .merge(review_agg, on="product_id", how="left")
        .merge(inv_agg, on="product_id", how="left")
    )
    
    # --- 4.1 Data Quality Fix: Some brands have prices 100x too high ---
    # Add brand names to this list if they have the same issue
    BRANDS_WITH_PRICE_ERROR = ['lamthaocosmetics']  # Thêm brand vào đây nếu cần
    
    if 'brand_name' in df.columns:
        for brand in BRANDS_WITH_PRICE_ERROR:
            brand_mask = df['brand_name'].str.lower().str.contains(brand.lower(), na=False)
            price_cols = ['avg_final_price', 'median_final_price']
            for col in price_cols:
                if col in df.columns:
                    df.loc[brand_mask, col] = df.loc[brand_mask, col] / 100
    
    # --- 4.2 Fix all remaining high prices (user confirmed all products < 5M VND) ---
    MAX_PRICE = 5_000_000
    if 'avg_final_price' in df.columns:
        high_mask = df['avg_final_price'] > MAX_PRICE
        df.loc[high_mask, 'avg_final_price'] = df.loc[high_mask, 'avg_final_price'] / 100
    if 'median_final_price' in df.columns:
        high_mask = df['median_final_price'] > MAX_PRICE
        df.loc[high_mask, 'median_final_price'] = df.loc[high_mask, 'median_final_price'] / 100

    # --- 5. Create binary indicator for missing reviews ---
    df["has_review"] = (~df["rating_mean"].isna()).astype(int)
    
    # --- 6. Handle missing values properly ---
    # Price: fill with median (product with no price data is rare)
    df["avg_final_price"] = df["avg_final_price"].fillna(df["avg_final_price"].median())
    df["median_final_price"] = df["median_final_price"].fillna(df["median_final_price"].median())
    df["avg_discount_percent"] = df["avg_discount_percent"].fillna(0)  # No discount if no data
    df["max_discount_percent"] = df["max_discount_percent"].fillna(0)
    df["avg_bought"] = df["avg_bought"].fillna(0)  # No sales data = 0
    df["price_std"] = df["price_std"].fillna(0)
    
    # Review: impute rating_mean with category median if enabled
    if impute_missing_rating:
        category_rating_median = df.groupby("category_id")["rating_mean"].transform("median")
        global_rating_median = df["rating_mean"].median()
        # Use category median first, then global median
        df["rating_mean"] = df["rating_mean"].fillna(category_rating_median)
        df["rating_mean"] = df["rating_mean"].fillna(global_rating_median)
    else:
        # Keep NaN for products without reviews (will be excluded from clustering)
        pass
    
    df["rating_std"] = df["rating_std"].fillna(0)
    df["review_count"] = df["review_count"].fillna(0)
    
    # Inventory: fill with sensible defaults
    df["avg_stock_available"] = df["avg_stock_available"].fillna(0)
    df["stockout_rate"] = df["stockout_rate"].fillna(1.0)  # No data = assume stockout
    df["stock_std"] = df["stock_std"].fillna(0)
    
    # --- 7. Create derived features ---
    # Stock rate (inverse of stockout)
    df["stock_rate"] = 1 - df["stockout_rate"]
    
    # Discount intensity (how often product is discounted heavily)
    df["discount_intensity"] = df["avg_discount_percent"] / (df["max_discount_percent"] + 1)
    
    # --- 8. Apply log1p transformation for skewed features ---
    if apply_log_transform:
        df["log_bought"] = np.log1p(df["avg_bought"])
        df["log_review_count"] = np.log1p(df["review_count"])
        df["log_price"] = np.log1p(df["avg_final_price"])
        df["log_stock"] = np.log1p(df["avg_stock_available"])
    
    # --- 9. Define feature columns for clustering ---
    if apply_log_transform:
        feature_cols = [
            "log_price",           # Log-transformed price
            "avg_discount_percent",# Discount (already bounded 0-100)
            "log_bought",          # Log-transformed sales
            "rating_mean",         # Rating (bounded 1-5)
            "log_review_count",    # Log-transformed review count
            "stock_rate",          # Stock availability rate (0-1)
            "has_review",          # Binary: has review or not
        ]
    else:
        feature_cols = [
            "avg_final_price",
            "avg_discount_percent",
            "avg_bought",
            "rating_mean",
            "review_count",
            "stock_rate",
            "has_review",
        ]
    
    return df, feature_cols


def clean_review_text(text: str) -> str:
    """
    Clean review text: remove HTML tags, normalize whitespace.
    """
    import re
    if pd.isna(text):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', str(text))
    # Remove special chars but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def classify_price_sentiment_rule_based(
    text: str,
    positive_keywords: List[str] = None,
    negative_keywords: List[str] = None,
    handle_negation: bool = True
) -> str:
    """
    Rule-based price sentiment classification with negation handling.
    
    Args:
        text: Cleaned review text
        positive_keywords: List of positive price keywords
        negative_keywords: List of negative price keywords
        handle_negation: Whether to handle negation patterns
        
    Returns:
        'Positive', 'Negative', 'Neutral', or 'No_Mention'
    """
    if pd.isna(text) or len(text) < 5:
        return 'No_Mention'
    
    # Default keywords
    if positive_keywords is None:
        positive_keywords = [
            'xứng đáng', 'xứng tiền', 'rẻ', 'hợp lý', 'giá tốt', 'giá rẻ',
            'giá hợp lý', 'đáng tiền', 'đáng giá', 'phải chăng', 'tiết kiệm', 
            'hời', 'giá mềm', 'giá ok', 'giá ổn', 'giá phù hợp'
        ]
    if negative_keywords is None:
        negative_keywords = [
            'đắt', 'mắc', 'chát', 'đắt quá', 'giá cao', 'quá đắt', 
            'đắt đỏ', 'giá đắt', 'không xứng tiền', 'lãng phí', 'giá trên trời'
        ]
    
    text_lower = text.lower()
    
    # Negation patterns to exclude
    negation_patterns_exclude_neg = ['không đắt', 'không mắc', 'không chát', 'ko đắt', 'ko mắc']
    negation_patterns_exclude_pos = ['không rẻ', 'không hợp lý', 'ko rẻ']
    
    pos_count = 0
    neg_count = 0
    
    # Count positive keywords
    for kw in positive_keywords:
        if kw in text_lower:
            # Check if preceded by negation
            if handle_negation:
                is_negated = any(neg_pat in text_lower for neg_pat in negation_patterns_exclude_pos if kw in neg_pat)
                if not is_negated:
                    pos_count += 1
            else:
                pos_count += 1
    
    # Count negative keywords
    for kw in negative_keywords:
        if kw in text_lower:
            # Check if negated
            if handle_negation:
                is_negated = any(neg_pat in text_lower for neg_pat in negation_patterns_exclude_neg if kw in neg_pat)
                if not is_negated:
                    neg_count += 1
            else:
                neg_count += 1
    
    # Check if any price mention
    has_price_mention = pos_count > 0 or neg_count > 0
    
    if not has_price_mention:
        return 'No_Mention'
    elif pos_count > neg_count:
        return 'Positive'
    elif neg_count > pos_count:
        return 'Negative'
    else:
        return 'Neutral'


def aggregate_product_sentiment(
    reviews_df: pd.DataFrame,
    sentiment_col: str = 'price_sentiment',
    min_mentions: int = 3
) -> pd.DataFrame:
    """
    Aggregate sentiment per product with ratio-based metrics.
    
    Args:
        reviews_df: DataFrame with review_content and sentiment
        sentiment_col: Column name for sentiment
        min_mentions: Minimum price mentions to include product
        
    Returns:
        DataFrame with per-product sentiment metrics
    """
    # Filter reviews with price mentions
    price_mentions = reviews_df[reviews_df[sentiment_col] != 'No_Mention'].copy()
    
    # Aggregate
    agg = reviews_df.groupby('product_id').agg(
        total_reviews=('review_id', 'count'),
        product_name=('product_name', 'first'),
        brand_name=('brand_name', 'first'),
    ).reset_index()
    
    # Sentiment counts
    price_agg = price_mentions.groupby('product_id').agg(
        price_mentions_count=('review_id', 'count'),
        positive_count=(sentiment_col, lambda x: (x == 'Positive').sum()),
        negative_count=(sentiment_col, lambda x: (x == 'Negative').sum()),
        neutral_count=(sentiment_col, lambda x: (x == 'Neutral').sum()),
    ).reset_index()
    
    # Merge
    result = agg.merge(price_agg, on='product_id', how='left')
    result['price_mentions_count'] = result['price_mentions_count'].fillna(0)
    result['positive_count'] = result['positive_count'].fillna(0)
    result['negative_count'] = result['negative_count'].fillna(0)
    result['neutral_count'] = result['neutral_count'].fillna(0)
    
    # Calculate ratios
    result['price_mention_rate'] = result['price_mentions_count'] / result['total_reviews']
    result['positive_ratio'] = np.where(
        result['price_mentions_count'] >= min_mentions,
        result['positive_count'] / result['price_mentions_count'],
        np.nan
    )
    result['negative_ratio'] = np.where(
        result['price_mentions_count'] >= min_mentions,
        result['negative_count'] / result['price_mentions_count'],
        np.nan
    )
    
    return result


def get_pricing_action_candidates(
    df_features: pd.DataFrame,
    cluster_col: str = 'cluster',
    stockout_threshold: float = 0.3,
    bought_percentile: float = 0.7,
    discount_low_percentile: float = 0.3,
    discount_high_percentile: float = 0.7
) -> dict:
    """
    Get actionable product lists based on cluster profile and business rules.
    
    Priority order (supply constraint first):
    1. Do not promo - need restock (stockout_rate > threshold)
    2. Increase price candidates (high bought, low discount, good stock)
    3. Reduce discount candidates (high discount, low bought)
    
    Returns:
        dict with 'increase_price', 'reduce_discount', 'need_restock' DataFrames
    """
    df = df_features.copy()
    
    # Thresholds
    bought_high = df['avg_bought'].quantile(bought_percentile)
    discount_low = df['avg_discount_percent'].quantile(discount_low_percentile)
    discount_high = df['avg_discount_percent'].quantile(discount_high_percentile)
    
    # 1. Need restock (priority)
    need_restock = df[df['stockout_rate'] > stockout_threshold].copy()
    need_restock = need_restock.sort_values('avg_bought', ascending=False)
    
    # 2. Increase price candidates
    # High bought + Low discount + Good stock
    increase_price = df[
        (df['avg_bought'] >= bought_high) &
        (df['avg_discount_percent'] <= discount_low) &
        (df['stockout_rate'] <= stockout_threshold)
    ].copy()
    increase_price = increase_price.sort_values('avg_bought', ascending=False)
    
    # 3. Reduce discount candidates
    # High discount + Low bought (discount not effective)
    bought_low = df['avg_bought'].quantile(1 - bought_percentile)
    reduce_discount = df[
        (df['avg_discount_percent'] >= discount_high) &
        (df['avg_bought'] <= bought_low) &
        (df['stockout_rate'] <= stockout_threshold)  # Exclude stock constrained
    ].copy()
    reduce_discount = reduce_discount.sort_values('avg_discount_percent', ascending=False)
    
    return {
        'increase_price': increase_price,
        'reduce_discount': reduce_discount,
        'need_restock': need_restock
    }
