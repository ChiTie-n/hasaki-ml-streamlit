
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
app_dir = Path(__file__).parent.parent
root_dir = app_dir.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(root_dir))

# Inject styling and load shared data
from styles import inject_page_css
from ml_utils import load_ml_data, clean_review_text, classify_price_sentiment_rule_based

st.set_page_config(page_title="Pricing Strategy", page_icon="💰", layout="wide")
inject_page_css()

st.title("Tổng hợp khuyến nghị về chiến lược giá")

try:
    df_feat, feature_cols, reviews_raw = load_ml_data()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["Decision Dashboard", "Price Simulator (ML)"])

# --------------------------------------------------------------------------------
# TAB 1: DECISION DASHBOARD (COPIED EXACTLY)
# --------------------------------------------------------------------------------
with tab1:
    st.header("Tổng hợp khuyến nghị về chiến lược giá")
    st.markdown("""
    *Tổng hợp signals từ Segmentation & Sentiment để ra quyết định.*
    """) 
    st.divider()

    # Run all analyses
    run_decision = st.button("🔄 Chạy Phân Tích Tổng Hợp", type="primary", use_container_width=True)
    
    # PERSISTENCE
    if run_decision or 'decision_df' in st.session_state:
        if run_decision:
            with st.spinner("Đang phân tích..."):
                
                # 1. Get feature data
                df_products = df_feat.copy()
                
                # 2. K-Means signals
                st.info("Đang chạy K-Means...")
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                
                feature_cols_km = ['avg_final_price', 'avg_discount_percent', 'avg_bought', 
                                  'rating_mean', 'stockout_rate']
                available_cols = [c for c in feature_cols_km if c in df_products.columns]
                
                X_km = df_products[available_cols].fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_km)
                
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                df_products['km_cluster'] = kmeans.fit_predict(X_scaled)
                
                # Assign K-Means signal based on cluster profile
                overall_price = df_products['avg_final_price'].mean()
                overall_bought = df_products['avg_bought'].mean()
                overall_discount = df_products['avg_discount_percent'].mean()
                
                def get_kmeans_signal(row):
                    if row['stockout_rate'] > 0.3:
                        return 0  # RESTOCK (neutral for pricing)
                    elif row['avg_bought'] > overall_bought * 1.2 and row['avg_discount_percent'] < overall_discount:
                        return 2  # INCREASE PRICE
                    elif row['avg_discount_percent'] > overall_discount * 1.2 and row['avg_bought'] < overall_bought * 0.8:
                        return -1  # REDUCE DISCOUNT / REVIEW
                    elif row['avg_final_price'] > overall_price * 1.3 and row['avg_bought'] > overall_bought:
                        return 1  # PREMIUM - can increase
                    else:
                        return 0  # HOLD
                
                df_products['km_signal'] = df_products.apply(get_kmeans_signal, axis=1)
                
                # 3. Discount-Demand Segmentation signals
                st.info("Đang phân loại Discount-Demand...")
                discount_median = df_products['avg_discount_percent'].median()
                bought_median = df_products['avg_bought'].median()
                
                def get_dd_signal(row):
                    if row['avg_discount_percent'] < discount_median and row['avg_bought'] > bought_median:
                        return 2  # Pricing Power → can increase
                    elif row['avg_discount_percent'] > discount_median and row['avg_bought'] > bought_median:
                        return 0  # Price Sensitive → hold
                    elif row['avg_discount_percent'] > discount_median and row['avg_bought'] < bought_median:
                        return -1  # Discount Ineffective → reduce/review
                    else:
                        return 0  # Low Performer → hold
                
                df_products['dd_signal'] = df_products.apply(get_dd_signal, axis=1)
                
                # 4. Sentiment signals (prefer ML over Rule-Based)
                if 'ml_sentiment_results' in st.session_state:
                    st.info("Đang dùng ML Sentiment từ Tab 4 (cached)...")
                    ml_results = st.session_state['ml_sentiment_results']
                    
                    # Merge ML sentiment results
                    df_products = df_products.merge(
                        ml_results[['product_id', 'pos_ratio', 'neg_ratio']], 
                        on='product_id', 
                        how='left'
                    )
                    df_products['pos_ratio'] = df_products['pos_ratio'].fillna(0.5)
                    df_products['neg_ratio'] = df_products['neg_ratio'].fillna(0.5)
                    
                    def get_sentiment_signal(row):
                        if row['pos_ratio'] > 0.6:
                            return 2  # Strong positive → can increase
                        elif row['pos_ratio'] > 0.4:
                            return 1  # Mild positive
                        elif row['neg_ratio'] > 0.5:
                            return -2  # Strong negative → should decrease
                        elif row['neg_ratio'] > 0.3:
                            return -1  # Mild negative
                        return 0
                    
                    df_products['sentiment_signal'] = df_products.apply(get_sentiment_signal, axis=1)
                    sentiment_source = "ML"
                else:
                    st.warning("⚠️ Chưa có ML Sentiment. Đang dùng Rule-Based (chạy Tab 4 để có kết quả tốt hơn)...")
                    if 'product_id' in reviews_raw.columns and 'review_content' in reviews_raw.columns:
                        reviews_sent = reviews_raw.copy()
                        reviews_sent['clean_text'] = reviews_sent['review_content'].fillna('').apply(clean_review_text)
                        reviews_sent['rule_sentiment'] = reviews_sent['clean_text'].apply(
                            lambda x: classify_price_sentiment_rule_based(x)
                        )
                        
                        rule_agg = reviews_sent.groupby('product_id').apply(
                            lambda x: pd.Series({
                                'rule_pos': (x['rule_sentiment'] == 'Positive').sum(),
                                'rule_neg': (x['rule_sentiment'] == 'Negative').sum(),
                                'rule_total': len(x[x['rule_sentiment'].isin(['Positive', 'Negative'])])
                            })
                        ).reset_index()
                        
                        rule_agg['pos_ratio'] = rule_agg['rule_pos'] / rule_agg['rule_total'].replace(0, 1)
                        rule_agg['neg_ratio'] = rule_agg['rule_neg'] / rule_agg['rule_total'].replace(0, 1)
                        
                        df_products = df_products.merge(rule_agg[['product_id', 'pos_ratio', 'neg_ratio']], 
                                                        on='product_id', how='left')
                        df_products['pos_ratio'] = df_products['pos_ratio'].fillna(0.5)
                        df_products['neg_ratio'] = df_products['neg_ratio'].fillna(0.5)
                        
                        def get_sentiment_signal(row):
                            if row['pos_ratio'] > 0.6:
                                return 1
                            elif row['neg_ratio'] > 0.5:
                                return -1
                            return 0
                        
                        df_products['sentiment_signal'] = df_products.apply(get_sentiment_signal, axis=1)
                    else:
                        df_products['sentiment_signal'] = 0
                    sentiment_source = "Rule-Based"
                
                # 5. Compute final score
                st.info("Đang tính điểm tổng hợp...")
                # Weight ML Sentiment higher than Rule-Based
                sentiment_weight = 2.0 if sentiment_source == 'ML' else 1.5
                df_products['total_score'] = (
                    df_products['km_signal'] * 2 +  # K-Means weight: 2
                    df_products['dd_signal'] * 1.5 +  # Discount-Demand weight: 1.5
                    df_products['sentiment_signal'] * sentiment_weight  # Sentiment weight
                )
                st.info(f"Sentiment source: **{sentiment_source}** (weight: {sentiment_weight})")
                
                # 6. Generate final recommendation
                def get_final_recommendation(score):
                    if score >= 3:
                        return "🟢 CÓ ROOM TĂNG GIÁ"
                    elif score >= 1.5:
                        return "🟡 CÓ THỂ TĂNG NHẸ"
                    elif score <= -2:
                        return "🔴 GIẢM GIÁ"
                    elif score <= -0.5:
                        return "🟠 XEM XÉT GIẢM"
                    else:
                        return "⚪ GIỮ GIÁ"
                
                df_products['recommendation'] = df_products['total_score'].apply(get_final_recommendation)
                
                # Calculate confidence
                def get_confidence(row):
                    signals = [row['km_signal'], row['dd_signal'], row.get('sentiment_signal', 0)]
                    # Count how many signals agree on direction
                    pos_count = sum(1 for s in signals if s > 0)
                    neg_count = sum(1 for s in signals if s < 0)
                    max_agree = max(pos_count, neg_count)
                    return "HIGH" if max_agree >= 2 else "MEDIUM" if max_agree == 1 else "LOW"
                
                df_products['confidence'] = df_products.apply(get_confidence, axis=1)
                
                st.success("✅ Phân tích hoàn tất!")
                
                # Store in session
                st.session_state['decision_df'] = df_products
    
    # Display results
    if 'decision_df' in st.session_state:
        df_decision = st.session_state['decision_df']
        
        st.divider()
        
        # Summary metrics
        st.subheader("Tổng Quan")
        rec_counts = df_decision['recommendation'].value_counts()
        
        cols = st.columns(5)
        for i, (rec, count) in enumerate(rec_counts.items()):
            cols[i % 5].metric(rec, count)
        
        st.divider()
        
        # Show by recommendation
        st.subheader("Chi Tiết Theo Khuyến Nghị")
        
        recommendation_filter = st.selectbox(
            "Lọc theo khuyến nghị:",
            ["Tất cả"] + list(df_decision['recommendation'].unique())
        )
        
        if recommendation_filter != "Tất cả":
            df_show = df_decision[df_decision['recommendation'] == recommendation_filter]
        else:
            df_show = df_decision
        
        display_cols = ['product_name', 'avg_final_price', 'avg_discount_percent', 
                       'avg_bought', 'total_score', 'confidence', 'recommendation']
        available_display = [c for c in display_cols if c in df_show.columns]
        
        st.dataframe(
            df_show[available_display].sort_values('total_score', ascending=False).head(50),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # Top Actions
        st.subheader("Top Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🟢 Top 10 CÓ ROOM TĂNG GIÁ**")
            top_increase = df_decision[df_decision['recommendation'].str.contains('TĂNG')].nlargest(10, 'total_score')
            if len(top_increase) > 0:
                for _, row in top_increase.iterrows():
                    st.write(f"• {row['product_name'][:30]}... ({row['confidence']})")
            else:
                st.info("Không có")
        
        with col2:
            st.markdown("**⚪ GIỮ GIÁ (Cần Review)**")
            hold_items = df_decision[df_decision['recommendation'].str.contains('GIỮ')].head(10)
            if len(hold_items) > 0:
                for _, row in hold_items.iterrows():
                    st.write(f"• {row['product_name'][:30]}...")
            else:
                st.info("Không có")
        
        with col3:
            st.markdown("**🔴 Top 10 GIẢM GIÁ**")
            top_decrease = df_decision[df_decision['recommendation'].str.contains('GIẢM')].nsmallest(10, 'total_score')
            if len(top_decrease) > 0:
                for _, row in top_decrease.iterrows():
                    st.write(f"• {row['product_name'][:30]}... ({row['confidence']})")
            else:
                st.info("Không có")
        
        st.divider()
    else:
        st.info("👆 Nhấn nút **Chạy Phân Tích Tổng Hợp** để bắt đầu.")

# --------------------------------------------------------------------------------
# TAB 2: PRICE SIMULATOR (ML REGRESSION) (COPIED EXACTLY)
# --------------------------------------------------------------------------------
with tab2:
    st.header("Dự đoán giá tối ưu (ML Regression)")
    
    st.divider()
    
    # Train model
    st.subheader("Train Demand Prediction Model")
    
    
    # Model selection
    model_options = {
        "GradientBoosting": "Ensemble - tốt cho tabular data, học từ errors",
        "RandomForest": "Ensemble - robust, ít overfit, dễ tune",
        "LinearRegression": "Simple baseline - dễ interpret",
        "Ridge": "Linear với regularization - tránh overfit"
    }
    
    selected_model = st.selectbox(
        "Chọn mô hình:",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x} - {model_options[x]}"
    )
    
    # Check if model already exists in session
    model_exists = 'demand_model' in st.session_state
    
    # Button to train (run)
    run_train = st.button("Train Model", type="primary")
    
    # LOGIC: Run if button clicked OR (results exist AND we want to display metrics)
    # But for training, typically we only re-train if requested. 
    # If model exists, we just show the metrics stored.
    
    if run_train:
        with st.spinner(f"Đang train {selected_model}..."):
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            df_train = df_feat.copy()
            
            # Basic features
            feature_cols_ml = ['avg_final_price', 'avg_discount_percent', 'rating_mean', 'stock_rate']
            available_cols_ml = [c for c in feature_cols_ml if c in df_train.columns]
            
            # Remove rows with missing target
            df_train = df_train.dropna(subset=['avg_bought'] + available_cols_ml)
            
            # Remove outliers using IQR
            Q1 = df_train['avg_bought'].quantile(0.05)
            Q3 = df_train['avg_bought'].quantile(0.95)
            df_train = df_train[(df_train['avg_bought'] >= Q1) & (df_train['avg_bought'] <= Q3)]
            
            # Remove zero/negative values
            df_train = df_train[df_train['avg_bought'] > 0]
            df_train = df_train[df_train['avg_final_price'] > 0]
            
            if len(df_train) < 100:
                st.error("Không đủ dữ liệu để train (cần ít nhất 100 rows)")
            else:
                # Feature Engineering
                df_train['price_per_rating'] = df_train['avg_final_price'] / (df_train['rating_mean'].replace(0, 4.0))
                df_train['log_price'] = np.log1p(df_train['avg_final_price'])
                df_train['discount_x_stock'] = df_train['avg_discount_percent'] * df_train['stock_rate']
                
                # Updated feature list
                feature_cols_final = available_cols_ml + ['price_per_rating', 'log_price', 'discount_x_stock']
                feature_cols_final = [c for c in feature_cols_final if c in df_train.columns]
                
                X = df_train[feature_cols_final].fillna(0)
                
                # Log transform target (critical for skewed data!)
                y = np.log1p(df_train['avg_bought'])
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols_final, index=X.index)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled_df, y, test_size=0.2, random_state=42
                )
                
                # Select and train model
                if selected_model == "GradientBoosting":
                    model = GradientBoostingRegressor(
                        n_estimators=100, max_depth=5, learning_rate=0.1,
                        min_samples_split=10, random_state=42
                    )
                elif selected_model == "RandomForest":
                    model = RandomForestRegressor(
                        n_estimators=100, max_depth=10, min_samples_split=5,
                        random_state=42, n_jobs=-1
                    )
                elif selected_model == "LinearRegression":
                    model = LinearRegression()
                else:  # Ridge
                    model = Ridge(alpha=1.0, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Evaluate on log scale
                y_pred_log = model.predict(X_test)
                
                # Convert back to original scale for metrics
                y_pred = np.expm1(y_pred_log)
                y_test_original = np.expm1(y_test)
                
                mae = mean_absolute_error(y_test_original, y_pred)
                r2 = r2_score(y_test, y_pred_log)  # R² on log scale
                
                # Store model and scaler
                st.session_state['demand_model'] = model
                st.session_state['demand_scaler'] = scaler
                st.session_state['model_features'] = feature_cols_final
                st.session_state['df_simulator'] = df_train
                st.session_state['use_log_transform'] = True
                st.session_state['model_name'] = selected_model
                
                # Store metrics for display
                st.session_state['train_mae'] = mae
                st.session_state['train_r2'] = r2
                st.session_state['train_samples'] = len(X_train)
                st.session_state['train_features'] = len(feature_cols_final)
                
                st.success(f"✅ **{selected_model}** trained thành công!")
    
    # DISPLAY if model exists
    if 'demand_model' in st.session_state:
        # Load metrics
        mae = st.session_state.get('train_mae', 0)
        r2 = st.session_state.get('train_r2', 0)
        n_samples = st.session_state.get('train_samples', 0)
        n_feats = st.session_state.get('train_features', 0)
        model = st.session_state['demand_model']
        feature_cols_final = st.session_state.get('model_features', [])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.1f}")
        col2.metric("R² Score", f"{r2:.3f}")
        col3.metric("Training samples", n_samples)
        col4.metric("Features", n_feats)
        
        # Quality assessment
        if r2 >= 0.3:
            st.success("✅ Model quality: ACCEPTABLE - Có thể dùng để tham khảo")
        elif r2 >= 0.1:
            st.warning("⚠️ Model quality: LOW - Kết quả chỉ mang tính tham khảo")
        else:
            st.error("❌ Model quality: POOR - Không nên dùng để ra quyết định")

        # Feature importance
        st.subheader("Chỉ số quan trọng")
        
        # Check if model has feature_importances_ (GBM, RF) or coef_ (Linear)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            importance = []
            
        if len(importance) > 0:
            importance_df = pd.DataFrame({
                'Feature': feature_cols_final,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Các yếu tố ảnh hưởng đến Demand"
            )
            fig_imp.update_layout(
                title_x=0.5,
                title_xanchor='center',
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    
    # Simulation
    if 'demand_model' in st.session_state:
        st.subheader("Simulate Price Changes")
        
        model = st.session_state['demand_model']
        features = st.session_state['model_features']
        df_sim = st.session_state['df_simulator']
        
        st.markdown("**Chọn loại simulation:**")
        
        sim_type = st.radio(
            "Simulation type:",
            ["Thay đổi Discount (%)", "Thay đổi Giá (%)", "Single Product"],
            horizontal=True
        )
        
        if sim_type == "Thay đổi Discount (%)":
            discount_change = st.slider(
                "Thay đổi discount (điểm %):",
                min_value=-20, max_value=20, value=5, step=1
            )
            
            if st.button("Ước tính thay đổi"):
                scaler = st.session_state.get('demand_scaler', None)
                use_log = st.session_state.get('use_log_transform', False)
                
                # Create original features
                X_original = df_sim[features].fillna(0).copy()
                
                # Create simulated features with updated discount
                X_simulated = X_original.copy()
                new_discount = (X_simulated['avg_discount_percent'] + discount_change).clip(0, 100)
                X_simulated['avg_discount_percent'] = new_discount
                
                # Update engineered features based on new discount
                if 'discount_x_stock' in features and 'stock_rate' in df_sim.columns:
                    X_simulated['discount_x_stock'] = new_discount * df_sim['stock_rate'].fillna(0.5)
                
                # Apply scaler if available
                if scaler is not None:
                    X_original_scaled = scaler.transform(X_original)
                    X_simulated_scaled = scaler.transform(X_simulated)
                else:
                    X_original_scaled = X_original.values
                    X_simulated_scaled = X_simulated.values
                
                # Predict
                pred_original_raw = model.predict(X_original_scaled)
                pred_simulated_raw = model.predict(X_simulated_scaled)
                
                # Convert from log scale if needed
                if use_log:
                    pred_original = np.expm1(pred_original_raw)
                    pred_simulated = np.expm1(pred_simulated_raw)
                else:
                    pred_original = pred_original_raw
                    pred_simulated = pred_simulated_raw
                
                # Compare
                df_result = df_sim[['product_name', 'avg_final_price', 'avg_discount_percent', 'avg_bought']].copy()
                df_result['predicted_original'] = pred_original
                df_result['predicted_new'] = pred_simulated
                df_result['delta_bought'] = pred_simulated - pred_original
                df_result['delta_pct'] = np.where(
                    pred_original > 0.1,
                    (df_result['delta_bought'] / pred_original * 100).round(1),
                    0
                )
                
                # Summary
                avg_delta = df_result['delta_pct'].mean()
                
                if avg_delta > 0:
                    st.success(f"📈 **Kết quả:** Tăng discount {discount_change}% → Demand dự kiến tăng **{avg_delta:.1f}%**")
                else:
                    st.warning(f"📉 **Kết quả:** Thay đổi discount {discount_change}% → Demand dự kiến giảm **{abs(avg_delta):.1f}%**")
                
                # Show top gainers/losers
                st.markdown("**Top 10 sản phẩm hưởng lợi nhất:**")
                st.dataframe(
                    df_result.nlargest(10, 'delta_bought')[['product_name', 'avg_bought', 'predicted_new', 'delta_pct']],
                    use_container_width=True,
                    hide_index=True
                )
        
        elif sim_type == "Thay đổi Giá (%)":
            price_change = st.slider(
                "Thay đổi giá (%):",
                min_value=-30, max_value=30, value=-10, step=5
            )
            
            if st.button("Ước tính thay đổi giá"):
                scaler = st.session_state.get('demand_scaler', None)
                use_log = st.session_state.get('use_log_transform', False)
                
                # Create original features
                X_original = df_sim[features].fillna(0).copy()
                
                # Create simulated features with updated price
                X_simulated = X_original.copy()
                new_prices = X_simulated['avg_final_price'] * (1 + price_change/100)
                X_simulated['avg_final_price'] = new_prices.clip(0, None)
                
                # Update engineered features based on new price
                if 'log_price' in features:
                    X_simulated['log_price'] = np.log1p(new_prices)
                if 'price_per_rating' in features and 'rating_mean' in df_sim.columns:
                    rating = df_sim['rating_mean'].fillna(4.0)
                    X_simulated['price_per_rating'] = new_prices / rating.clip(lower=1.0)
                
                # Apply scaler if available
                if scaler is not None:
                    X_original_scaled = scaler.transform(X_original)
                    X_simulated_scaled = scaler.transform(X_simulated)
                else:
                    X_original_scaled = X_original.values
                    X_simulated_scaled = X_simulated.values
                
                # Predict
                pred_original_raw = model.predict(X_original_scaled)
                pred_simulated_raw = model.predict(X_simulated_scaled)
                
                # Convert from log scale if needed
                if use_log:
                    pred_original = np.expm1(pred_original_raw)
                    pred_simulated = np.expm1(pred_simulated_raw)
                else:
                    pred_original = pred_original_raw
                    pred_simulated = pred_simulated_raw
                
                # Compare
                df_result = df_sim[['product_name', 'avg_final_price', 'avg_bought']].copy()
                df_result['new_price'] = df_result['avg_final_price'] * (1 + price_change/100)
                df_result['predicted_original'] = pred_original
                df_result['predicted_new'] = pred_simulated
                df_result['delta_bought'] = pred_simulated - pred_original
                df_result['delta_pct'] = np.where(
                    pred_original > 0.1,
                    (df_result['delta_bought'] / pred_original * 100).round(1),
                    0
                )
                
                # Revenue impact estimate
                df_result['revenue_original'] = df_result['avg_final_price'] * pred_original
                df_result['revenue_new'] = df_result['new_price'] * pred_simulated
                df_result['revenue_delta_pct'] = np.where(
                    df_result['revenue_original'] > 0.1,
                    ((df_result['revenue_new'] - df_result['revenue_original']) / 
                     df_result['revenue_original'] * 100).round(1),
                    0
                )
                
                # Summary
                avg_demand_delta = df_result['delta_pct'].mean()
                avg_revenue_delta = df_result['revenue_delta_pct'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    if avg_demand_delta > 0:
                        st.success(f"📈 Demand: **+{avg_demand_delta:.1f}%**")
                    else:
                        st.warning(f"📉 Demand: **{avg_demand_delta:.1f}%**")
                
                with col2:
                    if avg_revenue_delta > 0:
                        st.success(f"💰 Revenue: **+{avg_revenue_delta:.1f}%**")
                    else:
                        st.error(f"💸 Revenue: **{avg_revenue_delta:.1f}%**")
                
                st.markdown("**Chi tiết top 10:**")
                st.dataframe(
                    df_result.nlargest(10, 'revenue_delta_pct')[
                        ['product_name', 'avg_final_price', 'new_price', 'delta_pct', 'revenue_delta_pct']
                    ],
                    use_container_width=True,
                    hide_index=True
                )
        
        else:  # Single Product
            product_list = df_sim['product_name'].tolist()
            selected_product = st.selectbox("Chọn sản phẩm:", product_list[:100])
            
            product_data = df_sim[df_sim['product_name'] == selected_product].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Giá hiện tại:** {product_data['avg_final_price']:,.0f}đ")
                st.write(f"**Discount hiện tại:** {product_data['avg_discount_percent']:.1f}%")
            with col2:
                st.write(f"**Bought hiện tại:** {product_data['avg_bought']:.0f}")
                st.write(f"**Rating:** {product_data.get('rating_mean', 'N/A')}")
            
            # Limit to training data range to avoid extrapolation
            max_discount_in_data = int(df_sim['avg_discount_percent'].quantile(0.95))  # 95th percentile
            min_discount_in_data = int(df_sim['avg_discount_percent'].quantile(0.05))  # 5th percentile
            
            new_price = st.number_input(
                "Nhập giá mới (VNĐ):", 
                min_value=1000, 
                max_value=10000000,
                value=int(product_data['avg_final_price']),
                step=1000
            )
            new_discount = st.slider(
                f"Nhập discount mới (%) - Range trong data: {min_discount_in_data}%-{max_discount_in_data}%:",
                min_value=min_discount_in_data, 
                max_value=max_discount_in_data,
                value=min(max(int(product_data['avg_discount_percent']), min_discount_in_data), max_discount_in_data)
            )
            
            if st.button("Ước tính thay đổi"):
                # Get scaler if available
                scaler = st.session_state.get('demand_scaler', None)
                use_log = st.session_state.get('use_log_transform', False)
                
                # Get current product's other features
                rating = product_data.get('rating_mean', 4.0)
                stock = product_data.get('stock_rate', 0.5)
                
                # Create feature dict matching training features
                feature_dict = {
                    'avg_final_price': new_price,
                    'avg_discount_percent': new_discount,
                    'rating_mean': rating,
                    'stock_rate': stock,
                    'price_per_rating': new_price / max(rating, 1.0),
                    'log_price': np.log1p(new_price),
                    'discount_x_stock': new_discount * stock
                }
                
                # Create DataFrame with only the features the model expects
                X_new = pd.DataFrame([[feature_dict.get(f, 0) for f in features]], columns=features)
                
                # Apply scaling if scaler exists
                if scaler is not None:
                    X_new_scaled = scaler.transform(X_new)
                    X_new = pd.DataFrame(X_new_scaled, columns=features)
                
                # Predict
                predicted_log = model.predict(X_new)[0]
                
                # Convert back from log if needed
                if use_log:
                    predicted_bought = np.expm1(predicted_log)
                else:
                    predicted_bought = predicted_log
                
                current_bought = product_data['avg_bought']
                delta = predicted_bought - current_bought
                delta_pct = (delta / current_bought * 100) if current_bought > 0 else 0
                
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Bought hiện tại", f"{current_bought:.0f}")
                col2.metric("Bought dự đoán", f"{predicted_bought:.0f}", f"{delta_pct:+.1f}%")
                
                # Revenue comparison
                rev_current = product_data['avg_final_price'] * current_bought
                rev_new = new_price * predicted_bought
                rev_delta_pct = ((rev_new - rev_current) / rev_current * 100) if rev_current > 0 else 0
                
                col3.metric("Revenue impact", f"{rev_delta_pct:+.1f}%")
                
                if rev_delta_pct > 0:
                    st.success(f"✅ Thay đổi giá này có thể **TĂNG revenue** {rev_delta_pct:.1f}%")
                else:
                    st.warning(f"⚠️ Thay đổi giá này có thể **GIẢM revenue** {abs(rev_delta_pct):.1f}%")
        
        st.divider()

    else:
        st.info("👆 Train model trước khi simulation.")
