"""
ML Models - Phân cụm sản phẩm cho Chiến lược Giá (Academic Version)
4 Tabs: K-Means Clustering | Discount-Demand Segmentation | Sentiment Rule-Based | Sentiment ML

Author: AI Agent
Updated: 2024-12-21
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

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

st.set_page_config(page_title="ML Models", page_icon="🤖", layout="wide")

st.title("🤖 ML Models - Chiến Lược Giá")
st.markdown("*Phân tích phân cụm và sentiment để đưa ra quyết định giá*")


@st.cache_data(ttl=3600)
def load_data(limit=3000):
    """Load and prepare all data"""
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

try:
    df_feat, feature_cols, reviews_raw = load_data()
    st.success(f"✅ Đã tải {len(df_feat)} sản phẩm | Features: {len(feature_cols)}")
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 K-Means Clustering", 
    "💰 Discount-Demand Segmentation",
    "💬 Sentiment (Rule-Based)", 
    "🤖 Sentiment (ML Model)"
])

# TAB 1: K-MEANS CLUSTERING
with tab1:
    st.header("📊 K-Means Product Clustering")
    st.markdown("""
    *Phân cụm sản phẩm dựa trên đặc điểm giá, doanh số, rating, tồn kho.*
    
    **Cải tiến:**
    - ✅ Log-transform cho biến lệch (bought, review_count, price)
    - ✅ Impute rating theo category median thay vì fillna(0)
    - ✅ Ưu tiên ràng buộc tồn kho trong chiến lược
    """)
    
    st.divider()
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        use_auto_k = st.checkbox("Tự động chọn K (Silhouette)", value=True, key="t1_auto_k")
    with col2:
        k_manual = st.slider("Số cụm K", 2, 8, 4, key="t1_k")
    with col3:
        stockout_threshold = st.slider("Stockout threshold", 0.1, 0.5, 0.3, key="t1_stockout")
    
    run_kmeans = st.button("🚀 Chạy K-Means Clustering", type="primary", key="t1_run")
    
    if run_kmeans:
        with st.spinner("Đang phân cụm..."):
            # Prepare features
            X = df_feat[feature_cols].values
            
            # Handle any remaining NaN
            X = np.nan_to_num(X, nan=0)
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            

            st.subheader("📈 K Selection (Silhouette Analysis)")
            
            if use_auto_k:
                silhouette_scores = []
                k_range = range(2, 9)
                
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_tmp = km.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, labels_tmp)
                    silhouette_scores.append({'K': k, 'Silhouette': round(score, 4)})
                
                sil_df = pd.DataFrame(silhouette_scores)
                best_k = sil_df.loc[sil_df['Silhouette'].idxmax(), 'K']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(sil_df, use_container_width=True, hide_index=True)
                with col2:
                    fig_sil = px.line(sil_df, x='K', y='Silhouette', markers=True,
                                      title="Silhouette Score vs K")
                    fig_sil.add_vline(x=best_k, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_sil, use_container_width=True)
                
                st.success(f"🎯 Best K = {best_k} (Silhouette = {sil_df[sil_df['K']==best_k]['Silhouette'].values[0]})")
            else:
                best_k = k_manual
            
            st.divider()
            

            st.subheader("🎨 Clustering Results")
            
            # Final clustering
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Add to dataframe
            df_result = df_feat.copy()
            df_result['cluster'] = labels
            df_result['cluster_str'] = df_result['cluster'].astype(str)  # For discrete color
            df_result['PC1'] = X_pca[:, 0]
            df_result['PC2'] = X_pca[:, 1]
            
            st.success(f"✅ Đã phân thành {best_k} cụm!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PCA Scatter - DISCRETE COLOR
                fig_pca = px.scatter(
                    df_result, x='PC1', y='PC2',
                    color='cluster_str',
                    hover_data=['product_name', 'brand_name', 'avg_final_price', 'avg_bought'],
                    title="Clusters trên không gian PCA",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_pca.update_layout(height=450)
                st.plotly_chart(fig_pca, use_container_width=True)
            
            with col2:
                # Cluster size
                cluster_counts = df_result['cluster'].value_counts().sort_index()
                fig_size = px.bar(
                    x=[str(x) for x in cluster_counts.index],
                    y=cluster_counts.values,
                    title="Số sản phẩm mỗi cluster",
                    labels={'x': 'Cluster', 'y': 'Số sản phẩm'},
                    text=cluster_counts.values,
                    color=[str(x) for x in cluster_counts.index],
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_size.update_traces(textposition='outside')
                fig_size.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig_size, use_container_width=True)
            
            st.divider()
            

            st.subheader("📋 Cluster Profile")
            
            profile_cols = ['avg_final_price', 'avg_discount_percent', 'avg_bought', 
                           'rating_mean', 'review_count', 'stockout_rate', 'stock_rate']
            
            cluster_summary = df_result.groupby('cluster')[profile_cols].agg(['mean', 'median']).round(2)
            cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
            cluster_summary['count'] = df_result.groupby('cluster').size()
            cluster_summary = cluster_summary.reset_index()
            
            st.dataframe(cluster_summary, use_container_width=True)
            
            st.divider()
            

            st.subheader("💰 Chiến Lược Giá (Ưu Tiên Ràng Buộc Tồn Kho)")
            
            st.warning(f"⚠️ **Lưu ý:** Stockout threshold = {stockout_threshold}. Sản phẩm có stockout_rate > {stockout_threshold} sẽ được xét ưu tiên trước.")
            
            # Get overall stats for comparison
            overall_price = df_result['avg_final_price'].mean()
            overall_bought = df_result['avg_bought'].mean()
            overall_discount = df_result['avg_discount_percent'].mean()
            
            for cluster_id in sorted(df_result['cluster'].unique()):
                cluster_data = df_result[df_result['cluster'] == cluster_id]
                
                avg_price = cluster_data['avg_final_price'].mean()
                avg_bought = cluster_data['avg_bought'].mean()
                avg_discount = cluster_data['avg_discount_percent'].mean()
                avg_stockout = cluster_data['stockout_rate'].mean()
                avg_rating = cluster_data['rating_mean'].mean()
                review_count_total = cluster_data['review_count'].sum()
                
                # PRIORITY 1: Supply constraint
                if avg_stockout > stockout_threshold:
                    cluster_type = "📦 SUPPLY CONSTRAINT"
                    strategy = "ƯU TIÊN RESTOCK trước khi PROMO"
                    color = "warning"
                    detail = f"Stockout rate cao ({avg_stockout:.1%}). Không chạy promotion cho đến khi đủ hàng."
                # PRIORITY 2: Premium performer
                elif avg_price > overall_price * 1.2 and avg_bought > overall_bought:
                    cluster_type = "🏆 PREMIUM PERFORMER"
                    strategy = "GIỮ GIÁ hoặc TĂNG NHẸ 5-10%"
                    color = "success"
                    detail = f"Giá cao ({avg_price:,.0f}đ) nhưng vẫn bán tốt ({avg_bought:.0f} bought)."
                # PRIORITY 3: Best seller low discount
                elif avg_bought > overall_bought * 1.3 and avg_discount < overall_discount:
                    cluster_type = "🔥 BEST SELLER (Low Discount)"
                    strategy = "CÓ THỂ TĂNG GIÁ 5-15%"
                    color = "success"
                    detail = f"Bán chạy mà không cần giảm giá nhiều. Có pricing power."
                # PRIORITY 4: Discount dependent
                elif avg_discount > overall_discount * 1.2 and avg_bought < overall_bought:
                    cluster_type = "⚠️ DISCOUNT DEPENDENT"
                    strategy = "GIẢM DISCOUNT, đổi chiến lược marketing"
                    color = "warning"
                    detail = f"Discount cao ({avg_discount:.1f}%) nhưng bought thấp ({avg_bought:.0f})."
                # PRIORITY 5: Value segment
                elif avg_price < overall_price * 0.7:
                    cluster_type = "💵 VALUE SEGMENT"
                    strategy = "GIỮ GIÁ, tối ưu cost"
                    color = "info"
                    detail = f"Phân khúc giá thấp ({avg_price:,.0f}đ). Focus on volume."
                else:
                    cluster_type = "📊 AVERAGE"
                    strategy = "MONITOR & A/B TEST giá"
                    color = "info"
                    detail = "Performance trung bình. Có thể test các mức giá khác."
                
                with st.expander(f"**Cluster {cluster_id}**: {cluster_type} ({len(cluster_data)} sản phẩm)", expanded=True):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Giá TB", f"{avg_price:,.0f}đ")
                    col2.metric("Discount TB", f"{avg_discount:.1f}%")
                    col3.metric("Bought TB", f"{avg_bought:.0f}")
                    col4.metric("Rating TB", f"{avg_rating:.2f}⭐")
                    col5.metric("Stockout Rate", f"{avg_stockout:.1%}")
                    
                    if color == "success":
                        st.success(f"**📌 Chiến lược:** {strategy}")
                    elif color == "warning":
                        st.warning(f"**📌 Chiến lược:** {strategy}")
                    else:
                        st.info(f"**📌 Chiến lược:** {strategy}")
                    
                    st.caption(f"💡 {detail}")
            
            st.divider()
            

            st.subheader("📝 Action Lists")
            
            action_candidates = get_pricing_action_candidates(
                df_result,
                stockout_threshold=stockout_threshold
            )
            
            display_cols = ['product_name', 'brand_name', 'avg_final_price', 
                           'avg_discount_percent', 'avg_bought', 'stockout_rate']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔼 Candidates TĂNG GIÁ**")
                inc_df = action_candidates['increase_price'].head(20)
                if len(inc_df) > 0:
                    st.dataframe(inc_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"💡 {len(inc_df)} sản phẩm bán chạy + discount thấp + stock tốt")
                else:
                    st.info("Không có sản phẩm phù hợp")
            
            with col2:
                st.markdown("**🔽 Candidates GIẢM DISCOUNT**")
                red_df = action_candidates['reduce_discount'].head(20)
                if len(red_df) > 0:
                    st.dataframe(red_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"💡 {len(red_df)} sản phẩm discount cao nhưng không hiệu quả")
                else:
                    st.info("Không có sản phẩm phù hợp")
            
            with col3:
                st.markdown("**📦 CẦN RESTOCK (Không promo)**")
                rest_df = action_candidates['need_restock'].head(20)
                if len(rest_df) > 0:
                    st.dataframe(rest_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"⚠️ {len(rest_df)} sản phẩm stockout cao - cần bổ sung hàng trước")
                else:
                    st.success("✅ Không có sản phẩm stockout cao")
            
            # Store for other tabs
            st.session_state['kmeans_result'] = df_result
            st.session_state['cluster_col'] = 'cluster'

# TAB 2: DISCOUNT-DEMAND SEGMENTATION
with tab2:
    st.header("💰 Discount-Demand Segmentation")
    st.markdown("""
    *Phân nhóm mô tả mối quan hệ giữa discount và demand (bought).*
    
    **⚠️ Lưu ý:** Đây là phân tích **mô tả (descriptive)**, không phải ước lượng **price elasticity** 
    thực sự vì cần dữ liệu theo thời gian với delta_bought để tính elasticity coefficient.
    
    **Cách đọc kết quả:**
    - High discount + High bought: Nhạy cảm giá (cần discount để bán)
    - Low discount + High bought: Ít nhạy cảm (có pricing power)
    - High discount + Low bought: Discount không hiệu quả
    """)
    
    st.divider()
    
    # Prepare data
    df_dd = df_feat[(df_feat['avg_final_price'] > 0) & (df_feat['avg_bought'] > 0)].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        n_clusters_dd = st.slider("Số nhóm phân tích", 2, 5, 3, key="t2_k")
    with col2:
        run_dd = st.button("🚀 Chạy Discount-Demand Segmentation", type="primary", key="t2_run")
    
    if run_dd:
        with st.spinner("Đang phân tích..."):
            # Create price segments for normalization
            df_dd['price_segment'] = pd.qcut(
                df_dd['avg_final_price'], q=4, 
                labels=['Budget', 'Mid', 'Premium', 'Luxury'],
                duplicates='drop'
            )
            
            # Normalize bought within price segment (fair comparison)
            df_dd['bought_normalized'] = df_dd.groupby('price_segment', observed=True)['avg_bought'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
            )
            
            # Features for clustering: discount + normalized bought
            features_dd = ['avg_discount_percent', 'bought_normalized']
            X_dd = df_dd[features_dd].fillna(0).values
            
            scaler = StandardScaler()
            X_dd_scaled = scaler.fit_transform(X_dd)
            
            # Clustering
            kmeans_dd = KMeans(n_clusters=n_clusters_dd, random_state=42, n_init=10)
            dd_labels = kmeans_dd.fit_predict(X_dd_scaled)
            
            df_dd['segment'] = dd_labels
            df_dd['segment_str'] = df_dd['segment'].astype(str)
            
            st.success(f"✅ Đã phân thành {n_clusters_dd} nhóm!")
            
            st.divider()
            

            st.subheader("📊 Discount vs Bought Scatter")
            
            fig_dd = px.scatter(
                df_dd,
                x='avg_discount_percent',
                y='avg_bought',
                color='segment_str',
                hover_data=['product_name', 'brand_name', 'avg_final_price'],
                title="Discount vs Bought theo Segment",
                labels={'avg_discount_percent': 'Discount (%)', 'avg_bought': 'Bought'},
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_dd.update_layout(height=500)
            st.plotly_chart(fig_dd, use_container_width=True)
            
            st.divider()
            

            st.subheader("📋 Segment Profile")
            
            segment_summary = df_dd.groupby('segment').agg({
                'product_id': 'count',
                'avg_final_price': ['mean', 'median'],
                'avg_discount_percent': 'mean',
                'avg_bought': ['mean', 'median'],
                'stockout_rate': 'mean'
            }).round(2)
            segment_summary.columns = ['Count', 'Price_Mean', 'Price_Median', 
                                       'Discount_Mean', 'Bought_Mean', 'Bought_Median', 'Stockout_Mean']
            
            # Determine segment type
            overall_discount = df_dd['avg_discount_percent'].mean()
            overall_bought = df_dd['avg_bought'].mean()
            
            segment_types = []
            strategies = []
            
            for seg in segment_summary.index:
                disc = segment_summary.loc[seg, 'Discount_Mean']
                bought = segment_summary.loc[seg, 'Bought_Mean']
                
                if disc > overall_discount and bought > overall_bought:
                    segment_types.append("🎯 Nhạy cảm discount")
                    strategies.append("Duy trì discount khi cần push sales")
                elif disc < overall_discount and bought > overall_bought:
                    segment_types.append("💎 Ít nhạy cảm (Pricing Power)")
                    strategies.append("CÓ THỂ TĂNG GIÁ 5-15%")
                elif disc > overall_discount and bought < overall_bought:
                    segment_types.append("⚠️ Discount không hiệu quả")
                    strategies.append("GIẢM DISCOUNT, đổi chiến lược")
                else:
                    segment_types.append("📊 Trung bình")
                    strategies.append("A/B Test giá")
            
            segment_summary['Type'] = segment_types
            segment_summary['Strategy'] = strategies
            
            st.dataframe(segment_summary, use_container_width=True)
            
            st.divider()
            

            st.subheader("📝 Actionable Insights")
            
            # Find products with pricing power
            pricing_power_seg = df_dd[
                (df_dd['avg_discount_percent'] < overall_discount) &
                (df_dd['avg_bought'] > overall_bought) &
                (df_dd['stockout_rate'] < 0.3)
            ]
            
            # Find ineffective discount products
            ineffective_seg = df_dd[
                (df_dd['avg_discount_percent'] > overall_discount) &
                (df_dd['avg_bought'] < overall_bought)
            ]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**💎 Có Pricing Power: {len(pricing_power_seg)} sản phẩm**")
                if len(pricing_power_seg) > 0:
                    show_cols = ['product_name', 'brand_name', 'avg_final_price', 
                                'avg_discount_percent', 'avg_bought']
                    st.dataframe(pricing_power_seg[show_cols].head(15), use_container_width=True, hide_index=True)
                    st.caption("💡 Bán tốt mà không cần discount nhiều → Có thể tăng giá")
            
            with col2:
                st.warning(f"**⚠️ Discount Không Hiệu Quả: {len(ineffective_seg)} sản phẩm**")
                if len(ineffective_seg) > 0:
                    show_cols = ['product_name', 'brand_name', 'avg_final_price', 
                                'avg_discount_percent', 'avg_bought']
                    st.dataframe(ineffective_seg[show_cols].head(15), use_container_width=True, hide_index=True)
                    st.caption("💡 Discount cao nhưng không tăng sales → Cần đổi chiến lược")

# TAB 3: RULE-BASED SENTIMENT
with tab3:
    st.header("💬 Price Sentiment Analysis (Rule-Based)")
    st.markdown("""
    *Phân tích cảm xúc khách hàng về giá từ reviews bằng keyword matching.*
    
    **Cải tiến:**
    - ✅ Xử lý negation: "không đắt" ≠ negative
    - ✅ Ratio-based ranking thay vì count
    - ✅ Threshold tối thiểu mentions
    - ✅ Price mention rate metric
    """)
    
    st.divider()
    
    # Load reviews with text
    @st.cache_data(ttl=3600)
    def prepare_reviews_sentiment():
        reviews = reviews_raw.copy()
        products = df_feat[['product_id', 'product_name', 'brand_name', 'avg_final_price']].copy()
        reviews = reviews.merge(products, on='product_id', how='left')
        reviews = reviews[reviews['review_content'].notna() & (reviews['review_content'].str.len() > 10)]
        return reviews
    
    reviews_df = prepare_reviews_sentiment()
    st.success(f"✅ Đã tải {len(reviews_df):,} reviews có nội dung")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        min_mentions = st.slider("Minimum price mentions", 3, 20, 5, key="t3_min_mentions")
    with col2:
        run_sent = st.button("🚀 Phân Tích Price Sentiment", type="primary", key="t3_run")
    
    # Show keywords
    with st.expander("🔑 Từ khóa sử dụng"):
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Positive:**")
            st.write("xứng đáng, xứng tiền, rẻ, hợp lý, giá tốt, giá rẻ, đáng tiền, phải chăng, tiết kiệm, hời, giá mềm, giá ok")
        with col2:
            st.error("**Negative:**")
            st.write("đắt, mắc, chát, đắt quá, giá cao, quá đắt, đắt đỏ, không xứng tiền, lãng phí")
        st.info("**Negation handling:** 'không đắt', 'không mắc' → không tính là negative")
    
    if run_sent:
        with st.spinner("Đang phân tích sentiment..."):
            # Clean text
            reviews_df['clean_text'] = reviews_df['review_content'].apply(clean_review_text)
            
            # Classify
            reviews_df['price_sentiment'] = reviews_df['clean_text'].apply(
                lambda x: classify_price_sentiment_rule_based(x, handle_negation=True)
            )
            
            # Aggregate per product
            product_sentiment = aggregate_product_sentiment(
                reviews_df, 
                sentiment_col='price_sentiment',
                min_mentions=min_mentions
            )
            
            # Merge with price info
            product_sentiment = product_sentiment.merge(
                df_feat[['product_id', 'avg_final_price', 'avg_discount_percent', 'stockout_rate']],
                on='product_id', how='left'
            )
            
            st.success("✅ Phân tích hoàn tất!")
            
            st.divider()
            

            st.subheader("📊 Tổng Quan")
            
            sentiment_counts = reviews_df['price_sentiment'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("😊 Positive", sentiment_counts.get('Positive', 0))
            col2.metric("😞 Negative", sentiment_counts.get('Negative', 0))
            col3.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
            col4.metric("📝 No Mention", sentiment_counts.get('No_Mention', 0))
            
            # Pie chart
            price_mentions = reviews_df[reviews_df['price_sentiment'] != 'No_Mention']
            if len(price_mentions) > 0:
                fig_pie = px.pie(
                    values=price_mentions['price_sentiment'].value_counts().values,
                    names=price_mentions['price_sentiment'].value_counts().index,
                    title="Phân bố Sentiment về Giá (chỉ reviews có đề cập)",
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            

            st.subheader("📝 Sản Phẩm Cần Hành Động (Ratio-Based)")
            
            # Filter products with enough mentions
            products_enough = product_sentiment[
                product_sentiment['price_mentions_count'] >= min_mentions
            ].copy()
            
            st.info(f"📊 {len(products_enough)} sản phẩm có ≥{min_mentions} price mentions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚠️ CẦN GIẢM GIÁ (High Negative Ratio)**")
                neg_products = products_enough.nlargest(15, 'negative_ratio')
                if len(neg_products) > 0:
                    show_cols = ['product_name', 'brand_name', 'avg_final_price', 
                                'negative_ratio', 'positive_ratio', 'price_mention_rate']
                    neg_display = neg_products[show_cols].copy()
                    neg_display['negative_ratio'] = neg_display['negative_ratio'].apply(lambda x: f"{x:.1%}")
                    neg_display['positive_ratio'] = neg_display['positive_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    neg_display['price_mention_rate'] = neg_display['price_mention_rate'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(neg_display, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**💎 CÓ THỂ TĂNG GIÁ (High Positive Ratio)**")
                pos_products = products_enough.nlargest(15, 'positive_ratio')
                if len(pos_products) > 0:
                    show_cols = ['product_name', 'brand_name', 'avg_final_price', 
                                'positive_ratio', 'negative_ratio', 'price_mention_rate']
                    pos_display = pos_products[show_cols].copy()
                    pos_display['positive_ratio'] = pos_display['positive_ratio'].apply(lambda x: f"{x:.1%}")
                    pos_display['negative_ratio'] = pos_display['negative_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    pos_display['price_mention_rate'] = pos_display['price_mention_rate'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(pos_display, use_container_width=True, hide_index=True)
            
            st.divider()
            

            st.subheader("📄 Sample Reviews")
            
            tab_pos, tab_neg = st.tabs(["😊 Positive", "😞 Negative"])
            
            with tab_pos:
                pos_reviews = price_mentions[price_mentions['price_sentiment'] == 'Positive'].head(5)
                for _, row in pos_reviews.iterrows():
                    with st.expander(f"⭐ {row.get('rating_star', 'N/A')}/5 - {str(row.get('product_name', ''))[:40]}..."):
                        st.write(f"**Giá:** {row.get('avg_final_price', 0):,.0f}đ")
                        st.write(f"**Review:** {row['review_content']}")
            
            with tab_neg:
                neg_reviews = price_mentions[price_mentions['price_sentiment'] == 'Negative'].head(5)
                for _, row in neg_reviews.iterrows():
                    with st.expander(f"⭐ {row.get('rating_star', 'N/A')}/5 - {str(row.get('product_name', ''))[:40]}..."):
                        st.write(f"**Giá:** {row.get('avg_final_price', 0):,.0f}đ")
                        st.write(f"**Review:** {row['review_content']}")

# TAB 4: ML SENTIMENT (PRE-TRAINED)
with tab4:
    st.header("🤖 ML Sentiment Analysis (Pre-trained)")
    st.markdown("""
    *Phân loại sentiment bằng model đã train sẵn cho tiếng Việt - KHÔNG cần train!*
    
    **Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` (hỗ trợ 8 ngôn ngữ kể cả Việt)
    """)
    
    st.divider()
    

    st.subheader("📦 Dependencies Status")
    
    torch_available = False
    transformers_available = False
    
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
    except ImportError:
        torch_version = "Not installed"
    
    try:
        import transformers
        transformers_available = True
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "Not installed"
    
    col1, col2 = st.columns(2)
    with col1:
        if torch_available:
            st.success(f"✅ PyTorch: {torch_version}")
        else:
            st.error("❌ PyTorch: Not installed")
    with col2:
        if transformers_available:
            st.success(f"✅ Transformers: {transformers_version}")
        else:
            st.error("❌ Transformers: Not installed")
    
    if not (torch_available and transformers_available):
        st.error("""
        ⚠️ **Thiếu thư viện!**
        
        Cài đặt:
        ```bash
        pip install torch transformers sentencepiece
        ```
        """)
        st.stop()
    
    st.divider()
    

    st.subheader("📥 Load Pre-trained Model")
    
    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    
    @st.cache_resource
    def load_pretrained_sentiment():
        """Load pre-trained multilingual sentiment model"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        return tokenizer, model
    
    try:
        with st.spinner("Đang load model (lần đầu sẽ download ~1GB)..."):
            tokenizer, model = load_pretrained_sentiment()
        st.success(f"✅ Model loaded: `{MODEL_NAME}`")
        st.info("""
        **Về model này:**
        - XLM-RoBERTa fine-tuned trên 198M tweets
        - Hỗ trợ 8 ngôn ngữ: AR, EN, FR, DE, HI, IT, SP, PT (và hoạt động tốt với Việt)
        - 3 classes: Negative (0), Neutral (1), Positive (2)
        - **KHÔNG cần train thêm!**
        """)
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        model_loaded = False
        st.stop()
    
    st.divider()
    

    st.subheader("🔮 Predict Sentiment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_reviews = st.slider("Min reviews per product", 3, 20, 5, key="pt_min_reviews")
    with col2:
        neg_threshold = st.slider("Neg ratio threshold", 0.3, 0.8, 0.5, key="pt_neg_thresh")
    with col3:
        pos_threshold = st.slider("Pos ratio threshold", 0.3, 0.8, 0.5, key="pt_pos_thresh")
    
    predict_button = st.button("🚀 Predict All Reviews", type="primary", key="pt_predict")
    
    if predict_button and model_loaded:
        import torch
        
        # Prepare reviews
        predict_reviews = reviews_raw.copy()
        predict_reviews = predict_reviews[
            predict_reviews['review_content'].notna() & 
            (predict_reviews['review_content'].str.len() > 10)
        ].copy()
        
        # Merge with product info
        predict_reviews = predict_reviews.merge(
            df_feat[['product_id', 'product_name', 'brand_name', 
                    'avg_final_price', 'avg_discount_percent', 'stockout_rate']],
            on='product_id', how='left'
        )
        
        st.write(f"📊 Predicting {len(predict_reviews)} reviews...")
        
        # Batch prediction
        ID2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        predictions = []
        confidences = []
        batch_size = 16
        
        progress = st.progress(0)
        
        with st.spinner("Đang predict..."):
            for i in range(0, len(predict_reviews), batch_size):
                batch = predict_reviews.iloc[i:i+batch_size]
                texts = [clean_review_text(str(t))[:512] for t in batch['review_content'].tolist()]
                
                try:
                    inputs = tokenizer(
                        texts, truncation=True, max_length=128,
                        padding=True, return_tensors='pt'
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        preds = outputs.logits.argmax(-1).cpu().numpy()
                        confs = probs.max(dim=-1).values.cpu().numpy()
                    
                    predictions.extend([ID2LABEL[p] for p in preds])
                    confidences.extend(confs.tolist())
                except Exception as e:
                    # Fallback for failed batches
                    predictions.extend(['Neutral'] * len(batch))
                    confidences.extend([0.5] * len(batch))
                
                progress.progress(min(1.0, (i + batch_size) / len(predict_reviews)))
        
        predict_reviews['ml_sentiment'] = predictions
        predict_reviews['confidence'] = confidences
        
        st.success(f"✅ Predicted {len(predict_reviews)} reviews!")
        

        st.divider()
        st.subheader("📊 Overall Sentiment Distribution")
        
        sent_counts = predict_reviews['ml_sentiment'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("😊 Positive", sent_counts.get('Positive', 0), 
                   f"{sent_counts.get('Positive', 0)/len(predict_reviews)*100:.1f}%")
        col2.metric("😐 Neutral", sent_counts.get('Neutral', 0),
                   f"{sent_counts.get('Neutral', 0)/len(predict_reviews)*100:.1f}%")
        col3.metric("😞 Negative", sent_counts.get('Negative', 0),
                   f"{sent_counts.get('Negative', 0)/len(predict_reviews)*100:.1f}%")
        
        # Pie chart
        fig_pie = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            title="ML Sentiment Distribution",
            color=sent_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        

        st.divider()
        st.subheader("📦 Product-Level Sentiment")
        
        product_agg = predict_reviews.groupby('product_id').agg({
            'product_name': 'first',
            'brand_name': 'first',
            'avg_final_price': 'first',
            'avg_discount_percent': 'first',
            'stockout_rate': 'first',
            'ml_sentiment': [
                ('total', 'count'),
                ('pos_count', lambda x: (x == 'Positive').sum()),
                ('neg_count', lambda x: (x == 'Negative').sum()),
                ('neu_count', lambda x: (x == 'Neutral').sum()),
            ]
        })
        product_agg.columns = ['product_name', 'brand_name', 'avg_final_price', 
                              'avg_discount_percent', 'stockout_rate',
                              'total', 'pos_count', 'neg_count', 'neu_count']
        product_agg = product_agg.reset_index()
        
        # Calculate ratios
        product_agg['pos_ratio'] = np.where(
            product_agg['total'] >= min_reviews,
            product_agg['pos_count'] / product_agg['total'],
            np.nan
        )
        product_agg['neg_ratio'] = np.where(
            product_agg['total'] >= min_reviews,
            product_agg['neg_count'] / product_agg['total'],
            np.nan
        )
        
        # Filter products with enough reviews
        product_enough = product_agg[product_agg['total'] >= min_reviews].copy()
        st.info(f"📊 {len(product_enough)} products có ≥{min_reviews} reviews")
        

        st.divider()
        st.subheader("💰 Pricing Action Lists")
        
        stockout_threshold = 0.3
        
        # 1. SUPPLY CONSTRAINT FIRST
        supply_constrained = product_enough[
            product_enough['stockout_rate'] > stockout_threshold
        ].sort_values('neg_ratio', ascending=False).head(20)
        
        # 2. GIẢM GIÁ (high neg_ratio + good stock)
        reduce_price = product_enough[
            (product_enough['neg_ratio'] >= neg_threshold) &
            (product_enough['stockout_rate'] <= stockout_threshold)
        ].sort_values('neg_ratio', ascending=False).head(20)
        
        # 3. TĂNG GIÁ (high pos_ratio + low discount + good stock)
        median_discount = product_enough['avg_discount_percent'].median()
        increase_price = product_enough[
            (product_enough['pos_ratio'] >= pos_threshold) &
            (product_enough['avg_discount_percent'] < median_discount) &
            (product_enough['stockout_rate'] <= stockout_threshold)
        ].sort_values('pos_ratio', ascending=False).head(20)
        
        display_cols = ['product_name', 'brand_name', 'avg_final_price', 
                       'avg_discount_percent', 'stockout_rate', 'total', 
                       'pos_ratio', 'neg_ratio']
        
        # Display action lists
        st.markdown("**📦 1. FIX SUPPLY FIRST (Stockout cao)**")
        if len(supply_constrained) > 0:
            supply_display = supply_constrained[display_cols].copy()
            supply_display['pos_ratio'] = supply_display['pos_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            supply_display['neg_ratio'] = supply_display['neg_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            st.dataframe(supply_display, use_container_width=True, hide_index=True)
            st.warning(f"⚠️ {len(supply_constrained)} sản phẩm cần restock trước!")
        else:
            st.success("✅ Không có sản phẩm stockout cao")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**⬇️ 2. GIẢM GIÁ (neg_ratio ≥ {neg_threshold:.0%})**")
            if len(reduce_price) > 0:
                red_display = reduce_price[display_cols].copy()
                red_display['pos_ratio'] = red_display['pos_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                red_display['neg_ratio'] = red_display['neg_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(red_display, use_container_width=True, hide_index=True)
            else:
                st.info("Không có sản phẩm phù hợp")
        
        with col2:
            st.markdown(f"**⬆️ 3. TĂNG GIÁ (pos_ratio ≥ {pos_threshold:.0%})**")
            if len(increase_price) > 0:
                inc_display = increase_price[display_cols].copy()
                inc_display['pos_ratio'] = inc_display['pos_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                inc_display['neg_ratio'] = inc_display['neg_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(inc_display, use_container_width=True, hide_index=True)
            else:
                st.info("Không có sản phẩm phù hợp")
        

        st.divider()
        st.subheader("⚖️ So Sánh với Rule-Based")
        
        predict_reviews['rule_sentiment'] = predict_reviews['review_content'].apply(
            lambda x: classify_price_sentiment_rule_based(clean_review_text(str(x)), handle_negation=True)
        )
        
        # Only compare where rule-based found something
        comparable = predict_reviews[predict_reviews['rule_sentiment'] != 'No_Mention']
        
        if len(comparable) > 0:
            agreement = (comparable['rule_sentiment'] == comparable['ml_sentiment']).mean()
            st.metric("Agreement Rate", f"{agreement:.1%}")
            
            crosstab = pd.crosstab(comparable['rule_sentiment'], comparable['ml_sentiment'])
            st.dataframe(crosstab, use_container_width=True)
        
        # Store for other tabs
        st.session_state['ml_predictions'] = predict_reviews
    
    st.divider()
    

    st.subheader("✍️ Thử Predict")
    
    user_text = st.text_area("Nhập review:", "Sản phẩm tốt, giá hợp lý, đáng mua", key="pt_user_input")
    
    if st.button("🔮 Predict", key="pt_single_predict"):
        if model_loaded:
            import torch
            
            ID2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            
            clean = clean_review_text(user_text)
            inputs = tokenizer(clean, return_tensors='pt', truncation=True, max_length=128)
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_id = outputs.logits.argmax(-1).item()
                confidence = probs[0][pred_id].item()
            
            pred = ID2LABEL[pred_id]
            
            if pred == 'Positive':
                st.success(f"😊 **Positive** (confidence: {confidence:.1%})")
            elif pred == 'Negative':
                st.error(f"😞 **Negative** (confidence: {confidence:.1%})")
            else:
                st.info(f"😐 **Neutral** (confidence: {confidence:.1%})")

st.divider()
st.success("✅ Phân tích hoàn tất! Sử dụng insights để tối ưu chiến lược giá.")


    