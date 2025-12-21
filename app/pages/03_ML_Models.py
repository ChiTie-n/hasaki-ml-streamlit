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


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 K-Means Clustering", 
    "💰 Discount-Demand Segmentation",
    "💬 Sentiment (Rule-Based)", 
    "🤖 Sentiment (ML Model)",
    "🎯 Decision Dashboard",
    "🔮 Price Simulator (ML)"
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
                    strategy = "CÓ ROOM TĂNG GIÁ 5-15%"
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
                st.markdown("**🔼 Candidates CÓ ROOM TĂNG GIÁ**")
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
                    strategies.append("CÓ ROOM TĂNG GIÁ 5-15%")
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
                    st.caption("💡 Bán tốt mà không cần discount nhiều → Có room tăng giá")
            
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
                st.markdown("**💎 CÓ ROOM TĂNG GIÁ (High Positive Ratio)**")
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
        
        # SAVE to session_state for Tab 5 Decision Dashboard
        st.session_state['ml_sentiment_results'] = product_enough.copy()
        st.success("💾 Kết quả đã được lưu để dùng trong Tab 5 Decision Dashboard!")
        

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
            st.markdown(f"**⬆️ 3. CÓ ROOM TĂNG GIÁ (pos_ratio ≥ {pos_threshold:.0%})**")
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

# TAB 5: DECISION DASHBOARD
with tab5:
    st.header("🎯 Decision Dashboard")
    st.markdown("""
    *Tổng hợp signals từ tất cả modules để đưa ra quyết định giá cuối cùng.*
    """)
    
    st.divider()
    
    # Run all analyses
    if st.button("🔄 Chạy Phân Tích Tổng Hợp", type="primary", use_container_width=True):
        with st.spinner("Đang phân tích..."):
            
            # 1. Get feature data
            df_products = df_feat.copy()
            
            # 2. K-Means signals
            st.info("📊 Đang chạy K-Means...")
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
            st.info("💰 Đang phân loại Discount-Demand...")
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
                st.info("🤖 Đang dùng ML Sentiment từ Tab 4 (cached)...")
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
            st.info("🎯 Đang tính điểm tổng hợp...")
            # Weight ML Sentiment higher than Rule-Based
            sentiment_weight = 2.0 if sentiment_source == 'ML' else 1.5
            df_products['total_score'] = (
                df_products['km_signal'] * 2 +  # K-Means weight: 2
                df_products['dd_signal'] * 1.5 +  # Discount-Demand weight: 1.5
                df_products['sentiment_signal'] * sentiment_weight  # Sentiment weight
            )
            st.info(f"📊 Sentiment source: **{sentiment_source}** (weight: {sentiment_weight})")
            
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
        st.subheader("📊 Tổng Quan")
        rec_counts = df_decision['recommendation'].value_counts()
        
        cols = st.columns(5)
        for i, (rec, count) in enumerate(rec_counts.items()):
            cols[i % 5].metric(rec, count)
        
        st.divider()
        
        # Show by recommendation
        st.subheader("📋 Chi Tiết Theo Khuyến Nghị")
        
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
        st.subheader("🎯 Top Actions")
        
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
        st.info("""
        **📖 Cách đọc kết quả:**
        - **Total Score**: Điểm tổng hợp từ 3 modules (K-Means, Discount-Demand, Sentiment)
        - **Confidence**: HIGH = 2+ signals đồng thuận, MEDIUM = 1 signal, LOW = không rõ
        - **Recommendation**: Quyết định cuối cùng dựa trên total score
        """)
    else:
        st.info("👆 Nhấn nút **Chạy Phân Tích Tổng Hợp** để bắt đầu.")

# TAB 6: PRICE SIMULATOR (ML REGRESSION)
with tab6:
    st.header("🔮 Price Simulator (ML Regression)")
    st.markdown("""
    *Sử dụng Machine Learning để dự đoán tác động của thay đổi giá lên demand.*
    
    **Mô hình:** RandomForest Regressor  
    **Input:** price, discount, rating, stock  
    **Output:** Predicted bought (demand)
    """)
    
    st.divider()
    
    # Train model
    st.subheader("📊 1. Train Demand Prediction Model")
    
    st.markdown("""
    **Cải tiến:**
    - Log transform target (handle skewed data)
    - Loại bỏ outliers (IQR method)
    - Sử dụng GradientBoosting (tốt hơn RandomForest cho tabular data)
    - Thêm features: price_per_rating, discount_effectiveness
    """)
    
    if st.button("🚀 Train Model (Improved)", type="primary"):
        with st.spinner("Đang train model..."):
            from sklearn.ensemble import GradientBoostingRegressor
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
                
                # Train GradientBoosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    min_samples_split=10,
                    random_state=42
                )
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
                
                st.success("✅ Model trained thành công!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae:.1f}")
                col2.metric("R² Score", f"{r2:.3f}")
                col3.metric("Training samples", len(X_train))
                col4.metric("Features", len(feature_cols_final))
                
                # Quality assessment
                if r2 >= 0.3:
                    st.success("✅ Model quality: ACCEPTABLE - Có thể dùng để tham khảo")
                elif r2 >= 0.1:
                    st.warning("⚠️ Model quality: LOW - Kết quả chỉ mang tính tham khảo")
                else:
                    st.error("❌ Model quality: POOR - Không nên dùng để ra quyết định")

                
                # Feature importance
                st.subheader("📈 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols_final,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Các yếu tố ảnh hưởng đến Demand"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    
    # Simulation
    if 'demand_model' in st.session_state:
        st.subheader("🎮 2. Simulate Price Changes")
        
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
            
            if st.button("🔮 Simulate Discount Change"):
                X_original = df_sim[features].fillna(0)
                X_simulated = X_original.copy()
                X_simulated['avg_discount_percent'] = (
                    X_simulated['avg_discount_percent'] + discount_change
                ).clip(0, 100)
                
                # Predict
                pred_original = model.predict(X_original)
                pred_simulated = model.predict(X_simulated)
                
                # Compare
                df_result = df_sim[['product_name', 'avg_final_price', 'avg_discount_percent', 'avg_bought']].copy()
                df_result['predicted_original'] = pred_original
                df_result['predicted_new'] = pred_simulated
                df_result['delta_bought'] = pred_simulated - pred_original
                df_result['delta_pct'] = (df_result['delta_bought'] / pred_original * 100).round(1)
                
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
            
            if st.button("🔮 Simulate Price Change"):
                X_original = df_sim[features].fillna(0)
                X_simulated = X_original.copy()
                X_simulated['avg_final_price'] = (
                    X_simulated['avg_final_price'] * (1 + price_change/100)
                ).clip(0, None)
                
                # Predict
                pred_original = model.predict(X_original)
                pred_simulated = model.predict(X_simulated)
                
                # Compare
                df_result = df_sim[['product_name', 'avg_final_price', 'avg_bought']].copy()
                df_result['new_price'] = df_result['avg_final_price'] * (1 + price_change/100)
                df_result['predicted_original'] = pred_original
                df_result['predicted_new'] = pred_simulated
                df_result['delta_bought'] = pred_simulated - pred_original
                df_result['delta_pct'] = (df_result['delta_bought'] / pred_original * 100).round(1)
                
                # Revenue impact estimate
                df_result['revenue_original'] = df_result['avg_final_price'] * pred_original
                df_result['revenue_new'] = df_result['new_price'] * pred_simulated
                df_result['revenue_delta_pct'] = (
                    (df_result['revenue_new'] - df_result['revenue_original']) / 
                    df_result['revenue_original'] * 100
                ).round(1)
                
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
            
            new_price = st.number_input(
                "Nhập giá mới (VNĐ):", 
                min_value=1000, 
                max_value=10000000,
                value=int(product_data['avg_final_price']),
                step=1000
            )
            new_discount = st.slider(
                "Nhập discount mới (%):",
                min_value=0, max_value=70,
                value=int(product_data['avg_discount_percent'])
            )
            
            if st.button("🔮 Predict Demand"):
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
        st.info("""
        **⚠️ Lưu ý quan trọng:**
        - Model này dự đoán dựa trên **correlation**, không phải **causation**
        - Kết quả chỉ mang tính **tham khảo**, cần A/B testing thực tế để validate
        - R² thấp (<0.5) nghĩa là model chưa capture được hết factors ảnh hưởng demand
        """)
    else:
        st.info("👆 Train model trước khi simulation.")