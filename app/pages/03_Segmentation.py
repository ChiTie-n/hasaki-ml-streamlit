
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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
from ml_utils import load_ml_data, get_pricing_action_candidates

st.set_page_config(page_title="Segmentation", page_icon="🧩", layout="wide")
inject_page_css()

st.title("🧩 Segmentation")

try:
    df_feat, feature_cols, reviews_raw = load_ml_data()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["K-Means Clustering", "Discount–Demand Segmentation"])

# --------------------------------------------------------------------------------
# TAB 1: K-MEANS CLUSTERING (COPIED EXACTLY)
# --------------------------------------------------------------------------------
with tab1:
    st.header("K-Means Product Clustering")
    st.markdown("""
    *Mô hình K-Means được dùng để phân cụm sản phẩm dựa trên vector đặc trưng (giá, discount, bought, rating, tồn kho...).  
    Quy trình: chuẩn hoá dữ liệu (StandardScaler) → thử nhiều K và chọn K tốt nhất bằng Silhouette (tuỳ chọn) → huấn luyện K-Means với K cuối cùng → giảm chiều bằng PCA(2D) để trực quan hoá và phân tích profile từng cụm.  
    Từ thống kê theo cụm (mean/median), hệ thống suy ra nhóm hành vi và gợi ý chiến lược giá; đồng thời ưu tiên ràng buộc tồn kho bằng ngưỡng stockout_rate để tránh khuyến mãi khi nguy cơ thiếu hàng cao.*
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
    
    run_kmeans = st.button("Chạy K-Means Clustering", type="primary", key="t1_run")
    
    # Check if run or if result already exists in session
    if run_kmeans or ('kmeans_result' in st.session_state and 'cluster_col' in st.session_state):
        if run_kmeans:
            with st.spinner("Đang phân cụm..."):
                # Prepare features
                X = df_feat[feature_cols].values
                
                # Handle any remaining NaN
                X = np.nan_to_num(X, nan=0)
                
                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
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
                    
                    st.session_state['kmeans_sil_df'] = sil_df
                    st.session_state['kmeans_best_k'] = best_k
                else:
                    best_k = k_manual
                    st.session_state['kmeans_best_k'] = best_k
                
                # Final clustering
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Add to dataframe
                df_result = df_feat.copy()
                df_result['cluster'] = labels
                df_result['cluster_str'] = df_result['cluster'].astype(str)
                df_result['PC1'] = X_pca[:, 0]
                df_result['PC2'] = X_pca[:, 1]
                
                # Save to session
                st.session_state['kmeans_result'] = df_result
                st.session_state['cluster_col'] = 'cluster'
                
        # DISPLAY RESULTS (from session state)
        if 'kmeans_result' in st.session_state:
            df_result = st.session_state['kmeans_result']
            best_k = st.session_state.get('kmeans_best_k', len(df_result['cluster'].unique()))
            
            # Show Silhouette if available
            if 'kmeans_sil_df' in st.session_state:
                sil_df = st.session_state['kmeans_sil_df']
                st.subheader("Lựa chọn K (Silhouette Analysis)")
                col1, col2 = st.columns([1, 2])  # Give chart more space
                with col1:
                    st.dataframe(sil_df, use_container_width=True, hide_index=True, height=400) # Fixed height
                with col2:
                    fig_sil = px.line(sil_df, x='K', y='Silhouette', markers=True,
                                      title="Silhouette Score vs K")
                    fig_sil.add_vline(x=best_k, line_dash="dash", line_color="red")
                    fig_sil.update_layout(
                        height=400,
                        title_x=0.5,  # Center title
                        title_xanchor='center',
                        plot_bgcolor='white', # White background
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
                        yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
                    )
                    st.plotly_chart(fig_sil, use_container_width=True)
                
                st.success(f"K tốt nhất = {best_k}")
            
            st.divider()
            st.subheader("Kết quả Clustering")
            st.success(f"Đã phân thành {best_k} cụm!")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pca = px.scatter(
                    df_result, x='PC1', y='PC2',
                    color='cluster_str',
                    hover_data=['product_name', 'brand_name', 'avg_final_price', 'avg_bought'],
                    title="Clusters trên không gian PCA",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_pca.update_layout(
                    height=450,
                    title_x=0.5,
                    title_xanchor='center',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
                )
                st.plotly_chart(fig_pca, use_container_width=True)
            
            with col2:
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
                fig_size.update_layout(
                    height=450, 
                    showlegend=False,
                    title_x=0.5,
                    title_xanchor='center',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
                )
                st.plotly_chart(fig_size, use_container_width=True)
            
            st.divider()
            
            st.subheader("Thông tin của từng cụm")
            profile_cols = ['avg_final_price', 'avg_discount_percent', 'avg_bought', 
                           'rating_mean', 'review_count', 'stockout_rate', 'stock_rate']
            
            cluster_summary = df_result.groupby('cluster')[profile_cols].agg(['mean', 'median']).round(2)
            cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
            cluster_summary['count'] = df_result.groupby('cluster').size()
            cluster_summary = cluster_summary.reset_index()
            
            st.dataframe(cluster_summary, use_container_width=True)
            
            st.divider()
            
            st.markdown('### <i class="fa-solid fa-coins" style="color: #1D89E4;"></i>   Chiến Lược Giá (Ưu Tiên Ràng Buộc Tồn Kho)', unsafe_allow_html=True)
            
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
                
                # ƯU TIÊN 1: Ràng buộc nguồn cung (thiếu hàng)
                if avg_stockout > stockout_threshold:
                    cluster_type = "📦 THIẾU HÀNG / NGUỒN CUNG CĂNG"
                    strategy = "ƯU TIÊN BỔ SUNG HÀNG trước khi chạy khuyến mãi"
                    color = "warning"
                    detail = f"Tỷ lệ thiếu hàng cao ({avg_stockout:.1%}). Không nên chạy khuyến mãi cho đến khi đủ hàng."

                # ƯU TIÊN 2: Nhóm cao cấp bán tốt
                elif avg_price > overall_price * 1.2 and avg_bought > overall_bought:
                    cluster_type = "🏆 NHÓM CAO CẤP BÁN TỐT"
                    strategy = "GIỮ GIÁ hoặc TĂNG NHẸ 5–10%"
                    color = "success"
                    detail = f"Giá cao ({avg_price:,.0f}đ) nhưng vẫn bán tốt ({avg_bought:.0f} lượt mua)."

                # ƯU TIÊN 3: Bán chạy dù ít giảm giá
                elif avg_bought > overall_bought * 1.3 and avg_discount < overall_discount:
                    cluster_type = "🔥 BÁN CHẠY (Ít phụ thuộc giảm giá)"
                    strategy = "CÓ DƯ ĐỊA TĂNG GIÁ 5–15%"
                    color = "success"
                    detail = "Bán tốt mà không cần giảm giá nhiều → có sức mạnh định giá."

                # ƯU TIÊN 4: Phụ thuộc giảm giá nhưng hiệu quả thấp
                elif avg_discount > overall_discount * 1.2 and avg_bought < overall_bought:
                    cluster_type = "⚠️ PHỤ THUỘC GIẢM GIÁ"
                    strategy = "GIẢM MỨC DISCOUNT, xem lại chiến lược marketing"
                    color = "warning"
                    detail = f"Giảm giá cao ({avg_discount:.1f}%) nhưng lượt mua thấp ({avg_bought:.0f})."

                # ƯU TIÊN 5: Phân khúc giá rẻ
                elif avg_price < overall_price * 0.7:
                    cluster_type = "💵 PHÂN KHÚC GIÁ THẤP"
                    strategy = "GIỮ GIÁ, tối ưu chi phí"
                    color = "info"
                    detail = f"Nhóm giá thấp ({avg_price:,.0f}đ). Tập trung tăng sản lượng/độ phủ."

                else:
                    cluster_type = "📊 TRUNG BÌNH"
                    strategy = "THEO DÕI & A/B TEST giá"
                    color = "info"
                    detail = "Hiệu năng ở mức trung bình. Có thể thử nghiệm các mức giá khác nhau."

                with st.expander(f"**Cluster {cluster_id}**: {cluster_type} ({len(cluster_data)} sản phẩm)", expanded=True):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Giá TB", f"{avg_price:,.0f}đ")
                    col2.metric("Discount TB", f"{avg_discount:.1f}%")
                    col3.metric("Bought TB", f"{avg_bought:.0f}")
                    col4.metric("Rating TB", f"{avg_rating:.2f}")
                    col5.metric("Stockout Rate", f"{avg_stockout:.1%}")
                    
                    if color == "success":
                        st.success(f"**Chiến lược:** {strategy}")
                    elif color == "warning":
                        st.warning(f"**Chiến lược:** {strategy}")
                    else:
                        st.info(f"**Chiến lược:** {strategy}")
                    
                    st.caption(f"{detail}")
            
            st.divider()
            
            st.subheader("Khuyến nghị")
            
            # Using imported function from ml_utils
            action_candidates = get_pricing_action_candidates(
                df_result,
                stockout_threshold=stockout_threshold
            )
            
            display_cols = ['product_name', 'brand_name', 'avg_final_price', 
                           'avg_discount_percent', 'avg_bought', 'stockout_rate']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔼 Gợi ý: Có dư địa tăng giá**")
                inc_df = action_candidates['increase_price'].head(20)
                if len(inc_df) > 0:
                    st.dataframe(inc_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"💡 {len(inc_df)} sản phẩm bán chạy + discount thấp + stock tốt")
                else:
                    st.info("Không có sản phẩm phù hợp")
            
            with col2:
                st.markdown("**🔽 Gợi ý: Giảm mức khuyến mãi**")
                red_df = action_candidates['reduce_discount'].head(20)
                if len(red_df) > 0:
                    st.dataframe(red_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"💡 {len(red_df)} sản phẩm discount cao nhưng không hiệu quả")
                else:
                    st.info("Không có sản phẩm phù hợp")
            
            with col3:
                st.markdown("**📦 Gợi ý: Cần bổ sung hàng (không khuyến mãi)**")
                rest_df = action_candidates['need_restock'].head(20)
                if len(rest_df) > 0:
                    st.dataframe(rest_df[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"⚠️ {len(rest_df)} sản phẩm stockout cao - cần bổ sung hàng trước")
                else:
                    st.success("✅ Không có sản phẩm stockout cao")

# --------------------------------------------------------------------------------
# TAB 2: DISCOUNT-DEMAND SEGMENTATION (COPIED EXACTLY)
# --------------------------------------------------------------------------------
with tab2:
    st.header("🏷️ Phân nhóm theo khuyến mãi và sức mua")
    st.markdown("""
    *Phân nhóm sản phẩm để mô tả mối quan hệ giữa mức giảm giá và nhu cầu mua (lượt mua).*

    **Logic thực hiện:** Chuẩn hoá lượt mua theo phân khúc giá để so sánh công bằng, sau đó dùng K-Means để gom nhóm theo 2 trục:
    mức giảm giá và lượt mua đã chuẩn hoá.
    **Cách đọc kết quả:**
    - **Giảm giá cao + mua cao:** nhạy cảm giá
    - **Giảm giá thấp + mua cao:** có dư địa tăng giá
    - **Giảm giá cao + mua thấp:** giảm giá không hiệu quả
    """)
    
    st.divider()
    
    # Prepare data
    df_dd = df_feat[(df_feat['avg_final_price'] > 0) & (df_feat['avg_bought'] > 0)].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        n_clusters_dd = st.slider("Số nhóm phân tích", 2, 5, 3, key="t2_k")
    with col2:
        run_dd = st.button("Chạy phân nhóm", type="primary", key="t2_run_new")
    
    # PERISTENCE: Run if button clicked OR results exist
    if run_dd or ('dd_result' in st.session_state and 'dd_k' in st.session_state):
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
                
                # SAVE TO SESSION
                st.session_state['dd_result'] = df_dd
                st.session_state['dd_k'] = n_clusters_dd
                
        # DISPLAY LOGIC
        if 'dd_result' in st.session_state:
            df_dd = st.session_state['dd_result']
            n_clusters_dd = st.session_state.get('dd_k', 3)
            
            st.success(f"Đã phân thành {n_clusters_dd} nhóm!")
            
            st.divider()
            
            st.subheader("Khuyến mãi vs Sức mua (Scatter)")
            
            fig_dd = px.scatter(
                df_dd,
                x='avg_discount_percent',
                y='avg_bought',
                color='segment_str',
                hover_data=['product_name', 'brand_name', 'avg_final_price'],
                title="Discount vs Bought theo Segment",
                labels={'avg_discount_percent': 'Discount (%)', 'avg_bought': 'Bought (Log Scale)'},
                opacity=0.7,
                log_y=True,  # Log scale to handle outliers
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_dd.update_layout(
                height=500,
                title_x=0.5,
                title_xanchor='center',
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
                yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
            )
            st.plotly_chart(fig_dd, use_container_width=True)
            
            st.divider()
            
            st.subheader("Thông tin")
            
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
                    segment_types.append("Nhạy cảm discount")
                    strategies.append("Duy trì discount khi cần push sales")
                elif disc < overall_discount and bought > overall_bought:
                    segment_types.append("Ít nhạy cảm (khả năng giữ giá)")
                    strategies.append("CÓ ROOM TĂNG GIÁ 5-15%")
                elif disc > overall_discount and bought < overall_bought:
                    segment_types.append("Discount không hiệu quả")
                    strategies.append("GIẢM DISCOUNT, đổi chiến lược")
                else:
                    segment_types.append("Trung bình")
                    strategies.append("A/B Test giá")
            
            segment_summary['Type'] = segment_types
            segment_summary['Strategy'] = strategies
            
            st.dataframe(segment_summary, use_container_width=True)
            
            st.divider()
            
            st.subheader("Khuyến nghị")
            
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
                st.success(f"**💎 Có khả năng giữ giá: {len(pricing_power_seg)} sản phẩm**")
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
