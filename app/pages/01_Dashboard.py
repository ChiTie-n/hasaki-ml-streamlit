import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Sklearn for Dashboard Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Add paths for imports
app_dir = Path(__file__).parent.parent
root_dir = app_dir.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(root_dir))

from styles import inject_page_css
from ml_utils import load_ml_data

st.set_page_config(page_title="Executive Dashboard", page_icon="📈", layout="wide")
# Custom CSS for Bento Grid Layout
st.markdown("""
<style>
    /* Card Style - Top KPIs */
    .metric-card {
        background-color: #F0F9FF;
        border: 2px solid #1D89E4;
        border-radius: 12px;
        padding: 20px;
        box-shadow: none;
        text-align: center;
    }
    .metric-label {
        font-size: 18px;
        font-weight: 700;
        color: #475569;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* Remove default Streamlit padding for tighter layout */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Global Background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Styling for st.container(border=True) - Bottom Charts 
       Target only containers inside columns to avoid global wrappers */
    div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: 2px solid #1D89E4 !important;
        box-shadow: none !important;
        border-radius: 12px;
        background-color: white;
        height: 100%; /* Try to force full height if flexbox allows, otherwise content dictates */
    }
</style>
""", unsafe_allow_html=True)

# inject_page_css() # Skip default injector to use custom tight layout

# st.title("Executive Dashboard") # Removed as per user request

try:
    with st.spinner("Loading..."):
        df, feature_cols, reviews_raw = load_ml_data()
except:
    st.stop()

# ==========================================
# 1. KPI CARDS (Custom HTML implementation for better control)
# ==========================================
total_products = len(df)
avg_discount = df['avg_discount_percent'].mean()
stockout_rate = (df['stockout_rate'] > 0.5).mean()
avg_rating = df['rating_mean'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tổng Sản phẩm</div>
        <div class="metric-value">{total_products:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Discount TB</div>
        <div class="metric-value">{avg_discount:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tỷ lệ hết hàng</div>
        <div class="metric-value" style="color: {'#ef4444' if stockout_rate > 0.1 else '#0f172a'}">{stockout_rate:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rating TB</div>
        <div class="metric-value">{avg_rating:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

# Gap removed -> Re-adding small gap for separation
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ==========================================
# 2. ROW 1: 3-COLUMN LAYOUT (Market | Promo | ACTION TABLE)
# ==========================================
# ==========================================
# 2. ROW 1: MARKET & PROMO & SENTIMENT
# ==========================================
col_r1_1, col_r1_2, col_r1_3 = st.columns(3)

# --- Col 1: Price Structure ---
with col_r1_1:
    with st.container(border=True):
        st.markdown("##### Cơ Cấu Giá & Discount")
        
        df['price_bin'] = pd.qcut(df['avg_final_price'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Premium'])
        price_dist = df.groupby('price_bin').agg({'avg_discount_percent': 'mean', 'avg_bought': 'mean'}).reset_index()
        
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(
            x=price_dist['price_bin'], y=price_dist['avg_bought'],
            name='Sức mua', marker_color='#3b82f6', yaxis='y'
        ))
        fig_combo.add_trace(go.Scatter(
            x=price_dist['price_bin'], y=price_dist['avg_discount_percent'],
            name='Discount %', mode='lines+markers', line=dict(color='#ef4444', width=2), yaxis='y2'
        ))
        
        fig_combo.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(showgrid=True, gridcolor='#f8fafc', showticklabels=False),
            yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False),
            paper_bgcolor='white', plot_bgcolor='white',
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_combo, use_container_width=True)

# --- Col 2: Promotion Efficiency ---
with col_r1_2:
    with st.container(border=True):
        st.markdown("##### Hiệu Quả Khuyến Mãi")
        
        fig_scatter = px.scatter(
            df, x='avg_discount_percent', y='avg_bought', color='price_bin',
            labels={'avg_discount_percent': 'Discount', 'avg_bought': 'Bought'},
            log_y=True,
            color_discrete_sequence=px.colors.sequential.Bluered
        )
        fig_scatter.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='white', plot_bgcolor='white',
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='#f8fafc'),
            yaxis=dict(showgrid=True, gridcolor='#f8fafc')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Col 3: Sentiment ---
with col_r1_3:
    with st.container(border=True):
        st.markdown("##### Phân bố Sentiment về Giá")
        
        if 'review_content' in reviews_raw.columns:
            pos_keywords = ['tốt', 'thích', 'ok', 'ưng', 'rẻ', 'hợp lý']
            neg_keywords = ['tệ', 'đắt', 'chán', 'khô', 'mắc']
            def simple_sentiment(text):
                t = str(text).lower()
                if any(w in t for w in pos_keywords): return 'Positive'
                if any(w in t for w in neg_keywords): return 'Negative'
                return 'Neutral'
            
            # Sample for speed if needed, but here we run full
            sent_counts = reviews_raw['review_content'].astype(str).apply(simple_sentiment).value_counts()
            
            fig_pie = px.pie(
                values=sent_counts.values, names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map={'Positive': '#3b82f6', 'Negative': '#ef4444', 'Neutral': '#94a3b8'},
                hole=0.5
            )
            fig_pie.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
                paper_bgcolor='white',
                plot_bgcolor='white',
                legend=dict(orientation="v", y=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No Review Data")

# Spacing
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ==========================================
# 4. ROW 3: ACTION TABLE & CLUSTERS (2:1 Ratio)
# ==========================================
col_r3_1, col_r3_2 = st.columns([2, 1])

# --- Col 1: Action Table (2/3) ---
with col_r3_1:
    with st.container(border=True):
        # Header Row: Title + Filter
        col_header, col_filter = st.columns([3, 1.5], vertical_alignment="bottom")
        with col_header:
            st.markdown("##### Danh sách Cần Lưu Ý")
        with col_filter:
            # Simulate Pricing Logic (moved up to ensure df['recommendation'] exists for filter)
            conditions = [
                (df['stockout_rate'] > 0.5),
                (df['avg_bought'] > df['avg_bought'].quantile(0.7)) & (df['avg_discount_percent'] < 15),
                (df['avg_bought'] < df['avg_bought'].quantile(0.25)) & (df['avg_discount_percent'] < 20),
                (df['avg_bought'] < df['avg_bought'].quantile(0.2)) & (df['avg_discount_percent'] >= 20)
            ]
            choices = ['🔴 Cần Restock', '🟢 CÓ ROOM TĂNG GIÁ', '🟠 XEM XÉT GIẢM GIÁ', '⚪ GIỮ GIÁ / CUT PROMO']
            df['recommendation'] = np.select(conditions, choices, default='⚪ GIỮ GIÁ')
            
            filter_val = st.selectbox("Lọc khuyến nghị:", ["Tất cả"] + sorted(list(df['recommendation'].unique())), label_visibility="collapsed")
        
        # Filter Data
        if filter_val != "Tất cả":
            df_show = df[df['recommendation'] == filter_val]
        else:
            df_show = df
            
        st.dataframe(
            df_show[['product_name', 'recommendation']],
            column_config={
                "product_name": st.column_config.TextColumn("Sản phẩm", width="medium"),
                "recommendation": st.column_config.TextColumn("Hành động", width="medium"),
            },
            use_container_width=True,
            height=300,
            hide_index=True
        )

# --- Col 2: PCA Cluster Chart (1/3) ---
with col_r3_2:
    with st.container(border=True):
        st.markdown("##### Clusters (PCA)")
        
        # Consistent PCA Pipeline - Matching 03_Segmentation.py
        # Use feature_cols from load_ml_data() to ensure same features
        if 'feature_cols' not in locals():
             feature_cols = ['avg_final_price', 'avg_discount_percent', 'avg_bought', 
                             'rating_mean', 'review_count', 'stockout_rate', 'stock_rate']
             
        # Select features and handle NaNs exactly like Segmentation page
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0)
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # PCA Projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['x_pca'] = X_pca[:, 0]
        df['y_pca'] = X_pca[:, 1]
        df['cluster_str'] = df['cluster'].astype(str)
        
        fig_pca = px.scatter(
            df, x='x_pca', y='y_pca',
            color='cluster_str',
            labels={'x_pca': 'PC1', 'y_pca': 'PC2'},
            color_discrete_map={'0': '#3b82f6', '1': '#94a3b8', '2': '#ef4444'},
            # Fallback if more clusters
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_pca.update_layout(
            height=370,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='white', plot_bgcolor='white',
            showlegend=True,
            xaxis=dict(showgrid=True, gridcolor='#f8fafc', showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='#f8fafc', showticklabels=False),
            legend=dict(orientation="v", y=0.5, x=1.0)
        )
        st.plotly_chart(fig_pca, use_container_width=True)


# ==========================================
# 3. ROW 2: STRATEGIC OVERVIEW (Pricing Style Visuals)
# ==========================================
