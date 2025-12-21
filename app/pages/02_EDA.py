"""
EDA - Exploratory Data Analysis (6 outputs tối giản)
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add paths for imports
app_dir = Path(__file__).parent.parent
root_dir = app_dir.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(root_dir))

from src.data_loader import (
    load_dim_product,
    load_fact_prices,
    load_fact_inventory,
    load_dim_category,
)

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

# Inject styles
from styles import inject_page_css
inject_page_css()

st.markdown('<h1><i class="fa-solid fa-chart-bar" style="color: #3b82f6; margin-right: 0.5rem;"></i> Phân Tích Dữ Liệu</h1>', unsafe_allow_html=True)
st.markdown("### 6 Phân Tích Chính")

@st.cache_data(ttl=3600)
def load_and_prepare_data():
    """Load và prepare dữ liệu"""
    with st.spinner("Đang tải dữ liệu..."):
        products = load_dim_product(limit=5000)
        prices = load_fact_prices(limit=10000)
        inventory = load_fact_inventory(limit=10000)
        categories = load_dim_category(limit=500)
        
        # Aggregate prices
        price_agg = prices.groupby('product_id').agg({
            'final_price': 'mean',
            'discount_percent': 'mean',
            'bought': 'sum'
        }).reset_index()
        
        # Aggregate inventory
        inv_agg = inventory.groupby('product_id').agg({
            'stock_available': 'mean',
            'total_branches': 'max'
        }).reset_index()
        inv_agg['stock_rate'] = inv_agg['stock_available'] / (inv_agg['total_branches'].replace(0, 1))
        
        # Merge all
        df = products[['product_id', 'product_name', 'brand_name', 'category_id']].copy()
        df = df.merge(price_agg, on='product_id', how='left')
        df = df.merge(inv_agg[['product_id', 'stock_rate']], on='product_id', how='left')
        
        # Join with category to get name
        if 'category_name' in categories.columns:
            df = df.merge(categories[['category_id', 'category_name']], on='category_id', how='left')
            df['category'] = df['category_name'].fillna('Unknown')
        else:
            # Fallback: use first 8 chars of UUID
            df['category'] = df['category_id'].astype(str).str[:8]
        
        # Fill NA
        df['final_price'] = df['final_price'].fillna(0)
        df['discount_percent'] = df['discount_percent'].fillna(0)
        df['bought'] = df['bought'].fillna(0)
        df['stock_rate'] = df['stock_rate'].fillna(0)
        
        return df

try:
    df = load_and_prepare_data()
    st.markdown(f'<div style="background: #d1fae5; border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px;"><i class="fa-solid fa-circle-check" style="color: #10b981; margin-right: 0.5rem;"></i>Đã tải {len(df)} sản phẩm</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# Filter out zero values, Unknown category, and extreme outliers
df_analysis = df[
    (df['final_price'] > 0) & 
    (df['final_price'] < 10_000_000) &  # Filter extreme prices (< 10M VNĐ)
    (df['bought'] > 0) &
    (df['category'] != 'Unknown')
].copy()


st.divider()

# ==================== OUTPUT 1: Boxplot final_price theo category ====================
st.markdown('<h2><span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; margin-right: 10px;">1</span> Phân bổ giá theo danh mục</h2>', unsafe_allow_html=True)
st.markdown("*Xác định category 'định giá cao/thấp' và outliers*")

# Get top categories by product count
top_categories = df_analysis['category'].value_counts().head(10).index.tolist()
df_cat = df_analysis[df_analysis['category'].isin(top_categories)]

fig1 = px.box(
    df_cat,
    x='category',
    y='final_price',
    title="Phân phối giá theo Category (Top 10 categories)",
    labels={'category': 'Category', 'final_price': 'Final Price (VNĐ)'},
    points='outliers'
)
fig1.update_layout(
    height=500, xaxis_tickangle=-45,
    title_x=0.5,
    title_xanchor='center',
    plot_bgcolor='white',
    paper_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
)
st.plotly_chart(fig1, use_container_width=True)

# Dynamic insights for Chart 1
price_by_cat = df_cat.groupby('category')['final_price'].median().sort_values(ascending=False)
highest_cat = price_by_cat.index[0]
lowest_cat = price_by_cat.index[-1]
st.markdown(f"""
<div style="background-color: #e0f2fe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; color: #1e3a8a; margin-bottom: 1rem;">
    <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong>
    <ul style="margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.5rem;">
        <li><strong>Định giá cao nhất:</strong> {highest_cat} (median: {price_by_cat[highest_cat]:,.0f}đ)</li>
        <li><strong>Định giá thấp nhất:</strong> {lowest_cat} (median: {price_by_cat[lowest_cat]:,.0f}đ)</li>
        <li>Các chấm tròn phía trên là <strong>outliers</strong> (sản phẩm giá cao bất thường trong category)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==================== OUTPUT 2: Boxplot discount_percent theo category ====================
st.markdown('<h2><span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; margin-right: 10px;">2</span> Boxplot Tỷ lệ giảm giá theo danh mục</h2>', unsafe_allow_html=True)  
st.markdown("*Category nào đang phụ thuộc khuyến mãi*")

fig2 = px.box(
    df_cat[df_cat['discount_percent'] > 0],
    x='category',
    y='discount_percent',
    title="Phân phối Discount theo Category (Top 10 categories)",
    labels={'category': 'Category', 'discount_percent': 'Discount (%)'},
    points='outliers'
)
fig2.update_layout(
    height=500, xaxis_tickangle=-45,
    title_x=0.5,
    title_xanchor='center',
    plot_bgcolor='white',
    paper_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
)
st.plotly_chart(fig2, use_container_width=True)

# Dynamic insights for Chart 2
disc_by_cat = df_cat[df_cat['discount_percent'] > 0].groupby('category')['discount_percent'].median().sort_values(ascending=False)
if len(disc_by_cat) > 0:
    high_disc_cat = disc_by_cat.index[0]
    st.markdown(f"""
    <div style="background-color: #e0f2fe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; color: #1e3a8a; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong>
        <ul style="margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.5rem;">
            <li><strong>Phụ thuộc KM nhiều nhất:</strong> {high_disc_cat} (median discount: {disc_by_cat[high_disc_cat]:.1f}%)</li>
            <li>Category có box càng cao = càng phụ thuộc vào khuyến mãi để bán hàng</li>
            <li><strong>Khuyến nghị:</strong> Xem xét giảm discount dần cho các category có median > 30%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==================== OUTPUT 3: Scatter discount_percent vs bought ====================
st.markdown('<h2><span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; margin-right: 10px;">3</span> Scatter: Tỷ lệ giảm giá vs Lượng mua</h2>', unsafe_allow_html=True)
st.markdown("*Giảm giá có kéo cầu không?*")

col1, col2 = st.columns(2)

with col1:
    # Toàn bộ
    fig3a = px.scatter(
        df_analysis,
        x='discount_percent',
        y='bought',
        title="Toàn bộ sản phẩm",
        labels={'discount_percent': 'Discount (%)', 'bought': 'Bought (Log Scale)'},
        opacity=0.6,
        log_y=True
    )
    fig3a.update_layout(
        height=400,
        title_x=0.5,
        title_xanchor='center',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
        yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
    )
    st.plotly_chart(fig3a, use_container_width=True)

with col2:
    # Theo category top
    fig3b = px.scatter(
        df_cat,
        x='discount_percent',
        y='bought',
        color='category',
        title="Theo Top Categories",
        labels={'discount_percent': 'Discount (%)', 'bought': 'Bought (Log Scale)'},
        opacity=0.7,
        log_y=True
    )
    fig3b.update_layout(
        height=400,
        title_x=0.5,
        title_xanchor='center',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
        yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
    )
    st.plotly_chart(fig3b, use_container_width=True)

# Insight for Chart 3
corr_disc_bought = df_analysis['discount_percent'].corr(df_analysis['bought'])
if corr_disc_bought > 0.1:
    st.markdown(f"""<div style="background-color: #dcfce7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; color: #14532d; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan dương ({corr_disc_bought:.2f}) → Giảm giá <strong>có</strong> giúp tăng sales
    </div>""", unsafe_allow_html=True)
elif corr_disc_bought < -0.1:
    st.markdown(f"""<div style="background-color: #fef9c3; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #eab308; color: #713f12; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan âm ({corr_disc_bought:.2f}) → Giảm giá <strong>không</strong> hiệu quả
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div style="background-color: #e0f2fe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; color: #1e3a8a; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan yếu ({corr_disc_bought:.2f}) → Discount có tác động <strong>không rõ ràng</strong> lên sales
    </div>""", unsafe_allow_html=True)

st.divider()

# ==================== OUTPUT 4: Scatter final_price vs bought ====================
st.markdown('<h2><span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; margin-right: 10px;">4</span> Scatter: Giá bán vs Lượng mua</h2>', unsafe_allow_html=True)
st.markdown("*Sản phẩm giá cao có bán được không? (định vị)*")

fig4 = px.scatter(
    df_analysis,
    x='final_price',
    y='bought',
    color='category',
    hover_data=['product_name', 'brand_name', 'discount_percent'],
    title="Mối quan hệ Giá - Doanh số",
    labels={'final_price': 'Final Price (VNĐ)', 'bought': 'Bought (Log Scale)'},
    opacity=0.6,
    log_y=True
)
fig4.update_layout(
    height=500,
    title_x=0.5,
    title_xanchor='center',
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
    yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
)
st.plotly_chart(fig4, use_container_width=True)

# Insight for Chart 4
corr_price_bought = df_analysis['final_price'].corr(df_analysis['bought'])
if corr_price_bought < -0.1:
    st.markdown(f"""<div style="background-color: #e0f2fe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; color: #1e3a8a; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan âm ({corr_price_bought:.2f}) → Sản phẩm giá cao <strong>bán ít hơn</strong> (bình thường)
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div style="background-color: #dcfce7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; color: #14532d; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan ({corr_price_bought:.2f}) → Giá cao vẫn bán được! Có thể định vị <strong>premium</strong>
    </div>""", unsafe_allow_html=True)

st.divider()

# ==================== OUTPUT 5: Scatter stock_rate vs bought ====================
st.markdown('<h2><span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; margin-right: 10px;">5</span> Scatter: Tỷ lệ tồn kho vs Lượng mua</h2>', unsafe_allow_html=True)
st.markdown("*Bought có bị 'kẹt' vì thiếu hàng không?*")

df_stock = df_analysis[df_analysis['stock_rate'] > 0].copy()

fig5 = px.scatter(
    df_stock,
    x='stock_rate',
    y='bought',
    color='category',
    hover_data=['product_name', 'brand_name'],
    title="Mối quan hệ Tồn kho - Doanh số",
    labels={'stock_rate': 'Stock Rate (tỷ lệ có hàng)', 'bought': 'Bought (Log Scale)'},
    opacity=0.6,
    log_y=True
)
fig5.update_layout(
    height=500,
    title_x=0.5,
    title_xanchor='center',
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='#f0f2f6'),
    yaxis=dict(showgrid=True, gridcolor='#f0f2f6')
)
st.plotly_chart(fig5, use_container_width=True)

# Insight for Chart 5
corr_stock_bought = df_stock['stock_rate'].corr(df_stock['bought'])
low_stock_count = len(df_stock[df_stock['stock_rate'] < 0.3])
if corr_stock_bought > 0.1:
    st.markdown(f"""
    <div style="background-color: #fef9c3; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #eab308; color: #713f12; margin-bottom: 1rem;">
        <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan dương ({corr_stock_bought:.2f}) → Stock rate thấp = Bought thấp
        <ul style="margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.5rem;">
            <li><strong>{low_stock_count} sản phẩm</strong> có stock_rate < 30% → Có thể đang mất doanh số vì thiếu hàng!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""<div style="background-color: #e0f2fe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; color: #1e3a8a; margin-bottom: 1rem;">
    <strong><i class='fa-solid fa-lightbulb' style='color: #f59e0b; margin-right: 5px;'></i> Insight:</strong> Tương quan ({corr_stock_bought:.2f}) → Stock không phải bottleneck chính
    </div>""", unsafe_allow_html=True)

st.divider()


# ==================== OUTPUT 6: Bảng 3 nhóm sản phẩm ====================
st.markdown('<h2><span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; margin-right: 10px;">6</span> Phân Nhóm Sản Phẩm Chiến Lược (Top 20 mỗi nhóm)</h2>', unsafe_allow_html=True)

# Nhóm A: bought cao, discount thấp (ứng viên tăng giá nhẹ)
st.subheader("🔥 Nhóm A: Bán chạy + Discount thấp → **Ứng viên TĂNG GIÁ NHẸ**")
group_a = df_analysis[
    (df_analysis['bought'] > df_analysis['bought'].quantile(0.7)) &
    (df_analysis['discount_percent'] < df_analysis['discount_percent'].quantile(0.3))
].nlargest(20, 'bought')[['product_name', 'brand_name', 'category', 'bought', 'discount_percent', 'final_price']]

st.dataframe(group_a, use_container_width=True, hide_index=True)
st.caption(f"💡 **Insight:** {len(group_a)} sản phẩm bán chạy mà không cần giảm giá nhiều → Có thể test tăng giá nhẹ")

st.divider()

# Nhóm B: discount cao, bought không tăng (ứng viên cắt khuyến mãi/đổi chiến thuật)
st.subheader("⚠️ Nhóm B: Discount cao + Bought thấp → **Ứng viên CẮT KHUYẾN MÃI / ĐỔI CHIẾN THUẬT**")
group_b = df_analysis[
    (df_analysis['discount_percent'] > df_analysis['discount_percent'].quantile(0.7)) &
    (df_analysis['bought'] < df_analysis['bought'].quantile(0.3))
].nlargest(20, 'discount_percent')[['product_name', 'brand_name', 'category', 'bought', 'discount_percent', 'final_price']]

st.dataframe(group_b, use_container_width=True, hide_index=True)
st.caption(f"💡 **Insight:** {len(group_b)} sản phẩm giảm giá nhiều nhưng vẫn không bán → Cần đổi chiến lược marketing hoặc ngừng khuyến mãi")

st.divider()

# Nhóm C: bought cao, stock_rate thấp (ứng viên ưu tiên tồn kho trước khi promo)
st.subheader("📦 Nhóm C: Bán chạy + Stock thấp → **Ứng viên ƯU TIÊN TỒN KHO trước khi PROMO**")
group_c = df_analysis[
    (df_analysis['bought'] > df_analysis['bought'].quantile(0.7)) &
    (df_analysis['stock_rate'] < df_analysis['stock_rate'].quantile(0.3))
].nlargest(20, 'bought')[['product_name', 'brand_name', 'category', 'bought', 'stock_rate', 'final_price']]

st.dataframe(group_c, use_container_width=True, hide_index=True)
st.caption(f"💡 **Insight:** {len(group_c)} sản phẩm bán chạy nhưng tồn kho thấp → Cần restock trước khi chạy promotion để tận dụng tối đa")

st.divider()

# ==================== SUMMARY STATS ====================
st.header("Tổng Kết")

col1, col2, col3 = st.columns(3)

col1.metric("Nhóm A (Tăng giá)", len(group_a), "sản phẩm")
col2.metric("Nhóm B (Cắt KM)", len(group_b), "sản phẩm")
col3.metric("Nhóm C (Ưu tiên stock)", len(group_c), "sản phẩm")

st.success("EDA hoàn tất với 6 outputs chính!")
