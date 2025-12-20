"""
EDA - Exploratory Data Analysis (6 outputs tối giản)
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import (
    load_dim_product,
    load_fact_prices,
    load_fact_inventory,
    load_dim_category,
)

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
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
    st.success(f"✅ Đã tải {len(df)} sản phẩm")
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
st.header("1️⃣ Boxplot Final Price theo Category")
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
fig1.update_layout(height=500, xaxis_tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)

# Dynamic insights for Chart 1
price_by_cat = df_cat.groupby('category')['final_price'].median().sort_values(ascending=False)
highest_cat = price_by_cat.index[0]
lowest_cat = price_by_cat.index[-1]
st.info(f"""
**📊 Insight:**
- **Định giá cao nhất:** {highest_cat} (median: {price_by_cat[highest_cat]:,.0f}đ)
- **Định giá thấp nhất:** {lowest_cat} (median: {price_by_cat[lowest_cat]:,.0f}đ)
- Các chấm tròn phía trên là **outliers** (sản phẩm giá cao bất thường trong category)
""")

st.divider()


# ==================== OUTPUT 2: Boxplot discount_percent theo category ====================
st.header("2️⃣ Boxplot Discount Percent theo Category")  
st.markdown("*Category nào đang phụ thuộc khuyến mãi*")

fig2 = px.box(
    df_cat[df_cat['discount_percent'] > 0],
    x='category',
    y='discount_percent',
    title="Phân phối Discount theo Category (Top 10 categories)",
    labels={'category': 'Category', 'discount_percent': 'Discount (%)'},
    points='outliers'
)
fig2.update_layout(height=500, xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

# Dynamic insights for Chart 2
disc_by_cat = df_cat[df_cat['discount_percent'] > 0].groupby('category')['discount_percent'].median().sort_values(ascending=False)
if len(disc_by_cat) > 0:
    high_disc_cat = disc_by_cat.index[0]
    st.info(f"""
**📊 Insight:**
- **Phụ thuộc KM nhiều nhất:** {high_disc_cat} (median discount: {disc_by_cat[high_disc_cat]:.1f}%)
- Category có box càng cao = càng phụ thuộc vào khuyến mãi để bán hàng
- **Khuyến nghị:** Xem xét giảm discount dần cho các category có median > 30%
""")

st.divider()

# ==================== OUTPUT 3: Scatter discount_percent vs bought ====================
st.header("3️⃣ Scatter: Discount Percent vs Bought")
st.markdown("*Giảm giá có kéo cầu không?*")

col1, col2 = st.columns(2)

with col1:
    # Toàn bộ
    fig3a = px.scatter(
        df_analysis,
        x='discount_percent',
        y='bought',
        title="Toàn bộ sản phẩm",
        labels={'discount_percent': 'Discount (%)', 'bought': 'Bought'},
        opacity=0.6
    )
    fig3a.update_layout(height=400)
    st.plotly_chart(fig3a, use_container_width=True)

with col2:
    # Theo category top
    fig3b = px.scatter(
        df_cat,
        x='discount_percent',
        y='bought',
        color='category',
        title="Theo Top Categories",
        labels={'discount_percent': 'Discount (%)', 'bought': 'Bought'},
        opacity=0.7
    )
    fig3b.update_layout(height=400)
    st.plotly_chart(fig3b, use_container_width=True)

# Insight for Chart 3
corr_disc_bought = df_analysis['discount_percent'].corr(df_analysis['bought'])
if corr_disc_bought > 0.1:
    st.success(f"**📊 Insight:** Tương quan dương ({corr_disc_bought:.2f}) → Giảm giá **có** giúp tăng sales")
elif corr_disc_bought < -0.1:
    st.warning(f"**📊 Insight:** Tương quan âm ({corr_disc_bought:.2f}) → Giảm giá **không** hiệu quả")
else:
    st.info(f"**📊 Insight:** Tương quan yếu ({corr_disc_bought:.2f}) → Discount có tác động **không rõ ràng** lên sales")

st.divider()

# ==================== OUTPUT 4: Scatter final_price vs bought ====================
st.header("4️⃣ Scatter: Final Price vs Bought")
st.markdown("*Sản phẩm giá cao có bán được không? (định vị)*")

fig4 = px.scatter(
    df_analysis,
    x='final_price',
    y='bought',
    color='category',
    hover_data=['product_name', 'brand_name', 'discount_percent'],
    title="Mối quan hệ Giá - Doanh số",
    labels={'final_price': 'Final Price (VNĐ)', 'bought': 'Bought'},
    opacity=0.6
)
fig4.update_layout(height=500)
st.plotly_chart(fig4, use_container_width=True)

# Insight for Chart 4
corr_price_bought = df_analysis['final_price'].corr(df_analysis['bought'])
if corr_price_bought < -0.1:
    st.info(f"**📊 Insight:** Tương quan âm ({corr_price_bought:.2f}) → Sản phẩm giá cao **bán ít hơn** (bình thường)")
else:
    st.success(f"**📊 Insight:** Tương quan ({corr_price_bought:.2f}) → Giá cao vẫn bán được! Có thể định vị **premium**")

st.divider()

# ==================== OUTPUT 5: Scatter stock_rate vs bought ====================
st.header("5️⃣ Scatter: Stock Rate vs Bought")
st.markdown("*Bought có bị 'kẹt' vì thiếu hàng không?*")

df_stock = df_analysis[df_analysis['stock_rate'] > 0].copy()

fig5 = px.scatter(
    df_stock,
    x='stock_rate',
    y='bought',
    color='category',
    hover_data=['product_name', 'brand_name'],
    title="Mối quan hệ Tồn kho - Doanh số",
    labels={'stock_rate': 'Stock Rate (tỷ lệ có hàng)', 'bought': 'Bought'},
    opacity=0.6
)
fig5.update_layout(height=500)
st.plotly_chart(fig5, use_container_width=True)

# Insight for Chart 5
corr_stock_bought = df_stock['stock_rate'].corr(df_stock['bought'])
low_stock_count = len(df_stock[df_stock['stock_rate'] < 0.3])
if corr_stock_bought > 0.1:
    st.warning(f"""
**📊 Insight:** Tương quan dương ({corr_stock_bought:.2f}) → Stock rate thấp = Bought thấp
- **{low_stock_count} sản phẩm** có stock_rate < 30% → Có thể đang mất doanh số vì thiếu hàng!
""")
else:
    st.info(f"**📊 Insight:** Tương quan ({corr_stock_bought:.2f}) → Stock không phải bottleneck chính")

st.divider()


# ==================== OUTPUT 6: Bảng 3 nhóm sản phẩm ====================
st.header("6️⃣ Phân Nhóm Sản Phẩm Chiến Lược (Top 20 mỗi nhóm)")

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
st.header("📊 Tổng Kết")

col1, col2, col3 = st.columns(3)

col1.metric("Nhóm A (Tăng giá)", len(group_a), "sản phẩm")
col2.metric("Nhóm B (Cắt KM)", len(group_b), "sản phẩm")
col3.metric("Nhóm C (Ưu tiên stock)", len(group_c), "sản phẩm")

st.success("✅ EDA hoàn tất với 6 outputs chính!")
