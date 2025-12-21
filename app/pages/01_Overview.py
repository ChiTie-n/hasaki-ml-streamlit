"""
Trang tổng quan về dữ liệu
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add paths for imports
app_dir = Path(__file__).parent.parent
root_dir = app_dir.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(root_dir))

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

# Inject styles
from styles import inject_page_css
inject_page_css()

st.markdown('<h1><i class="fa-solid fa-database" style="color: #3b82f6; margin-right: 0.5rem;"></i> Tổng Quan Dữ Liệu</h1>', unsafe_allow_html=True)
st.markdown("### Thống kê và truy xuất dữ liệu từ Supabase")

from src.config import SUPABASE_URL, SUPABASE_KEY
from src.data_loader import load_from_supabase

# Kiểm tra config
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠️ Chưa cấu hình SUPABASE_URL và SUPABASE_KEY trong file .env")
    st.info("Vui lòng cập nhật file .env với thông tin dự án Supabase của bạn.")
    st.stop()

# Input lấy dữ liệu
st.markdown('<p style="margin-bottom: 0.5rem;"><i class="fa-solid fa-cloud-arrow-down" style="color: #3b82f6; margin-right: 0.5rem;"></i>Lấy dữ liệu từ Supabase</p>', unsafe_allow_html=True)
with st.expander("Cấu hình", expanded=True):
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    with col1:
        KNOWN_TABLES = [
            "dim_brand", 
            "dim_category", 
            "dim_product", 
            "dim_session", 
            "fact_inventory", 
            "fact_prices",
            "fact_reviews"
        ]
        table_name = st.selectbox(
            "Chọn bảng (Table):", 
            options=KNOWN_TABLES,
            index=2
        )
    
    with col2:
        schema_name = st.text_input("Schema:", value="raw_clean_dwh")
        
    with col3:
        rows_limit = st.number_input("Limit (Rows):", min_value=1, max_value=100000, value=1000, step=1000)
        st.write("") 
        load_btn = st.button("Tải dữ liệu", type="primary", use_container_width=True)

    if load_btn:
        with st.spinner(f"Đang tải dữ liệu từ bảng '{table_name}' (Schema: {schema_name}, Limit: {rows_limit})..."):
            df = load_from_supabase(table_name, schema=schema_name, limit=rows_limit)
            
            if not df.empty:
                st.session_state['current_data'] = df
                st.success(f"Đã tải thành công {len(df)} dòng!")

# Hiển thị dữ liệu nếu có trong session state
if 'current_data' in st.session_state:
    df = st.session_state['current_data']
    
    st.divider()
    st.subheader(f"Dữ liệu: {table_name}")
    
    # Metrics cơ bản
    m1, m2, m3 = st.columns(3)
    m1.metric("Số dòng (Rows)", df.shape[0])
    m1.metric("Số cột (Columns)", df.shape[1])
    missing_values = df.isna().sum().sum()
    m2.metric("Missing Values", missing_values)
    
    # Hiển thị DataFrame
    st.dataframe(df, use_container_width=True)
    
    # Data Types
    with st.expander("Thông tin cột (Data Types)"):
        dtype_df = pd.DataFrame(df.dtypes).reset_index()
        dtype_df.columns = ["Column", "Type"]
        dtype_df["Type"] = dtype_df["Type"].astype(str)
        st.table(dtype_df)
else:
    st.markdown('<div style="background: #e0f2fe; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 8px;"><i class="fa-solid fa-hand-pointer" style="color: #3b82f6; margin-right: 0.5rem;"></i>Nhập tên bảng và nhấn <strong>Tải dữ liệu</strong> để bắt đầu.</div>', unsafe_allow_html=True)
