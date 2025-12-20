"""
Trang tổng quan về dữ liệu
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Thêm thư mục src vào path
# Thêm thư mục gốc vào path để import được module src
sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

st.title("📈 Tổng Quan Dữ Liệu")

st.markdown("### Thống kê tổng quan")

from src.config import SUPABASE_URL, SUPABASE_KEY
from src.data_loader import load_from_supabase

# Kiểm tra config
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠️ Chưa cấu hình SUPABASE_URL và SUPABASE_KEY trong file .env")
    st.info("Vui lòng cập nhật file .env với thông tin dự án Supabase của bạn.")
    st.stop()

# Input lấy dữ liệu
with st.expander("📥 Lấy dữ liệu từ Supabase", expanded=True):
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    with col1:
        # Danh sách bảng dựa trên hình ảnh user cung cấp
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
            index=2 # Default to dim_product
        )
    
    with col2:
        schema_name = st.text_input("Schema:", value="raw_clean_dwh")
        
    with col3:
        rows_limit = st.number_input("Limit (Rows):", min_value=1, max_value=100000, value=1000, step=1000)
        st.write("") 
        load_btn = st.button("Tải dữ liệu", type="primary", use_container_width=True)

    if load_btn:
        with st.spinner(f"Đang tải dữ liệu từ bảng '{table_name}' (Schema: {schema_name}, Limit: {rows_limit})..."):
            # Hàm load_from_supabase mới tự lấy URL/KEY từ config
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
    
    # Data Data Types
    with st.expander("ℹ️ Thông tin cột (Data Types)"):
        dtype_df = pd.DataFrame(df.dtypes).reset_index()
        dtype_df.columns = ["Column", "Type"]
        dtype_df["Type"] = dtype_df["Type"].astype(str)
        st.table(dtype_df)
else:
    st.info("👆 Nhập tên bảng và nhấn 'Tải dữ liệu' để bắt đầu.")
