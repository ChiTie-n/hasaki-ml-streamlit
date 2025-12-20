"""
File chạy chính của ứng dụng Streamlit
"""
import streamlit as st

st.set_page_config(
    page_title="Hasaki ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🏪 Hasaki ML Dashboard")
st.markdown("### Chào mừng đến với hệ thống phân tích ML")

st.info("👈 Vui lòng chọn một trang từ sidebar để bắt đầu")

# Thông tin tổng quan
st.markdown("""
### Các tính năng:
- **Overview**: Tổng quan về dữ liệu
- **EDA**: Phân tích dữ liệu khám phá
- **ML Models**: Các mô hình Machine Learning
- **Reviews NLP**: Phân tích đánh giá bằng NLP
""")
