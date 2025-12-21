"""
File chạy chính của ứng dụng Streamlit
"""
import streamlit as st

st.set_page_config(
    page_title="Hasaki ML Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers styling */
    h1 {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
