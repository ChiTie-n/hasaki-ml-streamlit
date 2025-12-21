"""
Welcome Page - Pricing Decision Support System
"""
import streamlit as st
import base64
from pathlib import Path

st.set_page_config(
    page_title="Pricing Decision Support",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load background image
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Get background
current_dir = Path(__file__).parent.parent
image_path = current_dir / "static" / "img" / "Blue_background.png"
bg_image = get_base64_image(str(image_path))

# CSS for Welcome page only
if bg_image:
    bg_css = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: url('data:image/png;base64,{bg_image}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    bg_css = ""

st.markdown(f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {{ font-family: 'Inter', sans-serif; }}
    
    {bg_css}
    
    /* Hide Deploy toolbar - Welcome page only */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    
    /* Header Box */
    .header-box {{
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 2px solid #3b82f6;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }}
    
    .header-title {{
        font-size: 2rem;
        font-weight: 700;
        color: #0E47A1 !important;
        margin: 0 0 0.5rem 0;
    }}
    
    .header-subtitle {{
        font-size: 1rem;
        color: #64748b;
        margin: 0;
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <h1 class="header-title">
        <i class="fa-solid fa-chart-line" style="color: #3b82f6; margin-right: 0.5rem;"></i>
        Phân tích và xây dựng chiến lược giá cho sản phẩm mỹ phẩm
    </h1>
    <p class="header-subtitle">Phân tích dữ liệu và Machine Learning cho chiến lược giá</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="background: #e0f2fe; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 8px;">
    <i class="fa-solid fa-arrow-left" style="color: #3b82f6; margin-right: 0.5rem;"></i>
    <strong>Bắt đầu:</strong> Chọn một trang từ sidebar bên trái để bắt đầu phân tích
</div>
""", unsafe_allow_html=True)
