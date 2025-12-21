"""
Styles Components - Centralized styling for Streamlit app
"""
import streamlit as st
import base64
from pathlib import Path

# Get paths
STYLES_DIR = Path(__file__).parent
APP_DIR = STYLES_DIR.parent
STATIC_DIR = APP_DIR / "static"


@st.cache_data
def get_base64_image(image_path: str) -> str:
    """Encode image to base64 for CSS embedding (cached)"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None


@st.cache_resource
def load_css() -> str:
    """Load CSS from main.css file (cached)"""
    css_file = STYLES_DIR / "main.css"
    try:
        with open(css_file, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


@st.cache_data
def get_background_css() -> str:
    """Generate CSS for background image (cached)"""
    image_path = STATIC_DIR / "img" / "Blue_background.png"
    bg_image = get_base64_image(str(image_path))
    
    if bg_image:
        return f"""
        [data-testid="stAppViewContainer"] {{
            background-image: url('data:image/png;base64,{bg_image}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp {{
            background-image: url('data:image/png;base64,{bg_image}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    return ""


def inject_css():
    """Inject all CSS styles into Streamlit page"""
    # Load main CSS
    main_css = load_css()
    
    # Get background CSS
    bg_css = get_background_css()
    
    # Inject Font Awesome
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
        unsafe_allow_html=True
    )
    
    # Inject CSS styles
    st.markdown(
        f"<style>{main_css}\n{bg_css}</style>",
        unsafe_allow_html=True
    )


def render_header(title: str, subtitle: str = "", icon: str = "fa-chart-line"):
    """Render styled header box"""
    st.markdown(f"""
    <div class="header-box">
        <h1 class="header-title">
            <i class="fa-solid {icon}" style="color: #3b82f6; margin-right: 0.5rem;"></i>
            {title}
        </h1>
        <p class="header-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_info_box(text: str, icon: str = "fa-arrow-left"):
    """Render styled info box"""
    st.markdown(f"""
    <div class="info-box">
        <i class="fa-solid {icon}" style="color: #3b82f6; margin-right: 0.5rem;"></i>
        {text}
    </div>
    """, unsafe_allow_html=True)


def inject_page_css():
    """Inject CSS for regular pages (white background, keep deploy toolbar)"""
    # Font Awesome for modern icons
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
        unsafe_allow_html=True
    )
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        /* White background */
        [data-testid="stAppViewContainer"] {
            background-color: white;
        }
        .stApp {
            background-color: white;
        }
        
        /* Expander - blue border, no shadow */
        .streamlit-expanderHeader {
            background-color: white !important;
            border: 2px solid #3b82f6 !important;
            border-radius: 10px !important;
        }
        
        [data-testid="stExpander"] {
            border: 2px solid #3b82f6 !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }
        
        /* Metrics - blue border, no shadow */
        [data-testid="stMetricValue"] {
            color: #0E47A1 !important;
        }
        
        /* DataFrame - blue outer border */
        .stDataFrame {
            border: 2px solid #3b82f6 !important;
            border-radius: 10px !important;
            overflow: hidden;
        }
        
        /* DataFrame text - dark gray */
        .stDataFrame [data-testid="stDataFrameResizable"] {
            color: #374151 !important;
        }
        
        /* Selectbox, Input - default style (no blue border) */
        .stSelectbox > div > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        
        /* Number input +/- buttons - no border */
        .stNumberInput button {
            border: none !important;
            background: transparent !important;
        }
        
        /* Button - blue style */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        
        /* Success/Info/Warning boxes */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 10px !important;
            border: 2px solid !important;
            box-shadow: none !important;
        }
        
        /* Title styling */
        h1 {
            color: #0E47A1 !important;
        }
    </style>
    """, unsafe_allow_html=True)
