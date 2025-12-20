"""
File cấu hình chung (path, DB URL, API keys, v.v.)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Đường dẫn gốc của project
BASE_DIR = Path(__file__).parent.parent

# Đường dẫn đến thư mục data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database configuration
DB_URL = os.getenv("DB_URL", "")
DB_SCHEMA = os.getenv("DB_SCHEMA", "raw_clean_dwh")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# API Keys
API_KEY = os.getenv("API_KEY", "")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
