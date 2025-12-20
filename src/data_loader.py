import pandas as pd
from typing import Optional
import streamlit as st
from .config import SUPABASE_URL, SUPABASE_KEY, DB_SCHEMA

def load_from_supabase(table_name: str, columns: str = "*", filters: Optional[dict] = None, 
                      schema: str = None, limit: int = 1000) -> pd.DataFrame:
    """
    Đọc dữ liệu từ Supabase bằng thư viện `supabase` (REST API).
    Không cần DB_URL (connection string).
    """
    try:
        from supabase import create_client, Client
        
        # Dùng schema từ config nếu không truyền vào
        target_schema = schema if schema else DB_SCHEMA
        
        # Kết nối client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Tạo query
        # Nếu target_schema khác 'public', ta dùng .schema()
        if target_schema and target_schema != "public":
             query = supabase.schema(target_schema).table(table_name).select(columns)
        else:
             query = supabase.table(table_name).select(columns)
        
        # Áp dụng filters nếu có
        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)
        
        # Áp dụng limit
        if limit and limit > 0:
            query = query.limit(limit)
        
        # Thực thi query
        response = query.execute()
        
        # Chuyển đổi sang DataFrame
        if response.data:
            df = pd.DataFrame(response.data)
            # st.success(f"✅ Đã tải {len(df)} dòng dữ liệu từ bảng '{table_name}'")
            return df
        else:
            st.warning(f"⚠️ Không có dữ liệu trong bảng '{table_name}'")
            return pd.DataFrame()
            
    except ImportError:
        st.error("❌ Chưa cài đặt thư viện supabase. Chạy: pip install supabase")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc từ Supabase ({table_name}): {e}")
        return pd.DataFrame()

# --- Các hàm tiện ích cho Viz / ML (Interface thống nhất) ---

def load_dim_product(limit: int | None = None) -> pd.DataFrame:
    return load_from_supabase("dim_product", limit=limit or 1000)

def load_fact_prices(limit: int | None = None) -> pd.DataFrame:
    return load_from_supabase("fact_prices", limit=limit or 1000)

def load_fact_inventory(limit: int | None = None) -> pd.DataFrame:
    return load_from_supabase("fact_inventory", limit=limit or 1000)

def load_fact_reviews(limit: int | None = None) -> pd.DataFrame:
    return load_from_supabase("fact_reviews", limit=limit or 1000)

def load_dim_category(limit: int | None = None) -> pd.DataFrame:
    return load_from_supabase("dim_category", limit=limit or 500)


# Wrapper cũ để tương thích ngược nếu cần (cho trang Overview cũ)
def load_from_supabase_wrapper(table_name, url, key, schema="public", limit=1000):
    # Hàm này chỉ để tương thích với code cũ ở 01_Overview.py nếu nó vẫn gọi kiểu cũ
    # Nhưng tốt nhất nên sửa 01_Overview.py dùng hàm load_from_supabase chuẩn ở trên.
    return load_from_supabase(table_name, schema=schema, limit=limit)
