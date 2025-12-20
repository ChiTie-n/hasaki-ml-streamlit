# 🏪 Hasaki ML Streamlit Dashboard

Ứng dụng Streamlit để phân tích dữ liệu và Machine Learning cho Hasaki.

## 📁 Cấu trúc dự án

```
hasaki-ml-streamlit/
├── app/                    # Code Streamlit
│   ├── app.py             # File chạy chính
│   └── pages/             # Các trang con (multi-page app)
│       ├── 01_Overview.py
│       ├── 02_EDA.py
│       ├── 03_ML_Models.py
│       └── 04_Reviews_NLP.py
├── src/                   # Logic xử lý dữ liệu & ML
│   ├── __init__.py
│   ├── config.py         # Config chung (path, DB URL,...)
│   ├── data_loader.py    # Hàm đọc dữ liệu (Supabase/CSV/Parquet)
│   ├── features.py       # Tạo features chung (price, discount, KPI,...)
│   ├── viz.py           # Hàm vẽ biểu đồ (plotly/matplotlib)
│   └── models/
│       ├── clustering.py # KMeans, PCA
│       ├── regression.py # XGBoost/LightGBM, RandomForest
│       └── sentiment.py  # Model NLP (nếu có)
├── data/                 # Dữ liệu raw/processed
│   ├── raw/
│   └── processed/
├── notebooks/            # Jupyter để thử nghiệm (EDA, thử model)
├── .env                 # Biến môi trường (DB_URL, API_KEY, ...)
├── requirements.txt     # Dependencies
└── README.md           # File này
```

## 🚀 Cài đặt

1. Clone repository:
```bash
cd hasaki-ml-streamlit
```

2. Tạo virtual environment (khuyến nghị):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Cấu hình biến môi trường:
   - Copy file `.env` và điền thông tin:
   ```
   DB_URL=your_database_url
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   API_KEY=your_api_key
   ```

## 🎯 Chạy ứng dụng

```bash
streamlit run app/app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

## 📊 Tính năng

- **Overview**: Tổng quan về dữ liệu
- **EDA**: Phân tích dữ liệu khám phá
- **ML Models**: Các mô hình Machine Learning (Clustering, Regression)
- **Reviews NLP**: Phân tích đánh giá sản phẩm bằng NLP

## 🛠️ Công nghệ sử dụng

- **Streamlit**: Framework web app
- **Pandas, NumPy**: Xử lý dữ liệu
- **Plotly, Matplotlib, Seaborn**: Visualization
- **Scikit-learn, XGBoost, LightGBM**: Machine Learning
- **Python-dotenv**: Quản lý biến môi trường

## 📝 Hướng dẫn phát triển

1. Thêm dữ liệu vào folder `data/raw/`
2. Xử lý dữ liệu và lưu vào `data/processed/`
3. Phát triển các module trong folder `src/`
4. Tạo các trang mới trong `app/pages/`
5. Update `requirements.txt` khi thêm dependencies mới

## 👨‍💻 Tác giả

Dự án được tạo cho môn Big Data

## 📄 License

This project is licensed under the MIT License.
