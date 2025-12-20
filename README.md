# 🏪 Hasaki ML Streamlit - Price Strategy Dashboard

Ứng dụng phân tích chiến lược giá bằng Machine Learning và NLP cho Hasaki.

## 📁 Cấu trúc dự án

```
hasaki-ml-streamlit/
├── app/
│   ├── app.py                 # Entry point
│   └── pages/
│       ├── 01_Overview.py     # Data preview
│       ├── 02_EDA.py          # 6 biểu đồ phân tích
│       └── 03_ML_Models.py    # 4 tabs ML
├── src/
│   ├── config.py              # Supabase config
│   ├── data_loader.py         # Load data từ Supabase
│   └── features.py            # Feature engineering & sentiment
├── data/
│   └── gold_price_sentiment.csv
├── .env                       # SUPABASE_URL, SUPABASE_KEY
└── requirements.txt
```

## 🚀 Cài đặt

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Cấu hình .env
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# 3. Chạy app
streamlit run app/app.py
```

## 📊 Tính năng

### 1. Overview
- Load data từ Supabase
- Preview & basic stats

### 2. EDA (6 Outputs)
- Boxplot giá/discount theo category
- Scatter discount vs bought
- Scatter price vs bought  
- Scatter stock vs bought
- 3 nhóm sản phẩm chiến lược

### 3. ML Models (4 Tabs)

| Tab | Mô tả |
|-----|-------|
| K-Means Clustering | Phân cụm sản phẩm + chiến lược giá |
| Discount-Demand | Phân tích mô tả discount effectiveness |
| Sentiment Rule-Based | Keywords matching cho price sentiment |
| Sentiment ML | Pre-trained XLM-RoBERTa (no training needed) |

## 🛠️ Tech Stack

- **Streamlit** - Web framework
- **Pandas, NumPy** - Data processing
- **Plotly** - Visualization
- **Scikit-learn** - K-Means, StandardScaler
- **Transformers** - Pre-trained sentiment model
- **Supabase** - Database

## 👨‍💻 Author

Dự án Big Data - 2024
