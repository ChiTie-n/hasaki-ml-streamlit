# Hasaki Price Strategy & Sentiment Intelligence Dashboard

> **Ứng dụng phân tích dữ liệu và chiến lược giá thông minh dành cho sàn TMĐT Hasaki, tích hợp Machine Learning và NLP.**

Dashboard này cung cấp cái nhìn toàn diện về hiệu suất sản phẩm, phân tích cảm xúc khách hàng từ đánh giá, và đề xuất các chiến lược điều chỉnh giá dựa trên dữ liệu thực tế từ Supabase.

---

## Cấu trúc dự án

Dự án được tổ chức theo mô hình Streamlit đa trang:

```
hasaki-ml-streamlit/
├── app/
│   ├── app.py                 # Entry point (Main Dashboard)
│   ├── ml_utils.py            # Shared ML functions & Data Processing
│   ├── styles/                # CSS styling components
│   └── pages/
│       ├── 01_Overview.py     # Tổng quan dữ liệu & KPIs
│       ├── 02_EDA.py          # Phân tích khám phá (Charts & Correlation)
│       ├── 03_Segmentation.py # Phân cụm sản phẩm & Gợi ý chiến lược
│       ├── 04_Sentiment.py    # Phân tích cảm xúc (Rule-based & ML)
│       └── 05_Pricing.py      # Mô hình hồi quy dự đoán giá
├── src/
│   ├── config.py              # Supabase & Env configurations
│   ├── data_loader.py         # Data fetching layer (Supabase SDK)
│   └── features.py            # Feature Engineering logic
├── data/                      # Local data cache
├── .env                       # Environment variables
└── requirements.txt           # Python dependencies
```

---

## Tính năng chính

### 1. Exploratory Data Analysis (EDA)

Bộ công cụ trực quan hóa mạnh mẽ giúp khám phá các mẫu hình trong dữ liệu kinh doanh:

- **Price Distribution**: Phân bổ giá bán theo từng danh mục.
- **Discount Analysis**: Đánh giá hiệu quả của các mức giảm giá lên lượng bán.
- **Correlation**: Tìm mối liên hệ giữa Tồn kho - Tỷ lệ giảm giá - Doanh số.
- **Strategic Groups**: Tự động nhận diện 3 nhóm sản phẩm: *Cần tăng giá*, *Cần cắt giảm khuyến mãi*, và *Cần ưu tiên bổ sung hàng*.

### 2. Product Segmentation (Clustering)

Sử dụng **K-Means Clustering** kết hợp **PCA** để phân nhóm sản phẩm thành các phân khúc chiến lược:

- **Budget / Mass**: Sản phẩm phổ thông, giá thấp, volume cao.
- **Mid-range / Sleeping**: Sản phẩm tầm trung, doanh số trung bình.
- **Premium / Best-seller**: Sản phẩm cao cấp hoặc bán chạy nhất.

> Gợi ý hành động chiến lược cụ thể cho từng nhóm (ví dụ: Duy trì stock, đẩy mạnh cross-sell, cắt giảm mã hàng thừa).

### 3.  Advanced Sentiment Analysis

Hai lớp phân tích cảm xúc giúp thấu hiểu khách hàng sâu sắc:

- **Rule-Based**: Phân tích từ khóa liên quan đến **GIÁ** (đắt, rẻ, đáng tiền...) để tìm ra các sản phẩm đang gặp vấn đề về định giá.
- **ML Model (XLM-RoBERTa)**: Sử dụng mô hình Deep Learning pre-trained để phân loại cảm xúc tổng quát của toàn bộ review đa ngôn ngữ (Việt/Anh), giúp đánh giá "sức khỏe" thương hiệu của sản phẩm.

### 4. Auto-Pricing Recommendations

Hệ thống đề xuất điều chỉnh giá tự động:

- Đề xuất **Tăng giá** cho sản phẩm có sentiment tốt, bán chạy nhưng đang discount quá sâu.
- Đề xuất **Giảm giá/Xả hàng** cho sản phẩm bị phàn nàn giá cao hoặc tồn kho lớn kèm doanh số kém.
- Cảnh báo **Restock** cho các sản phẩm hot đang đứt hàng.

---

## Công nghệ sử dụng

- **Core**: `Python 3.10+`, `Streamlit`
- **Data Manipulation**: `Pandas`, `NumPy`
- **Visualization**: `Plotly Express` (Interactive charts)
- **Machine Learning**: `Scikit-learn` (K-Means, PCA, StandardScaler, LinearRegression)
- **NLP / Deep Learning**: `Transformers` (HuggingFace), `PyTorch` (XLM-RoBERTa model)
- **Database**: `Supabase` (PostgreSQL cloud database)

---

## Cài đặt & Hướng dẫn chạy

### 1. Yêu cầu tiên quyết

- Python 3.10 trở lên được cài đặt.
- Tài khoản Supabase và các thông tin kết nối.

### 2. Cài đặt môi trường

```bash
# Clone dự án
git clone https://github.com/your-username/hasaki-ml-streamlit.git
cd hasaki-ml-streamlit

# Tạo & kích hoạt môi trường ảo (Khuyên dùng)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Cấu hình .env

Tạo file `.env` tại thư mục gốc và điền thông tin Supabase của bạn:

```properties
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-anon-or-service-role-key"
```

### 4. Chạy ứng dụng

```bash
streamlit run app/app.py
```

Ứng dụng sẽ tự động mở tại `http://localhost:8501`.

## Ghi chú

- Lần đầu chạy trang **Sentiment**, hệ thống sẽ tải model `twitter-xlm-roberta-base-sentiment` (~1GB), vui lòng chờ trong giây lát.
- Dữ liệu được fetch trực tiếp từ Supabase, đảm bảo kết nối mạng ổn định.

---

**Big Data Project - 2024**
*Hasaki Price Intelligence Team*
