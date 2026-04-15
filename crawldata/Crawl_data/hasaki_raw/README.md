# Hasaki Raw Data Crawler

Hệ thống tự động thu thập dữ liệu sản phẩm và đánh giá từ Hasaki.vn.

---

## Tổng quan hệ thống

```
┌─────────────┐        ┌──────────────┐        ┌─────────────┐        ┌──────────────┐
│  Hasaki.vn  │ ────>  │   Crawler    │ ────>  │  Database   │ ────>  │  Dữ liệu     │
│  (Website)  │  API   │  (Thu thập)  │  Lưu   │  (Supabase) │  Dùng  │  (Phân tích) │
└─────────────┘        └──────────────┘        └─────────────┘        └──────────────┘
```

### Quy trình 2 giai đoạn

```
GIAI ĐOẠN 1: DANH SÁCH (Tuần 1 lần)          GIAI ĐOẠN 2: CHI TIẾT (Ngày 1 lần)
┌─────────────────────────────────────┐      ┌─────────────────────────────────────┐
│  crawl_listings.py                  │      │  crawler.py                         │
│  ─────────────────                  │      │  ──────────                         │
│  Trang chủ → Danh mục → Mã sản phẩm │ ───> │  Mã sản phẩm → Chi tiết + Đánh giá │
│  Kết quả: ~10,000 mã sản phẩm       │      │  Kết quả: Giá, mô tả, reviews...    │
│  Thời gian: 2-5 phút                │      │  Thời gian: 1-2 phút                │
└─────────────────────────────────────┘      └─────────────────────────────────────┘
```

**Tại sao chia 2 giai đoạn?**
- Danh sách sản phẩm ít thay đổi → crawl 1 lần/tuần
- Giá và đánh giá thay đổi hàng ngày → crawl mỗi ngày

---

## Quy trình thu thập chi tiết

### Giai đoạn 1: Thu thập danh sách (crawl_listings.py)

```
Home API ───> Danh mục (100+) ───> Listing Pages ───> Mã sản phẩm ───> Database
  1 call         20 workers           Pagination        10,000+ IDs      listing_api
```

### Giai đoạn 2: Thu thập chi tiết (crawler.py)

```
Database ───> Lọc brands ───> Product API ───> Review API ───> Database
listing_api    brands.txt      10 workers       20 workers      product_api
                                                                 review_api
```

**Hiệu suất**:
- Products: 20-25 sản phẩm/giây
- Reviews: 15-20 trang/giây
- Chỉ lưu dữ liệu mới/thay đổi (Incremental Snapshot)

---

## Cài đặt nhanh

### Bước 1: Cấu hình API Key

Tạo file `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
SUPABASE_SCHEMA=raw
```

**Lấy API key từ đâu?**
- Vào Supabase Dashboard
- Settings → API → Copy `URL` và `anon` key

### Bước 2: Chọn thương hiệu

Chỉnh file `brands.txt`:

```
105     # CeraVe
1927    # Cocoon
2043    # L'Oreal
```

Tìm mã thương hiệu: `python find_brands.py`

### Bước 3: Chạy crawler

```bash
# Lần đầu tiên (cài đặt)
pip install -r requirements.txt

# Bước 1: Thu thập danh sách (chỉ chạy 1 lần đầu)
python crawl_listings.py

# Bước 2: Thu thập chi tiết (chạy hàng ngày)
python crawler.py
```

---

## Tần suất cập nhật

### Chạy thủ công

```
Tuần 1 lần:  python crawl_listings.py    # Cập nhật danh sách sản phẩm mới
Hàng ngày:   python crawler.py           # Cập nhật giá và đánh giá
```

### Tự động (GitHub Actions)

```
Thứ 2, 1:00 AM UTC:    crawl_listings.py
Mỗi ngày, 2:00 AM UTC: crawler.py
```

**Thiết lập**:
1. Push code lên GitHub
2. Settings → Secrets → Add `SUPABASE_URL`, `SUPABASE_KEY`
3. Xong! Hệ thống tự chạy theo lịch

---

## Cấu trúc dữ liệu

```
Database: raw.*
├── crawl_sessions     (Theo dõi mỗi lần chạy)
├── home_api          (Snapshot trang chủ)
├── listing_api       (Mã sản phẩm + thương hiệu)
├── product_api       (Chi tiết: giá, mô tả, hình ảnh...)
└── review_api        (Đánh giá của khách hàng)
```

**Incremental Snapshot**: Chỉ lưu dữ liệu thay đổi

```
Crawl 1: Giá 100k ───> Lưu
Crawl 2: Giá 100k ───> Bỏ qua (không đổi)
Crawl 3: Giá 90k  ───> Lưu (có thay đổi)
```

---

## Xử lý lỗi

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| "No products found" | Chưa chạy giai đoạn 1 | `python crawl_listings.py` |
| "SUPABASE_URL must be set" | Chưa tạo file .env | Tạo `.env` từ `env.example` |
| "Permission denied" | Chưa cấp quyền DB | Chạy `schema.sql` trong Supabase |
| Socket errors | Quá nhiều workers | Giảm `MAX_REVIEW_WORKERS` trong `crawler.py` |

**Debug mode**:
```bash
# Windows
$env:LOG_LEVEL="DEBUG"
python crawler.py

# Linux/Mac
export LOG_LEVEL=DEBUG
python crawler.py
```

---

## Thống kê database

### Xem session mới nhất

```sql
SELECT * FROM raw.crawl_sessions 
ORDER BY started_at DESC LIMIT 1;
```

### Tỷ lệ sản phẩm có đánh giá

```sql
SELECT 
    COUNT(DISTINCT pa.product_id) as total_products,
    COUNT(DISTINCT ra.product_id) as with_reviews,
    ROUND(100.0 * COUNT(DISTINCT ra.product_id) / COUNT(DISTINCT pa.product_id), 1) as coverage_pct
FROM raw.product_api pa
LEFT JOIN raw.review_api ra ON pa.product_id = ra.product_id;
```

### Top sản phẩm nhiều đánh giá

```sql
SELECT product_id, COUNT(*) as pages, MAX(pages) as last_page
FROM raw.review_api
GROUP BY product_id
ORDER BY pages DESC
LIMIT 10;
```

---

## Cấu trúc project

```
hasaki_raw/
├── crawler.py              # [CHÍNH] Thu thập chi tiết (daily)
├── crawl_listings.py       # [CHÍNH] Thu thập danh sách (weekly)
├── api_client.py           # Gọi API Hasaki
├── supabase_client.py      # Lưu database
├── config.py               # Cấu hình
├── logger.py               # Logging
├── find_brands.py          # Tool tìm mã thương hiệu
├── brands.txt              # [CONFIG] Danh sách thương hiệu
├── .env                    # [CONFIG] API keys (tự tạo)
├── env.example             # Mẫu file .env
├── requirements.txt        # Python dependencies
├── schema.sql              # Database schema
└── .github/workflows/      # GitHub Actions
```

---

## API Endpoints

Crawler sử dụng các API công khai của Hasaki:

| API | Mục đích | Tần suất |
|-----|----------|----------|
| `/wap/v2/master/?page=newHeaderHome` | Lấy danh mục | 1 lần/tuần |
| `/wap/v2/catalog/category/get-listing-product` | Lấy danh sách sản phẩm | 1 lần/tuần |
| `/wap/v2/product/detail?id={}` | Chi tiết sản phẩm | Mỗi ngày |
| `/mobile/v3/detail/product/rating-reviews` | Đánh giá | Mỗi ngày |

**Lưu ý**: API có thể thay đổi, kiểm tra file `config.py` hoặc `.env` để cập nhật.

---

## Flow hoạt động tổng thể

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         HASAKI CRAWLER WORKFLOW                              │
└──────────────────────────────────────────────────────────────────────────────┘

TUẦN 1 LẦN (Thứ 2)                           HÀNG NGÀY
─────────────────────                         ─────────
                                              
1. Lấy danh mục                              4. Đọc mã sản phẩm từ DB
   ↓                                            ↓
2. Crawl listing pages                       5. Lọc theo brands.txt
   ↓                                            ↓
3. Lưu mã sản phẩm vào DB                    6. Crawl chi tiết sản phẩm
   │                                            ↓
   └──────────────────────────────────────→  7. Crawl đánh giá
                                                ↓
                                             8. Lưu vào DB (chỉ data mới)


KẾT QUẢ: Database luôn cập nhật với data mới nhất, tiết kiệm dung lượng
```

---

## Performance & Metrics

### Giai đoạn 1 (Listing)
- **Input**: ~100 danh mục
- **Output**: ~10,000 mã sản phẩm
- **Workers**: 20 luồng song song
- **Thời gian**: 2-5 phút
- **Tần suất**: 1 lần/tuần

### Giai đoạn 2 (Chi tiết)
- **Input**: ~10,000 mã (lọc theo brands)
- **Output**: Chi tiết + Reviews
- **Workers**: 10 (product) + 20 (review)
- **Tốc độ**: 20-25 products/s, 15-20 review pages/s
- **Thời gian**: 1-2 phút (212 products example)
- **Tần suất**: Hàng ngày

### Tiết kiệm
- Chỉ lưu data thay đổi → giảm 70-90% dung lượng
- Database comparison → không cần hash Python-side
- Batch insert → giảm 90% RPC calls

---

## Hỗ trợ

**Checklist khi gặp lỗi**:
1. File `.env` đã tạo và điền đầy đủ?
2. File `brands.txt` có mã thương hiệu?
3. Đã chạy `crawl_listings.py` trước?
4. Database schema đã được tạo?
5. Thử bật DEBUG mode xem log

**Workflow chuẩn**:
```
Setup → crawl_listings.py → crawler.py → Check DB → Done
  ↑                                          │
  └──────────────── Có lỗi? ────────────────┘
```

---

**Version**: 2.0  
**Updated**: Dec 2025  
**License**: MIT
