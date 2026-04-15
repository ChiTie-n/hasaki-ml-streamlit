# BÁO CÁO: Hệ thống thu thập dữ liệu mỹ phẩm

Pipeline tự động thu thập dữ liệu sản phẩm mỹ phẩm từ 2 website thương mại điện tử.

---

## 1. Tổng quan hệ thống

```
Website 1: lamthaocosmetics.vn ──┐                    ┌──> Listings (URL sản phẩm)
                                 ├──> Crawler ──> DB ──┼──> Products (Chi tiết sản phẩm)
Website 2: thegioiskinfood.com ──┘                    └──> Reviews  (Đánh giá khách hàng)
```

### Dữ liệu thu thập

| Loại dữ liệu | Nguồn | Nội dung |
|--------------|-------|----------|
| **Listings** | 2 websites | URL, tên, brand, giá hiển thị |
| **Products** | 2 websites | Chi tiết: giá gốc, giá sale, mô tả, hình ảnh |
| **Reviews** | thegioiskinfood | Đánh giá, rating, ngày đăng |

---

## 2. Quy trình thu thập dữ liệu

### Sơ đồ 2 giai đoạn

```
GIAI ĐOẠN 1: LISTING (Tuần 1 lần)         GIAI ĐOẠN 2: CHI TIẾT (Hàng ngày)
────────────────────────────────          ─────────────────────────────────
listing_crawler_only.py                    main_pipeline.py

brands.txt ──> Crawl URL ──> listing_api   listing_api ──> Crawl Product ──> product_api
                                                       ──> Crawl Review  ──> review_api
```

### Chi tiết từng bước

**Giai đoạn 1: Thu thập danh sách (Weekly)**

```
Bước 1: Đọc brands.txt ──> Bước 2: Crawl listing pages ──> Bước 3: Lưu vào listing_api
         (15 brands)           (5 brands song song)              (URL + metadata)
```

**Giai đoạn 2: Thu thập chi tiết (Daily)**

```
Bước 1: Đọc listing_api ──> Bước 2: Crawl product ──> Bước 3: Crawl reviews ──> Bước 4: Lưu DB
        (theo brand)          (20 requests/s)           (20 pages/s)           (product_api,
                                                                                 review_api)
```

---

## 3. Cấu hình API Key

### Thông tin cần thiết

| Key | Mô tả | Cách lấy |
|-----|-------|----------|
| `SUPABASE_URL` | Địa chỉ database | Supabase Dashboard > Settings > API |
| `SUPABASE_KEY` | Mã xác thực | Supabase Dashboard > Settings > API > `anon` key |
| `SUPABASE_SCHEMA` | Schema lưu trữ | Mặc định: `raw` |

### Tạo file cấu hình

Tạo file `.env` từ mẫu:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
SUPABASE_SCHEMA=raw
```

### Endpoints sử dụng

| Website | Endpoint | Mục đích |
|---------|----------|----------|
| lamthaocosmetics | `/collections/vendors?q={brand}` | Lấy danh sách sản phẩm |
| thegioiskinfood | `/collections/{brand}` | Lấy danh sách sản phẩm |
| Haravan API | `customer-reviews-api.haravan.app` | Lấy đánh giá |

---

## 4. Tần suất cập nhật

### Lịch chạy

| Loại | Tần suất | File | Mục đích |
|------|----------|------|----------|
| **Listing** | 1 lần/tuần | `listing_crawler_only.py` | Cập nhật URL sản phẩm mới |
| **Product + Review** | Hàng ngày | `main_pipeline.py` | Cập nhật giá, đánh giá mới |

### Thời gian thực thi

| Giai đoạn | Thời gian | Ghi chú |
|-----------|-----------|---------|
| Listing | 2-5 phút | 5 brands song song |
| Product | 1-3 phút | 20 requests đồng thời |
| Review | 3-5 phút | 20 pages đồng thời |

---

## 5. Hướng dẫn sử dụng

### Cài đặt

```bash
# 1. Cài đặt thư viện (dùng UV)
uv sync

# 2. Tạo file .env
cp .envexample .env
# Điền thông tin Supabase vào .env

# 3. Setup database
# Mở Supabase SQL Editor > Chạy file database.sql
```

### Chạy crawler

```bash
# Tra cứu danh sách brands (chạy 1 lần)
uv run python crawl_brands.py

# Crawl listings (chạy mỗi tuần)
uv run python listing_crawler_only.py

# Crawl products + reviews (chạy hàng ngày)
uv run python main_pipeline.py
```

### Cấu hình brands

Chỉnh sửa file `brands.txt`:

```
CeraVe
Cocoon
L'Oreal
# Dòng bắt đầu bằng # sẽ bị bỏ qua
```

---

## 6. Cấu trúc dữ liệu

### Database schema

```
Database: raw.*
├── crawl_sessions     (Theo dõi mỗi lần chạy)
├── listing_api        (URL sản phẩm theo brand)
├── product_api        (Chi tiết sản phẩm - JSONB)
└── review_api         (Đánh giá khách hàng - JSONB)
```

### Format dữ liệu sản phẩm

```json
{
  "brand": "CeraVe",
  "category": "Cham soc da mat",
  "name": "Sua Rua Mat CeraVe...",
  "price": 300000,
  "price_sale": 250000,
  "bought": 150,
  "url": "https://..."
}
```

### Cơ chế chống trùng lặp

```
Crawl 1: San pham A, gia 100k ──> Luu
Crawl 2: San pham A, gia 100k ──> Bo qua (khong doi)
Crawl 3: San pham A, gia 90k  ──> Luu (co thay doi)
```

---

## 7. Cấu trúc project

```
website_crawl/
├── main_pipeline.py           # [CHINH] Crawl product + review (daily)
├── listing_crawler_only.py    # [CHINH] Crawl listings (weekly)
├── crawl_brands.py            # Tool tra cuu brands
├── config.py                  # Cau hinh he thong
├── brands.txt                 # [CONFIG] Danh sach brands
├── .env                       # [CONFIG] API keys (tu tao)
│
├── crawlers/                  # Cac module crawl
│   ├── listing_crawler.py
│   ├── product_crawler.py
│   ├── review_crawler.py
│   ├── async_product_crawler.py
│   └── async_review_crawler.py
│
├── database/
│   └── database_handler.py    # Xu ly luu tru Supabase
│
├── utils/
│   ├── logger.py              # Logging
│   └── helpers.py             # Tien ich
│
├── database.sql               # Schema database
└── .github/workflows/         # GitHub Actions
```

---

## 8. Thong so ky thuat

### Hieu suat

| Metric | Gia tri | Ghi chu |
|--------|---------|---------|
| Concurrent requests | 20 | Toi da dong thoi |
| Concurrent brands | 5 | Xu ly song song |
| Request delay | 0.3-0.5s | Tranh bi block |
| Timeout | 30s | Moi request |
| Max retries | 3 | Khi that bai |

### Anti-blocking

| Ky thuat | Mo ta |
|----------|-------|
| User-Agent | Gia lap Chrome browser |
| Smart delay | Delay ngau nhien giua requests |
| Session reuse | Tai su dung connection |
| Auto-retry | Tu dong thu lai khi loi |

---

## 9. Xu ly loi

| Loi | Nguyen nhan | Giai phap |
|-----|-------------|-----------|
| "No brands to crawl" | File brands.txt trong | Them brands vao file |
| "Ket noi Supabase that bai" | Sai API key | Kiem tra .env |
| "Cannot create sessions" | Chua chay database.sql | Chay schema trong Supabase |
| Request timeout | Website cham | Tang TIMEOUT trong config.py |

### Debug mode

```bash
# Xem log chi tiet trong thu muc logs/
# Format: crawl_YYYY-MM-DD.log
```

---

## 10. Workflow tong the

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           WEBSITE CRAWL WORKFLOW                           │
└────────────────────────────────────────────────────────────────────────────┘

SETUP (1 lan)                 TUAN 1 LAN                    HANG NGAY
─────────────                 ──────────                    ─────────
                              
1. uv sync                    4. listing_crawler_only.py    6. main_pipeline.py
   ↓                             ↓                             ↓
2. Tao .env                   5. Luu URL vao listing_api    7. Doc listings tu DB
   ↓                                                           ↓
3. Chay database.sql                                        8. Crawl product + review
                                                               ↓
                                                            9. Luu vao product_api,
                                                               review_api


KET QUA: Du lieu san pham va danh gia duoc cap nhat tu dong
```

---

## 11. Lien he va ho tro

**Checklist khi gap loi:**
1. File `.env` da tao va dien day du?
2. File `brands.txt` co brands?
3. Da chay `database.sql` trong Supabase?
4. Da chay `listing_crawler_only.py` truoc?
5. Kiem tra logs trong thu muc `logs/`

---

**Version**: 2.0  
**Updated**: Dec 2025
