# Clean Data VinCom

Dự án thu thập, làm sạch, gán nhãn và benchmark mô hình sentiment analysis cho review Google Maps của hệ thống trung tâm thương mại Vincom.

## Mục tiêu

- Thu thập review thực tế từ Google Maps cho nhiều Vincom tại Hà Nội và TP.HCM.
- Chuẩn hóa dữ liệu văn bản tiếng Việt để sử dụng cho NLP.
- Tự động gán nhãn cảm xúc bằng local LLM qua Ollama.
- Fine-tune và so sánh nhiều mô hình Transformer cho bài toán 3 lớp: `positive`, `neutral`, `negative`.
- Tạo bộ dữ liệu sẵn sàng cho EDA, trực quan hóa và phân tích kinh doanh.

## Bài toán

Input là review người dùng gồm nội dung đánh giá, số sao, thời gian review và một số metadata khác.  
Output mong muốn:

- Dữ liệu đã làm sạch và chuẩn hóa.
- Nhãn cảm xúc cho từng review.
- Kết quả benchmark giữa các mô hình.
- File lịch sử train để vẽ learning curve.

## Dữ liệu hiện có

Project hiện đã bao gồm dữ liệu mẫu trong thư mục `data/`:

- `data/raw/raw_reviews.json`: dữ liệu gốc sau khi scrape.
- `data/processed/cleaned_reviews.csv`: dữ liệu sau khi clean.
- `data/processed/labeled_reviews.csv`: dữ liệu đã gán nhãn sentiment.
- `data/processed/benchmark_results.csv`: tổng hợp kết quả benchmark.
- `data/processed/history_*.csv`: log huấn luyện của từng mô hình.

Thông tin nhanh từ bộ dữ liệu hiện tại:

- Tổng số review: `510`
- Số địa điểm Vincom: `8`
- Nhãn sentiment: `positive`, `neutral`, `negative`
- Khoảng thời gian review: `2018-04-14` đến `2026-03-14`

## Cấu trúc thư mục

```text
Clean_Data_VinCom-main/
|-- data/
|   |-- raw/
|   |   `-- raw_reviews.json
|   `-- processed/
|       |-- benchmark_results.csv
|       |-- cleaned_reviews.csv
|       |-- labeled_reviews.csv
|       `-- history_*.csv
|-- src/
|   |-- data_collection/
|   |   `-- scrape_vincom.py
|   |-- data_processing/
|   |   `-- clean_data.py
|   |-- data_labeling/
|   |   `-- label_data.py
|   `-- models/
|       |-- train_phobert.py
|       |-- train_bartpho.py
|       |-- train_xlmroberta.py
|       |-- train_mbert.py
|       `-- utils.py
|-- environment.yml
|-- run_all_models.bat
`-- README.md
```

## Pipeline

### 1. Thu thập dữ liệu

Script `src/data_collection/scrape_vincom.py` sử dụng Playwright để:

- Mở Google Maps
- Tìm từng trung tâm thương mại trong danh sách `MALLS`
- Lấy review, tên người review, rating, review_time, helpful_count, owner_reply
- Lưu kết quả vào `data/raw/raw_reviews.json`

### 2. Làm sạch dữ liệu

Script `src/data_processing/clean_data.py` thực hiện:

- Xóa emoji
- Đưa text về chữ thường
- Chuẩn hóa dấu câu và khoảng trắng
- Chuẩn hóa một số teencode/từ viết tắt
- Chuyển `review_time` từ dạng tương đối sang `datetime`
- Loại bỏ review rỗng và review trùng lặp

Kết quả được ghi ra `data/processed/cleaned_reviews.csv`.

### 3. Gán nhãn sentiment bằng Ollama

Script `src/data_labeling/label_data.py`:

- Đọc file `cleaned_reviews.csv`
- Gọi Ollama API tại `http://localhost:11434/api/generate`
- Sử dụng model mặc định `qwen2.5:7b`
- Gán nhãn `positive`, `neutral`, `negative`

Kết quả được ghi ra `data/processed/labeled_reviews.csv`.

### 4. Fine-tune và benchmark model

Thư mục `src/models/` chứa 4 script train độc lập:

- `train_phobert.py`
- `train_bartpho.py`
- `train_xlmroberta.py`
- `train_mbert.py`

Tất cả script đều:

- Đọc `data/processed/labeled_reviews.csv`
- Chia train/test bằng `train_test_split`
- Fine-tune mô hình classification 3 nhãn
- Tính `Accuracy` và `Macro F1`
- Đo `Training Time` và `Inference Time`
- Ghi kết quả vào `data/processed/benchmark_results.csv`
- Ghi log train vào `data/processed/history_<model>.csv`

## Mô hình đã benchmark

| Model | Accuracy | Macro F1 | Training Time (s) | Inference Time (ms/seq) |
|---|---:|---:|---:|---:|
| PhoBERT | 0.8922 | 0.8755 | 140.22 | 3.83 |
| BART-pho | 0.8039 | 0.7770 | 458.50 | 12.99 |
| XLM-RoBERTa | 0.8627 | 0.8490 | 184.60 | 3.83 |
| mBERT | 0.7941 | 0.7602 | 153.74 | 3.94 |

Nhận xét nhanh:

- `PhoBERT` đang cho kết quả tốt nhất trên bộ dữ liệu hiện tại.
- `XLM-RoBERTa` là lựa chọn cân bằng khá tốt giữa độ chính xác và tốc độ.
- `BART-pho` có số tham số lớn và thời gian train/inference cao hơn rõ rệt.

## Yêu cầu môi trường

Project có file `environment.yml`, nhưng file này hiện mới bao gồm nhóm thư viện phục vụ scrape và preprocessing.  
Để chạy đầy đủ pipeline labeling + training, bạn cần bổ sung thêm các gói sau nếu môi trường hiện tại chưa có:

- `requests`
- `tqdm`
- `scikit-learn`
- `torch`
- `transformers`
- `datasets`
- `evaluate`

## Cài đặt

### Cách 1: Tạo môi trường bằng Conda

```bash
conda env create -f environment.yml
conda activate claw_scraper
```

### Cách 2: Cài thêm các gói còn thiếu

```bash
pip install requests tqdm scikit-learn torch transformers datasets evaluate
```

### Cài Playwright browser

Nếu muốn scrape dữ liệu mới:

```bash
playwright install chromium
```

### Chuẩn bị Ollama

Cần chạy Ollama local và pull model:

```bash
ollama pull qwen2.5:7b
ollama serve
```

## Cách chạy

### 1. Scrape dữ liệu

```bash
python src/data_collection/scrape_vincom.py
```

Output:

- `data/raw/raw_reviews.json`

### 2. Làm sạch dữ liệu

```bash
python src/data_processing/clean_data.py
```

Output:

- `data/processed/cleaned_reviews.csv`

### 3. Gán nhãn sentiment

```bash
python src/data_labeling/label_data.py
```

Output:

- `data/processed/labeled_reviews.csv`

### 4. Train từng model

```bash
python src/models/train_phobert.py
python src/models/train_bartpho.py
python src/models/train_xlmroberta.py
python src/models/train_mbert.py
```

### 5. Chạy tất cả model trên Windows

```bat
run_all_models.bat
```

## Định dạng dữ liệu

### `cleaned_reviews.csv`

Cột chính:

- `mall_name`
- `reviewer_name`
- `rating`
- `review_time`
- `helpful_count`
- `owner_reply`
- `cleaned_text`
- `review_datetime`
- `review_date`

### `labeled_reviews.csv`

Thêm cột:

- `sentiment`

## Ứng dụng phân tích và trực quan hóa

Bộ dữ liệu này có thể dùng để:

- So sánh mức độ hài lòng giữa các Vincom
- Phân tích sentiment theo từng địa điểm
- Vẽ biểu đồ xu hướng review theo thời gian
- Phân tích độ dài review theo rating/sentiment
- Khám phá từ khóa nổi bật trong review tích cực và tiêu cực
- Benchmark mô hình NLP cho review tiếng Việt

## Hạn chế hiện tại

- `owner_reply` trong bộ dữ liệu hiện tại đang rỗng, nên chưa phù hợp cho phân tích phản hồi của doanh nghiệp.
- `helpful_count` hiện chưa có biến thiên, nên chưa dùng tốt cho phân tích độ hữu ích của review.
- Label sentiment được tạo bằng local LLM, chưa phải nhãn tay 100%.
- Split train/test đang sử dụng random split, chưa có cross-validation.
- `environment.yml` chưa bao phủ toàn bộ dependency cho phần benchmark.

## Hướng phát triển

- Bổ sung aspect extraction: nhân viên, bãi đỗ xe, vệ sinh, không gian, ẩm thực.
- Bổ sung language detection để tách review tiếng Việt / tiếng Anh.
- Cải thiện scraper để lấy owner reply và helpful count đầy đủ hơn.
- Xây dựng notebook EDA và dashboard trực quan hóa.
- Thử nghiệm thêm Qwen, Gemma hoặc các API model thương mại.

## Tác giả và đóng góp

Nếu bạn muốn mở rộng project, nên ưu tiên 3 hướng:

1. Nâng cấp scraper để lấy metadata đầy đủ và ổn định hơn.
2. Chuyển từ auto-label sang bộ dữ liệu có kiểm duyệt thủ công.
3. Bổ sung notebook EDA/deployment để project có giá trị ứng dụng rõ hơn.
