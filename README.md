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

## Các giai đoạn triển khai (Kế hoạch)

### Giai đoạn 1: Tiền xử lý & Làm sạch dữ liệu (Data Preprocessing)
Mục tiêu là biến dữ liệu "rác" thành dữ liệu "sạch" để các model dễ tiêu hóa.

- **Xử lý Emoji/Icon**: Viết script dùng thư viện emoji hoặc Regex trong Python để xóa bỏ hoặc chuyển đổi (ví dụ: 👍 thành "tốt").
- **Chuẩn hóa văn bản**: Chuyển về chữ thường, xóa khoảng trắng thừa, xóa dấu câu không cần thiết.
- **Xử lý Teencode/Viết tắt**: (Mở rộng) Rất quan trọng với review tiếng Việt. Cần một dictionary nhỏ để chuyển "đc", "ok", "bt" thành "được", "đồng ý", "bình thường".
- **Lọc nhiễu**: Xóa các review rỗng (chỉ có số sao) hoặc các review trùng lặp (spam).

### Giai đoạn 2: Khởi tạo Ground Truth với Local AI (Auto-Labeling)
Thay vì gán nhãn bằng tay, bạn dùng Ollama để tạo ra tập dữ liệu chuẩn (Ground Truth).

- **Phân loại Cảm xúc (Sentiment Analysis)**: Yêu cầu Ollama đọc review_text và gán nhãn: Tích cực (Positive), Tiêu cực (Negative), hoặc Trung lập (Neutral).
- **Nhận diện Ngôn ngữ (Language Detection)**: Phân loại vi (Tiếng Việt), en (Tiếng Anh), hoặc other. Điều này trả lời trực tiếp cho câu hỏi: Tỷ lệ khách du lịch/người nước ngoài đến Vincom là bao nhiêu?
- **Trích xuất Khía cạnh (Aspect Extraction - Mở rộng)**: Nhờ Ollama chỉ ra review đang nói về cái gì (VD: Bãi đỗ xe, Nhân viên, Vệ sinh, Không gian).

### Giai đoạn 3: Benchmark & Đánh giá các Model (Model Evaluation)
Đây là phần thể hiện hàm lượng kỹ thuật cao nhất của đề tài. Lấy tập dữ liệu đã gán nhãn ở Giai đoạn 2 làm tiêu chuẩn để "chấm điểm" các model khác.

Các Model đề xuất để so sánh:
- Các model truyền thống/nhỏ: PhoBERT, mBERT.
- Các LLM thương mại qua API (nếu có chi phí): GPT-4o mini, Gemini Flash, Claude 3.5 Haiku.
- Các Open-source LLM khác chạy trên Ollama: Qwen 2.5, Gemma 2.

Tiêu chí đo lường (Metrics): Tính toán độ chính xác tổng thể (Accuracy), Precision, Recall và F1-Score cho từng nhãn.
Đo lường hiệu năng: So sánh thời gian chạy (Latency) của từng model để xem model nào phù hợp nhất để triển khai thực tế.

### Giai đoạn 4: Khai phá dữ liệu & Phân tích chuyên sâu (EDA & Deep Analytics)
Đây là lúc bạn biến dữ liệu thành "Insights" (Thông tin chi tiết) có giá trị kinh doanh.

**Phân tích Hành vi Người dùng:**
- Top Reviewers: Ai là người bình luận nhiều nhất? (Phát hiện khách hàng thân thiết hoặc... đối thủ đi spam).
- Độ tin cậy: Những review có helpful_count cao nhất thường phàn nàn về vấn đề gì?

**Phân tích theo Thời gian (Time-series):**
- Mức độ hài lòng thay đổi thế nào giữa các ngày thường vs. ngày Lễ/Tết?
- Khung giờ/Tháng nào nhận được nhiều đánh giá nhất?

**Phân tích Phản hồi của Quản lý (Owner Reply):**
- Tỷ lệ review tiêu cực được Vincom phản hồi là bao nhiêu?
- Có sự tương quan nào giữa việc Vincom tích cực trả lời comment và việc số sao trung bình tăng lên không?

**Bản đồ Nhiệt Cảm xúc (Mở rộng):** So sánh chéo xem Vincom nào (Royal City vs. Landmark 81) có chất lượng dịch vụ đồng đều nhất.

## Tác giả và đóng góp

Nếu bạn muốn mở rộng project, nên ưu tiên 3 hướng:
1. Nâng cấp scraper để lấy metadata đầy đủ và ổn định hơn.
2. Chuyển từ auto-label sang bộ dữ liệu có kiểm duyệt thủ công.
3. Bổ sung notebook EDA/deployment để project có giá trị ứng dụng rõ hơn.
