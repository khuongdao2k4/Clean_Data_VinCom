Giai đoạn 1: Tiền xử lý & Làm sạch dữ liệu (Data Preprocessing)
Mục tiêu là biến dữ liệu "rác" thành dữ liệu "sạch" để các model dễ tiêu hóa.

Xử lý Emoji/Icon: Viết script dùng thư viện emoji hoặc Regex trong Python để xóa bỏ hoặc chuyển đổi (ví dụ: 👍 thành "tốt").

Chuẩn hóa văn bản: Chuyển về chữ thường, xóa khoảng trắng thừa, xóa dấu câu không cần thiết.

Xử lý Teencode/Viết tắt: (Mở rộng) Rất quan trọng với review tiếng Việt. Cần một dictionary nhỏ để chuyển "đc", "ok", "bt" thành "được", "đồng ý", "bình thường".

Lọc nhiễu: Xóa các review rỗng (chỉ có số sao) hoặc các review trùng lặp (spam).

Giai đoạn 2: Khởi tạo Ground Truth với Local AI (Auto-Labeling)
Thay vì gán nhãn bằng tay, bạn dùng Ollama để tạo ra tập dữ liệu chuẩn (Ground Truth).

Phân loại Cảm xúc (Sentiment Analysis): Yêu cầu Ollama đọc review_text và gán nhãn: Tích cực (Positive), Tiêu cực (Negative), hoặc Trung lập (Neutral).

Nhận diện Ngôn ngữ (Language Detection): Phân loại vi (Tiếng Việt), en (Tiếng Anh), hoặc other. Điều này trả lời trực tiếp cho câu hỏi: Tỷ lệ khách du lịch/người nước ngoài đến Vincom là bao nhiêu?

Trích xuất Khía cạnh (Aspect Extraction - Mở rộng): Nhờ Ollama chỉ ra review đang nói về cái gì (VD: Bãi đỗ xe, Nhân viên, Vệ sinh, Không gian).

Giai đoạn 3: Benchmark & Đánh giá các Model (Model Evaluation)
Đây là phần thể hiện hàm lượng kỹ thuật cao nhất của đề tài. Lấy tập dữ liệu đã gán nhãn ở Giai đoạn 2 làm tiêu chuẩn để "chấm điểm" các model khác.

Các Model đề xuất để so sánh:

Các model truyền thống/nhỏ: PhoBERT, mBERT.

Các LLM thương mại qua API (nếu có chi phí): GPT-4o mini, Gemini Flash, Claude 3.5 Haiku.

Các Open-source LLM khác chạy trên Ollama: Qwen 2.5, Gemma 2.

Tiêu chí đo lường (Metrics): Tính toán độ chính xác tổng thể (Accuracy), Precision, Recall và F1-Score cho từng nhãn.

Đo lường hiệu năng: So sánh thời gian chạy (Latency) của từng model để xem model nào phù hợp nhất để triển khai thực tế.

Giai đoạn 4: Khai phá dữ liệu & Phân tích chuyên sâu (EDA & Deep Analytics)
Đây là lúc bạn biến dữ liệu thành "Insights" (Thông tin chi tiết) có giá trị kinh doanh.

Phân tích Hành vi Người dùng:

Top Reviewers: Ai là người bình luận nhiều nhất? (Phát hiện khách hàng thân thiết hoặc... đối thủ đi spam).

Độ tin cậy: Những review có helpful_count cao nhất thường phàn nàn về vấn đề gì?

Phân tích theo Thời gian (Time-series):

Mức độ hài lòng thay đổi thế nào giữa các ngày thường vs. ngày Lễ/Tết?

Khung giờ/Tháng nào nhận được nhiều đánh giá nhất?

Phân tích Phản hồi của Quản lý (Owner Reply):

Tỷ lệ review tiêu cực được Vincom phản hồi là bao nhiêu?

Có sự tương quan nào giữa việc Vincom tích cực trả lời comment và việc số sao trung bình tăng lên không?

Bản đồ Nhiệt Cảm xúc (Mở rộng): So sánh chéo xem Vincom nào (Royal City vs. Landmark 81) có chất lượng dịch vụ đồng đều nhất.