import pandas as pd
import re
import emoji
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

# Từ điển Teencode/Viết tắt cơ bản cho domain TTTM
TEENCODE_DICT = {
    "tttm": "trung tâm thương mại",
    "nv": "nhân viên",
    "bv": "bảo vệ",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "đc": "được",
    "dc": "được",
    "vs": "với",
    "ok": "tốt",
    "oke": "tốt",
    "bt": "bình thường",
    "r": "rồi",
    "thik": "thích",
    "wc": "nhà vệ sinh",
    "mng": "mọi người"
}

def clean_vietnamese_text(text):
    """
    Hàm chuẩn hóa văn bản tiếng Việt
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Xóa Emoji
    text = emoji.replace_emoji(text, replace='')

    # 2. Chuyển về chữ thường
    text = text.lower()

    # 3. Xóa các dấu câu lặp lại liên tiếp
    text = re.sub(r'([.?!,])\1+', r'\1', text) 
    
    # Chỉ giữ lại BẢNG CHỮ CÁI TIẾNG VIỆT, SỐ và CÁC DẤU CÂU CƠ BẢN
    vietnamese_chars = r'a-z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    text = re.sub(f'[^{vietnamese_chars}\\s.?!,]', ' ', text)

    # 4. Chuẩn hóa Teencode
    words = text.split()
    cleaned_words = [TEENCODE_DICT.get(word, word) for word in words]
    text = ' '.join(cleaned_words)

    # 5. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def convert_relative_time_to_datetime(time_str):
    """
    Chuyển đổi thời gian tương đối của Google Maps thành datetime chuẩn xác dùng dateutil.
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
        
    time_str = time_str.lower().strip()
    now = datetime.now() # Lấy mốc hiện tại
    
    # Xử lý các trường hợp đặc biệt không có số
    if "vừa xong" in time_str or "mới đây" in time_str:
        return now
        
    # Xử lý chữ "một" -> "1"
    time_str = time_str.replace("một", "1")
    
    # Trích xuất con số từ chuỗi
    match = re.search(r'(\d+)', time_str)
    if not match:
        return None
        
    value = int(match.group(1))
    
    # Tính toán lùi thời gian bằng relativedelta (chuẩn xác theo lịch)
    if "phút" in time_str:
        return now - relativedelta(minutes=value)
    elif "giờ" in time_str or "tiếng" in time_str:
        return now - relativedelta(hours=value)
    elif "ngày" in time_str:
        return now - relativedelta(days=value)
    elif "tuần" in time_str:
        return now - relativedelta(weeks=value)
    elif "tháng" in time_str:
        return now - relativedelta(months=value)
    elif "năm" in time_str:
        return now - relativedelta(years=value)
        
    return None

def preprocess_pipeline(input_file, output_file):
    print(f"Đang đọc dữ liệu từ: {input_file}")
    
    if input_file.endswith('.json'):
        df = pd.read_json(input_file)
    else:
        df = pd.read_csv(input_file)

    initial_count = len(df)
    print(f"Số lượng đánh giá ban đầu: {initial_count}")

    print("Đang thực hiện làm sạch văn bản (Text Cleaning)...")
    df['cleaned_text'] = df['review_text'].apply(clean_vietnamese_text)

    print("Đang chuẩn hóa thời gian (Time Conversion)...")
    if 'review_time' in df.columns:
        df['review_datetime'] = df['review_time'].apply(convert_relative_time_to_datetime)
        # Tạo thêm cột review_date (chỉ lấy YYYY-MM-DD) để dễ vẽ biểu đồ
        df['review_date'] = df['review_datetime'].dt.date

    print("Đang lọc dữ liệu nhiễu (Noise Filtering)...")
    df = df[df['cleaned_text'] != ""]

    # Deduplicate: lúc này ta nên check trùng lặp dựa trên text đã clean và tên người review
    df = df.drop_duplicates(subset=['reviewer_name', 'cleaned_text'])

    final_count = len(df)
    print(f"Hoàn tất! Đã giữ lại {final_count}/{initial_count} đánh giá hợp lệ (Loại bỏ {initial_count - final_count} đánh giá rác).")

    if 'review_text' in df.columns:
        df = df.drop(columns=['review_text'])

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Đã xuất dữ liệu sẵn sàng cho Giai đoạn 2 tại: {output_file}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    INPUT_PATH = project_root / "data" / "raw" / "raw_reviews.json"
    OUTPUT_PATH = project_root / "data" / "processed" / "cleaned_reviews.csv"
    
    preprocess_pipeline(str(INPUT_PATH), str(OUTPUT_PATH))