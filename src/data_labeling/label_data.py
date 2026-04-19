import pandas as pd
import requests
import json
import time
from pathlib import Path

# Nếu chưa cài tqdm, có thể yêu cầu cài: pip install tqdm
try:
    from tqdm import tqdm
except ImportError:
    print("Vui lòng cài đặt tqdm: pip install tqdm")
    # Dự phòng giả tạo thư viện tqdm nếu không có
    def tqdm(iterable, *args, **kwargs):
        return iterable

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

def get_sentiment_label(text):
    """
    Hàm gọi API của Ollama để lấy nhãn cảm xúc từ text.
    Trả về 1 trong 3 nhãn: positive, negative, neutral
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    # Xây dựng prompt có hướng dẫn kỹ càng
    prompt = f"""
Nhiệm vụ của bạn là phân loại cảm xúc của câu nhận xét (review) về trung tâm thương mại dưới đây thành một trong ba nhãn: 'positive', 'negative', hoặc 'neutral'. 
Tuyệt đối CHỈ TRẢ VỀ một từ duy nhất đại diện cho nhãn (positive, negative hoặc neutral), ngôn ngữ trả về là tiếng Anh (không giải thích thêm, không thêm dấu câu).

Câu nhận xét cần phân loại: "{text}"
Nhãn:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Nhiệt độ thấp để model đưa ra kết quả nhất quán
            "top_p": 0.8
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Tiền xử lý kết quả trả về: chuyển thành chữ thường, xóa khoảng trắng thừa
        raw_label = result.get("response", "").strip().lower()
        
        # Lọc ra nhãn cuối cùng đảm bảo đúng 3 loại chuẩn
        if "positive" in raw_label:
            return "positive"
        elif "negative" in raw_label:
            return "negative"
        else:
            return "neutral"
            
    except Exception as e:
        # Nếu model lỗi hoặc service bị đóng rớt
        print(f"Lỗi khi gọi mô hình cho dòng: {text[:20]}... Lỗi: {e}")
        return "neutral" # Gán neutral nếu có lỗi (fallback)

def main():
    # 1. Định nghĩa các đường dẫn file (Dùng pathlib để tránh lỗi chạy máy khác)
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "processed" / "cleaned_reviews.csv"
    output_path = project_root / "data" / "processed" / "labeled_reviews.csv"
    
    if not input_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu tại: {input_path}")
        print("Vui lòng kiểm tra lại quá trình làm sạch (clean_data.py) trước.")
        return

    print(f"Đang đọc dữ liệu đầu vào từ: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Tổng số bản ghi cần gán nhãn: {len(df)}")
    print(f"Mô hình sử dụng: {MODEL_NAME}")
    print("Quá trình gán nhãn bắt đầu...")
    
    # 2. Tạo tiến trình chạy (tqdm) với Pandas
    tqdm.pandas(desc="Gán nhãn")
    
    # Đo thời gian
    start_time = time.time()
    
    # Áp dụng hàm get_sentiment_label vào cột cleaned_text
    df['sentiment'] = df['cleaned_text'].progress_apply(get_sentiment_label)
    
    end_time = time.time()
    print(f"Hoàn thành! Quá trình mất khoảng {round(end_time - start_time, 2)} giây.")
    
    # 3. Lưu trữ kết quả
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu kết quả đã gán nhãn tại: {output_path}")
    
    # Report số lượng
    print("\n---------- THỐNG KÊ KẾT QUẢ ----------")
    print(df['sentiment'].value_counts())
    print("--------------------------------------")

if __name__ == "__main__":
    main()
