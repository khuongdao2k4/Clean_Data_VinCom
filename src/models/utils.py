import os
from pathlib import Path

# CẤU HÌNH: Lưu mô hình HuggingFace vào ổ D (trong thư mục dự án)
# Điều này giúp tránh làm đầy ổ C khi tải nhiều model lớn
project_root = Path(__file__).parent.parent.parent
os.environ["HF_HOME"] = str(project_root / "huggingface_cache")

import time
import pandas as pd
import numpy as np
import evaluate
from datetime import datetime
from datasets import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

# Định nghĩa map Nhãn
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Thư viện tính điểm HuggingFace
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    """ Hàm tính Accuracy và Macro F1 cho Trainer """
    logits, labels = eval_pred
    
    # XỬ LÝ LỖI inhomogeneous shape: Nếu logits là tuple (thường gặp ở BART), lấy phần tử đầu tiên
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": acc, "macro_f1": f1}

def get_project_root():
    return Path(__file__).parent.parent.parent

def load_and_prepare_data(tokenizer, max_length=256, test_size=0.2):
    """ Hàm load CSV, map Nhãn, Tokenize và split Train/Test dạng HuggingFace Dataset """
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "labeled_reviews.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Chỉ lấy dòng có sentiment hợp lệ
    df = df[df['sentiment'].isin(LABEL_TO_ID.keys())].copy()
    df['label'] = df['sentiment'].map(LABEL_TO_ID)
    df['text'] = df['cleaned_text'].astype(str)
    
    # Giới hạn cột cần thiết
    df = df[['text', 'label']].dropna()
    
    # Cắt Train/Test theo Sklearn
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test, len(test_df)

def append_to_benchmark_csv(model_name, accuracy, macro_f1, train_time, infer_time_ms, num_params):
    """ Hàm lưu kết quả vào file CSV sau khi train xong """
    root_dir = get_project_root()
    benchmark_path = root_dir / "data" / "processed" / "benchmark_results.csv"
    
    new_data = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Training Time (s)": round(train_time, 2),
        "Inference Time (ms/seq)": round(infer_time_ms, 2),
        "Params": num_params
    }])
    
    if not benchmark_path.exists():
        new_data.to_csv(benchmark_path, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(benchmark_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        
    print(f"\n[INFO] Đã lưu kết quả của {model_name} vào {benchmark_path}")

def save_training_history(log_history, model_name):
    """ Hàm trích xuất lịch sử training (Loss, Acc) ra file CSV riêng biệt """
    root_dir = get_project_root()
    history_path = root_dir / "data" / "processed" / f"history_{model_name.lower()}.csv"
    
    # Chuyển log_history (list of dicts) sang DataFrame
    history_df = pd.DataFrame(log_history)
    
    # Lưu ra CSV
    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] Đã lưu lịch sử huấn luyện tại: {history_path}")
