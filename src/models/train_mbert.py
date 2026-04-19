import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from utils import load_and_prepare_data, compute_metrics, append_to_benchmark_csv, save_training_history, LABEL_TO_ID, ID_TO_LABEL

MODEL_NAME = "bert-base-multilingual-cased"
DISPLAY_NAME = "mBERT"

def main():
    print(f"=== Bắt đầu pipeline cho {DISPLAY_NAME} ({MODEL_NAME}) ===")
    
    # 1. Khởi tạo Tokenizer và Data
    print("Đang tải Tokenizer & Dữ liệu...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, test_dataset, num_tests = load_and_prepare_data(tokenizer, max_length=256)
    
    # 2. Khởi tạo Model
    print("Đang tải Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3, 
        id2label=ID_TO_LABEL, 
        label2id=LABEL_TO_ID
    )
    # Lấy số params
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Tham số: {num_params:,}")

    # 3. Cấu hình Training
    training_args = TrainingArguments(
        output_dir=f"./results_{DISPLAY_NAME.lower()}",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=torch.cuda.is_available() # Dùng mixed precision nếu có GPU
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 4. Huấn Luyện (đo thời gian)
    print("Đang tiến hành Huấn Luyện (Training)...")
    start_train = time.time()
    trainer.train()
    train_time = time.time() - start_train
    print(f"Thời gian huấn luyện: {train_time:.2f} giây")

    # Lưu lịch sử quá trình train ra CSV
    save_training_history(trainer.state.log_history, DISPLAY_NAME)

    # 5. Đánh giá (đo inference time & F1/Acc)
    print("Đang tiến hành Đánh giá trên tập Test...")
    start_infer = time.time()
    eval_results = trainer.evaluate()
    infer_time = time.time() - start_infer
    
    # Đổi Inference Time sang ml/sequence
    infer_time_ms = (infer_time * 1000) / num_tests
    
    acc = eval_results.get("eval_accuracy", 0.0)
    macro_f1 = eval_results.get("eval_macro_f1", 0.0)

    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Inference: {infer_time_ms:.2f} ms/seq")

    # 6. Viết kết quả
    append_to_benchmark_csv(DISPLAY_NAME, acc, macro_f1, train_time, infer_time_ms, num_params)
    print("Hoàn tất pipeline!")

if __name__ == "__main__":
    main()
