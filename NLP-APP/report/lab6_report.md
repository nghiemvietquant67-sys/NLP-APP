# Lab 6 — Giới thiệu về Transformers

## Mục tiêu
- Ôn lại kiến thức cơ bản về kiến trúc Transformer (Encoder/Decoder).
- Thực hành với mô hình pre-trained: Fill-Mask, Text Generation, Sentence Embedding.

## Nội dung chính
- Bài 1: Fill-Mask — Dự đoán token bị mask (ví dụ: `Hanoi is the [MASK] of Vietnam.`).
- Bài 2: Text Generation — Sinh văn bản (ví dụ: `The best thing about learning NLP is`).
- Bài 3: Sentence Embedding — Tính vector biểu diễn câu bằng mean pooling trên `last_hidden_state`.

## Kết quả mẫu
- Fill-Mask: 'capital' là dự đoán top-1 cho câu ví dụ.
- Text Generation: GPT-2 sinh kết quả mạch lạc phù hợp ngữ cảnh.
- Embedding: Vector kích thước 768 (BERT base hidden_size).

## Cách chạy
```powershell
pip install -r requirements.txt
python Lab_6/examples/run_lab6_examples.py
```

## Ghi chú
- Lưu sample data ở `data/sample/`.
- Kiểm tra file notebook `notebook/Lab_6.ipynb` để xem mã chi tiết và output.
