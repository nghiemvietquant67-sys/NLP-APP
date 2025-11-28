# Lab 6 — Giới thiệu về Transformers (Report ngắn)

## Mục tiêu
- Ôn lại kiến trúc Transformer (Encoder, Decoder, Encoder-Decoder) và ứng dụng chính.
- Sử dụng các mô hình pre-trained (BERT / GPT /...) để làm các tác vụ cơ bản: Fill-Mask, Text Generation, Sentence Representation.
- Luyện thao tác với `transformers` pipeline (HuggingFace) và hiểu kết quả thực nghiệm.

## Tóm tắt nội dung thực hành
1. Fill-Mask (Masked Language Modeling)
   - Pipeline: `pipeline("fill-mask")` (mặc định model: BERT variant / phobert trong ví dụ).
   - Input mẫu: `Hanoi is the [MASK] of Vietnam.`
   - Kết quả (top 5): 1) `capital` (0.9341), 2) `Republic` (0.0300), 3) `Capital` (0.0105), 4) `birthplace` (0.0054), 5) `heart` (0.0014)
   - Kết luận: Mô hình dự đoán chính xác `capital` là từ phù hợp nhất.

2. Text Generation (Next Token Prediction)
   - Pipeline: `pipeline("text-generation", model="gpt2")`.
   - Prompt: `The best thing about learning NLP is`
   - Kết quả mẫu: một hoặc nhiều câu được sinh ra (vd: "... that you don't need to understand every aspect of it...").
   - Kết luận: Kết quả sinh ra mạch lạc, phù hợp với ngữ cảnh; phù hợp với GPT-2 (decoder-only).

3. Sentence Embedding (Mean Pooling)
   - Model: `bert-base-uncased` (AutoTokenizer + AutoModel)
   - Phương pháp thu vector câu: Mean Pooling trên `last_hidden_state` (bỏ padding bằng `attention_mask`).
   - Kết quả thực tế: Vector đầu ra có kích thước: `torch.Size([1, 768])` (hidden_size = 768 là kích thước vector của BERT-base).
   - Lưu ý: Sử dụng `attention_mask` để bỏ phần padding khi tính mean pooling.

## Cấu trúc dữ liệu (phần mô tả data cần trong repo)
Gợi ý lưu trữ dữ liệu, đặt tại `data/`:
```
data/
|__raw/        # dữ liệu gốc (không nên push nếu quá lớn)
|__sample/     # dataset mẫu để chạy demo (được phép push)
|__models/     # scripts/links để download models (không push binaries lớn)
|__README.md   # mô tả cấu trúc data & cách lấy dữ liệu
```

- `sample/` nên chứa một vài câu ví dụ (ví dụ file `sample_sentences.txt`) để người khác có thể chạy demo mà không cần tải dataset lớn.
- Không nên push file/weights lớn (ví dụ pretrained model .pt, .bin) — thay vào đó cung cấp script `scripts/download_model.sh` hoặc hướng dẫn cài đặt trong README.

## Phần Test / Kết quả (từ notebook)
- Bài 1: Fill-Mask
  - Câu: `Hanoi is the [MASK] of Vietnam.`
  - Kết quả top1: `capital` với score ~0.9341 — coi là dự đoán đúng.
- Bài 2: Text Generation
  - Prompt: `The best thing about learning NLP is` → Output: văn bản tự nhiên, ngữ pháp phù hợp.
- Bài 3: Sentence Embedding
  - Vector embedding (ví dụ): `tensor([...])`
  - Kích thước: `[1, 768]`.

## Cách chạy (hướng dẫn ngắn)
1. Thiết lập môi trường (Python >= 3.8):
```powershell
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```
2. Hoặc cài nhanh:
```powershell
pip install transformers torch
```
3. Chạy notebook demo (hoặc các example scripts):
```powershell
# Mở notebook trong Jupyter hoặc VS Code
jupyter notebook Lab_6/note_book/Lab_6.ipynb
# Hoặc chạy script python nếu đã tách thành script
python Lab_6/examples/run_mask_fill.py
```

## Kết luận
- Lab 6 giúp các bạn hiểu rõ hơn về Transformer và biết cách áp dụng các mô hình pre-trained cho các tác vụ cơ bản.
- Ghi chú: Bắt buộc phải có report (mô tả), sample data nhỏ để reproduce và file hướng dẫn chạy (README).

---
Bạn có thể chỉnh sửa bản này nếu muốn bổ sung kết quả cụ thể hơn (ví dụ paste output chi tiết từ notebook vào phần 'Test').
