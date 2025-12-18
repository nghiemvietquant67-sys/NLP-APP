# Lab 5 — Nhập môn PyTorch (Phần 1)

**Harito ID:** 2025-10-30

## Mục tiêu
- Làm quen với PyTorch: Tensor, autograd, torch.nn (Linear, Embedding), và xây dựng mô hình RNN đơn giản cho bài toán phân loại token.

## Môi trường
- PyTorch: `2.9.1+cpu` (được in trong notebook)
- Thiết bị thực thi: `cpu` (nếu GPU có sẵn notebook sẽ dùng `cuda`)
- Python/venv: dùng venv của dự án

## Nội dung thực hành (tóm tắt các bước)
1. Khởi tạo môi trường, seed và kiểm tra device.
2. Khám phá Tensor (tạo tensor từ list/NumPy, ones_like, rand_like), in `shape`, `dtype`, `device`.
3. Các phép toán trên tensor: cộng, nhân, nhân ma trận (@), minh họa in-place vs out-of-place.
4. Indexing/Slicing, View vs Copy, reshape/view.
5. Autograd: minh hoạ `.backward()`, gradient accumulation và cách reset `.grad.zero_()`.
6. Kiểm tra `nn.Linear` và `nn.Embedding` (forward + backward để xem grad).
7. Định nghĩa `MyFirstModel` (Embedding -> Linear -> ReLU -> Output) với kiểm tra forward (assert kích thước output).
8. Định nghĩa một `SimpleRNNForTokenClassification` (Embedding -> RNN -> Linear) và huấn luyện trên dữ liệu giả (toy data).
9. So sánh `RNN`, `LSTM`, `GRU` (shape của outputs & hidden states).
10. Một số tiện ích nhỏ và kiểm thử nhanh (NumPy<->Tensor, device, output shapes).

## Kết quả chính (đã chạy notebook và thu được đầu ra)
- **Phiên bản PyTorch / Device:** `PyTorch version: 2.9.1+cpu`, `Device: cpu` ✅
- **Tensor & phép toán:**
  - `x_data + x_data` → `tensor([[2, 4], [6, 8]])`
  - `x_data @ x_data.T` → `tensor([[ 5, 11], [11, 25]])`
  - In-place `a.add_(1)` thay đổi tensor, out-of-place `b = x_data + 1` không thay đổi `x_data`.
- **Indexing / Slicing:** slicing trả về view (ví dụ sửa `s[0,0]=999` làm thay đổi `A`).
- **Autograd:** Với ví dụ x=1 và z=3*(x+2)^2, sau `z.backward()` thu được `x.grad = tensor([18.])`. Gọi `z.backward()` lần 2 gây lỗi trừ khi dùng `retain_graph=True`; sử dụng `x.grad.zero_()` để reset.
- **nn.Linear / nn.Embedding:** forward, backward hoạt động; `linear.weight.grad` và `embedding.weight.grad` có giá trị không rỗng sau backward.
- **MyFirstModel forward test:** output shape = `(1, 4, 2)` — assertion passed.
- **RNN token model (toy forward):** `tag_scores shape: torch.Size([2, 4, 5])` (batch=2, seq_len=4, num_tags=5).
- **Huấn luyện RNN trên dữ liệu giả:** loss giảm trong 6 epochs: 1.4318 → 1.2855 (ví dụ thu được) ✅
  - Ví dụ in ra:
    - Epoch 1, loss: 1.4318
    - Epoch 2, loss: 1.3943
    - Epoch 3, loss: 1.3624
    - Epoch 4, loss: 1.3346
    - Epoch 5, loss: 1.3093
    - Epoch 6, loss: 1.2855
  - Predict example: `Preds shape: torch.Size([2, 6])` và in một số nhãn dự đoán.
- **RNN / LSTM / GRU shapes:** tất cả cho output shape `(batch, seq_len, hidden_dim)` và hidden shapes tương ứng (ví dụ prints: `RNN out shape: torch.Size([2, 5, 10]) hidden shape: torch.Size([1, 2, 10])`).
- **Kiểm thử nhỏ:** các assert đơn giản (NumPy->Tensor, device, output shapes) đều pass.

## Phân tích & Ghi chú
- Notebook dùng **dữ liệu giả (toy data)** nên các kết quả (loss, dự đoán) chỉ minh hoạ luồng xử lý và không phản ánh hiệu năng thật trên dữ liệu thực.
- **Autograd**: phải chú ý rằng gradient được cộng dồn khi gọi `.backward()` nhiều lần; dùng `.zero_()` trước mỗi bước cập nhật nếu cần.
- **RNN training**: loss giảm trên toy data trong vài epoch, cho thấy vòng loop training hoạt động; để đánh giá thực sự cần dataset lớn (UD-English hoặc dataset NER) và các kỹ thuật như padding, batching, mask, và validation split.
- **Tiếp theo (gợi ý cho Part 2/3/4):**
  - Thử LSTM/Bi-LSTM với embedding pretrained/đào tạo từ đầu cho text classification và token tagging.
  - Thêm early stopping, validation, và metric per-token (accuracy, F1) cho token tasks.
  - Với NER, cân nhắc Bi-LSTM + CRF hoặc chuyển sang transformer-based models để cải thiện.

## Hướng dẫn chạy (tái tạo kết quả)
1. Kích hoạt venv dự án:
```
C:/Users/Quan/.vscode-R/NLP-APP/.venv/Scripts/Activate.ps1
```
2. Mở notebook `notebook/lab5_part1.ipynb` và chạy từng ô tuần tự (Run All).
3. Hoặc chạy bằng script/python nếu cần chuyển thành tập lệnh.

## File đã tạo / cập nhật
- `notebook/lab5_part1.ipynb` — notebook thực hành (đã thêm ô demo và chạy) ✅
- `report/lab5_part1_report_vi.md` — báo cáo tiếng Việt (tệp này).

---
Nếu bạn muốn, tôi sẽ:
- Mở rộng notebook để sử dụng dataset UD-English (Part 3) cho POS tagging demo; hoặc
- Thêm một thử nghiệm LSTM/Bi-LSTM với embedding (pretrained hoặc learning-from-scratch) và báo cáo so sánh (sẽ thêm test và plots).

Nếu muốn tôi làm tiếp, hãy cho biết bạn ưu tiên phần nào (Part 2: text classification, Part 3: POS tagging, Part 4: NER / Bi-LSTM-CRF). ✨
