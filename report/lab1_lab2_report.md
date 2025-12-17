# Lab 1 & Lab 2 Report — Tokenization & Count Vectorization

**Harito ID:** 2025-09-16  
**Ngày:** 2025-12-17

---

## 1) Tóm tắt tiến độ (Code + Tests)

- [x] Interface cho Tokenizer và Vectorizer — **Đã triển khai (hiện nằm trong `src/lab_1.py`)** ✅
- [x] Triển khai SimpleTokenizer — **Đã xong** ✅
- [x] Triển khai RegexTokenizer — **Đã xong** ✅
- [x] Test SimpleTokenizer và RegexTokenizer cho 1 số ví dụ cơ bản — **Test file `test/lab_1.py` có sẵn** ⚠️ (chưa chạy trên máy do môi trường Python chưa khởi tạo)
- [ ] Test SimpleTokenizer và RegexTokenizer cho dataset UD English EWT — **Chưa thực hiện**
- [x] Triển khai CountVectorizer — **Đã xong** ✅
- [x] Test CountVectorizer cho 1 số ví dụ cơ bản — **Test tích hợp trong `test/lab_1.py`** ⚠️ (chưa chạy)
- [ ] Test CountVectorizer cho dataset UD English EWT — **Chưa thực hiện**

**Tổng kết:** 4/8 mục hoàn thành. Code và test script đã được viết; cần chạy test thực tế để xác nhận code chạy không lỗi.

---

## 2) Các bước triển khai (ngắn gọn)

1. Định nghĩa `Tokenizer` (ABC) và `CountVectorizer`.
2. Triển khai `SimpleTokenizer` (lowercase, split, tách dấu câu) và `RegexTokenizer` (pattern `\w+|[^\w\s]`).
3. Triển khai `CountVectorizer` với `fit`, `transform`, `fit_transform`.
4. Gộp test Lab 1 và Lab 2 vào `test/lab_1.py` để chạy tuần tự.

---

## 3) Cách chạy code & thu log kết quả

1. Cài dependencies (nếu cần):

```powershell
pip install -r requirements.txt
```

2. Chạy test (cả Lab 1 + Lab 2):

```powershell
python test/lab_1.py
```

3. Kết quả mong đợi (tóm tắt):
- Danh sách token cho mỗi câu mẫu (Simple vs Regex)
- Vocabulary (token → index) và ma trận document-term cho corpus mẫu

**Lưu ý:** Nếu lệnh `python` không hợp lệ trên hệ của bạn, dùng launcher tương ứng (ví dụ `py`) hoặc chạy trong virtualenv.

---

## 4) Giải thích các kết quả (kỳ vọng)

- `RegexTokenizer` thường ổn định hơn trong việc tách từ và dấu câu; `SimpleTokenizer` phù hợp trường hợp đơn giản.
- `CountVectorizer` trả về ma trận đếm dựa trên vocabulary sắp xếp; các token không có trong vocabulary sẽ bị bỏ qua khi `transform`.

---

## 5) Khó khăn & cách giải quyết

- Môi trường tại thời điểm thực hiện thiếu lệnh `python` (không thể chạy test tự động ở đây).  
  → Kiến nghị: chạy `python test/lab_1.py` trên máy dev của bạn hoặc tạo virtualenv có Python.
- Trong quá trình refactor, có một số file interface bị di chuyển/xóa; để nhanh, tôi đã gộp tạm `Tokenizer` và `CountVectorizer` trong `src/lab_1.py`.  
  → Nếu muốn theo đúng phân tách ban đầu, tôi sẽ tách `Tokenizer`/`Vectorizer` về `src/core/interfaces.py` và cập nhật imports & tests.

---

## 6) Tài liệu tham khảo

- Regex tokenization pattern: `\w+|[^\w\s]` (tài liệu tham khảo: nhiều bài học về tokenization)
- Thư viện liên quan (chỉ dùng demo/notebook): `pyttsx3`, `gTTS` (không yêu cầu để chạy tests của lab này)

---

## 7) Kế hoạch tiếp theo (đề xuất)

1. Chạy test thực tế `python test/lab_1.py` và lưu log kết quả (nếu muốn tôi có thể chạy và cập nhật kết quả lên report).
2. Thêm test trên dataset UD English EWT (sử dụng `src/core/dataset_loaders.load_raw_text_data`) và hiển thị 20 token đầu tiên để so sánh.
3. (Tùy chọn) Tách `Tokenizer` và `Vectorizer` ra `src/core/interfaces.py` để đúng cấu trúc đề bài.

---

Nếu bạn muốn, tôi sẽ tiếp tục và thực hiện bước (1) hoặc (2). Chỉ cần xác nhận hành động bạn muốn tôi làm tiếp (ví dụ: "Chạy tests", hoặc "Tách interfaces").
