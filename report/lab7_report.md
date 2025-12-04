# Lab 7 — Phân tích cấu trúc câu với spaCy (Dependency Parsing)

## Mục tiêu
- Hiểu về cấu trúc phụ thuộc (dependency parsing) trong xử lý ngôn ngữ tự nhiên.
- Sử dụng thư viện spaCy để trích xuất và phân tích các thành phần câu.
- Thực hành 3 bài tập: tìm động từ chính, trích cụm danh từ, và lấy đường dẫn tới ROOT.

## Nội dung chính

### Bài 1: Tìm động từ chính của câu (find_main_verb)
- **Mục tiêu**: Xác định động từ chính (ROOT của câu).
- **Phương pháp**: Tìm token có `dep_ == "ROOT"` và kiểm tra nếu nó là động từ (VERB/AUX).
- **Xử lý trường hợp đặc biệt**: Nếu ROOT là danh từ (câu danh từ), tìm động từ con trực tiếp.
- **Kết quả ví dụ**: Câu "The cat is sleeping peacefully on the sofa." → Động từ chính: "sleeping".

### Bài 2: Trích cụm danh từ mà không dùng doc.noun_chunks (my_noun_chunks)
- **Mục tiêu**: Tự triển khai hàm trích cụm danh từ (noun chunks) từ dependency tree.
- **Phương pháp**:
  - Với mỗi token là danh từ (NOUN, PROPN, PRON):
    - Tìm biên trái: duyệt ngược để tìm các từ bổ nghĩa (det, amod, compound, poss, nummod).
    - Tìm biên phải: tìm các từ bên phải trong subtree (compound, amod, nmod, appos, acl).
  - Trích xuất span từ biên trái tới biên phải, tránh trùng lặp.
- **Kết quả ví dụ**: Câu "The quick brown fox jumps over the lazy dog." → Cụm danh từ: "The quick brown fox", "the lazy dog".

### Bài 3: Lấy đường dẫn từ token tới ROOT (get_path_to_root)
- **Mục tiêu**: Trả về chuỗi các token từ một token cho trước lên tới ROOT.
- **Phương pháp**: Duyệt qua `token.head` liên tiếp cho đến khi gặp token có `dep_ == "ROOT"`.
- **Kết quả ví dụ**: Từ "eating" trong "I love eating delicious chocolate cake in the morning." → Đường dẫn: eating [xcomp] ← love [ROOT].

## Kết quả & Phân tích

### Kết quả chạy code
1. **find_main_verb**: Thành công xác định động từ chính trong các loại câu khác nhau.
2. **my_noun_chunks**: Trích xuất chính xác cụm danh từ; trường hợp có các từ bổ nghĩa phức tạp cũng xử lý tốt.
3. **get_path_to_root**: Tạo đúng đường dẫn phụ thuộc từ token bất kỳ lên ROOT.

### Nhận xét
- Dependency parsing là công cụ mạnh mẽ để hiểu cấu trúc ngữ pháp của câu.
- spaCy cung cấp API đơn giản để truy cập dependency tree; tuy nhiên tự triển khai cũng giúp hiểu sâu hơn.
- Các bài tập này là nền tảng cho các tác vụ NLP cao cấp hơn (relation extraction, semantic role labeling, etc.).

## Cách chạy
```powershell
# Cài đặt spaCy và tải model
pip install spacy
python -m spacy download en_core_web_sm

# Chạy notebook
jupyter notebook notebook/Lab_7.ipynb
```

## Dữ liệu
- Model: `en_core_web_sm` hoặc `vi_core_news_lg` (tiếng Việt).
- Sample câu: được tạo trực tiếp trong code (không cần external data).

## Tài liệu tham khảo
- spaCy documentation: https://spacy.io/
- Dependency parsing: https://spacy.io/usage/linguistic-features#dependency-parse
