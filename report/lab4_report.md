# Lab 4 — Phân loại văn bản

Harito ID: 2025-10-23

## Mục tiêu
Xây dựng một pipeline phân loại văn bản hoàn chỉnh từ văn bản thô đến mô hình được huấn luyện và đánh giá hiệu suất.

## Những gì đã triển khai

- `src/lab_4.py`: Module gộp chứa:
  - `RegexTokenizer` — bộ tách từ đơn giản dựa trên regex
  - `TfidfVectorizer` — bộ chuyển đổi TF-IDF tối giản (fit/transform)
  - `TextClassifier` — wrapper cho các mô hình scikit-learn; hỗ trợ `logreg` và `nb`
- Tests:
  - `test/lab4_test.py` — pipeline cơ sở dùng Logistic Regression
  - `test/lab4_improvement_test.py` — thí nghiệm cải tiến dùng Multinomial Naive Bayes và tiền xử lý đơn giản

## Hướng dẫn chạy

1. Kích hoạt venv của dự án (Windows Powershell):

```
C:/Users/Quan/.vscode-R/NLP-APP/.venv/Scripts/Activate.ps1
```

2. Cài phụ thuộc (nếu cần):

```
pip install -r requirements.txt
pip install pytest
```

3. Chạy test:

```
C:/Users/Quan/.vscode-R/NLP-APP/.venv/Scripts/python.exe -m pytest -q
```

Kết quả: test suite đã chạy và tất cả test đều passed.

## Kết quả

- Mô hình cơ sở Logistic Regression (dataset mẫu nhỏ):
  - Ví dụ kết quả: `{'accuracy': 0.5, 'precision': 0.5, 'recall': 1.0, 'f1': 0.6666}`
- Mô hình cải tiến (Multinomial Naive Bayes + tiền xử lý):
  - Kết quả tương đương trên tập dữ liệu rất nhỏ (xem `test/lab4_improvement_test.py`).

Lưu ý: dataset mẫu rất nhỏ (6 ví dụ), nên kết quả có độ biến động cao và không đại diện cho ứng dụng thực tế.

## Phân tích

- Dataset nhỏ dẫn đến độ biến động lớn của các chỉ số; cần nhiều dữ liệu hơn để đánh giá chính xác.
- Naive Bayes thường là baseline tốt cho phân loại văn bản với đặc trưng bag-of-words/TF-IDF.
- Tiền xử lý (chuyển chữ thường, loại bỏ dấu câu) giúp giảm nhiễu và có thể cải thiện hiệu năng trên tập nhỏ.

## Vấn đề & Giải pháp

- Khi chạy `python test/...` trực tiếp có thể gặp lỗi `ModuleNotFoundError: No module named 'src'` do `src` chưa có trong `PYTHONPATH` — giải pháp: chạy bằng `python -m pytest` hoặc thêm project root vào `sys.path`.
- `pytest` chưa được cài trong venv ban đầu — đã cài và chạy lại test.

## Mở rộng & Gợi ý

- Dùng `sklearn.feature_extraction.text.TfidfVectorizer` để có nhiều tuỳ chọn và hiệu năng tốt hơn.
- Thử embedding (Word2Vec, FastText) hoặc mô hình tiền huấn luyện (BERT) để lấy biểu diễn ngữ nghĩa tốt hơn.
- Thêm cross-validation, tinh chỉnh tham số (GridSearchCV), và đánh giá trên dataset lớn hơn.

## Tài liệu tham khảo

- scikit-learn: https://scikit-learn.org/stable/
- Apache Spark ML: https://spark.apache.org/docs/latest/ml-guide.html
