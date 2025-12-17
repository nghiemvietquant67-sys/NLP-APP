# Báo cáo Lab 2 — Pipeline NLP với Spark

**Harito ID:** 2025-09-25 / 2025-10-02  
**Ngày:** 2025-12-17

---

## ✅ Tóm tắt
Tôi đã triển khai một pipeline dựa trên PySpark cho Lab 2: đọc file JSON C4, tokenize văn bản (mặc định dùng `RegexTokenizer`), loại bỏ stop words, tính `HashingTF` → `IDF`, chuẩn hóa vector TF-IDF và lưu mẫu kết quả. Repository hiện chứa:

- `spark_labs/lab2_pipeline.py` — triển khai pipeline PySpark ✅
- `test/test_lab2.py` — test smoke nhẹ ✅
- `report/lab2_report.md` — báo cáo này ✅

---

## Các bước triển khai (code) 🔧
1. Phát hiện cột chứa văn bản trong JSON đầu vào và đổi tên thành `text` để đồng nhất.
2. Xây dựng `Pipeline` với các stage có thể cấu hình:
   - `RegexTokenizer` (hoặc `Tokenizer`) → `tokens`
   - `StopWordsRemover` → `filtered`
   - `HashingTF(numFeatures=...)` → `rawFeatures`
   - `IDF` → `tfidf`
   - `Normalizer` → `norm`
3. `fit` và `transform` dataset (có thể giới hạn bằng tham số `--limitDocuments` để chạy nhanh khi phát triển).
4. Lưu một mẫu nhỏ kết quả vào `results/lab2_pipeline_output.txt`.
5. Tính top-5 document tương đồng (cosine similarity) cho một document được chọn bằng cách collect các vector đã chuẩn hóa về driver.
6. Ghi thời gian từng stage và lỗi vào `log/lab2_run.log`.

---

## Cách chạy (PowerShell) ▶️
1. Cài Java (JDK 17+) và Python (3.8+).
2. Cài PySpark: `pip install pyspark`.
3. Từ thư mục gốc repo chạy:

```powershell
python spark_labs/lab2_pipeline.py --input data/c4-train.00000-of-01024-30K.json --limitDocuments 1000 --numFeatures 20000 --use_regex True --output results/lab2_pipeline_output.txt
```

- Để chạy nhanh (smoke run), dùng `--limitDocuments 100` và `--numFeatures 1000`.
- File log: `log/lab2_run.log` chứa thời gian bắt đầu/kết thúc và thời lượng từng stage.

---

## Kết quả mong đợi & diễn giải 🧾
- File output chứa các dòng theo dạng: `_id \t text_preview \t Vector([...])`.
- Vector TF-IDF biểu thị tầm quan trọng của từ trong document; sau khi chuẩn hóa (Normalizer) các vector có thể so sánh bằng cosine similarity.
- Kết quả similarity liệt kê top-5 document gần nhất cho document được chọn (hữu ích cho retrieval hoặc nearest-neighbor tasks).

---

## Khó khăn & giải pháp ⚠️
- Hạn chế môi trường: PySpark yêu cầu Java; môi trường CI ban đầu thiếu Python hoặc PySpark. Giải pháp: cung cấp test smoke không cần dataset lớn và mô tả rõ các bước cài đặt.
- Xử lý dataset lớn: tính toán similarity trên toàn bộ tập dữ liệu rất tốn kém. Giải pháp: giới hạn số document bằng `--limitDocuments` và tính similarity bằng cách collect xuống driver cho mục đích lab.

---

## Tài liệu tham khảo & ghi chú 📚
- Tài liệu Spark MLlib: https://spark.apache.org/docs/latest/ml-classification-regression.html  
- Các lớp PySpark feature: `RegexTokenizer`, `StopWordsRemover`, `HashingTF`, `IDF`, `Normalizer`  

---

Nếu bạn muốn, tôi có thể: (A) thêm workflow CI để chạy test smoke khi push, hoặc (B) mở rộng bằng phần LogisticRegression để phân loại; bạn muốn tôi làm gì tiếp theo? 🔧
