# Lab 17 - Spark NLP Pipeline (PySpark)

## 1. Tổng quan
- Mục tiêu: Xây dựng **pipeline NLP** bằng PySpark để xử lý dữ liệu văn bản từ tập **C4**
- Dữ liệu: lấy mẫu **1000 bản ghi** từ tệp `c4-train.00000-of-01024-30K.json.gz`
- Các bước chính của pipeline:
  - Tách từ (tokenization) và loại bỏ từ dừng
  - Tạo vector từ bằng **HashingTF** và chuẩn hóa bằng **IDF**
  - Chuyển vector thành chuỗi để lưu
  - Lưu kết quả và log quá trình

---

## 2. Yêu cầu
- Hệ điều hành: Windows 10/11
- Phần mềm:
  - Python 3.9
  - PySpark 3.5.1
  - JDK 17
- Dữ liệu đặt tại ổ D:\

---

## 3. Chuẩn bị môi trường
- Cài Python và PySpark đúng phiên bản
- Cài JDK 17 để PySpark có thể chạy
- Đảm bảo file dữ liệu nằm đúng đường dẫn
- Tạo thư mục kết quả và cấp quyền ghi (nếu cần)

---

## 4. Quy trình chạy
- Lưu file mã vào thư mục dự án PyCharm
- Chạy chương trình bằng:
  - Terminal (PowerShell) hoặc
  - Trực tiếp trong PyCharm
- Chạy với quyền **Administrator** để tránh lỗi ghi file
- Kết quả được lưu vào thư mục kết quả trên Desktop
- Quá trình chạy được ghi lại trong file log

---

## 5. Kiểm tra kết quả
- Xem file log để kiểm tra các bước đã chạy
- Kết quả là danh sách vector TF-IDF cho 1000 văn bản
- Có thể đếm số dòng để xác nhận đủ số bản ghi

---

## 6. Một số lỗi thường gặp
- **Permission Denied:** chưa cấp quyền ghi cho thư mục kết quả → mở quyền và chạy lại
- **Output là thư mục:** xoá thủ công trước khi chạy lại
- **HADOOP_HOME:** không cần thiết nếu chỉ ghi bằng Python; nếu dùng API của Spark thì cần cài winutils
- **Thiếu dữ liệu:** kiểm tra lại đường dẫn tệp dữ liệu

---

## 7. Kết quả
- File log ghi các bước xử lý
- File kết quả chứa khoảng 1000 dòng vector TF-IDF
- Có thể tuỳ chọn lưu kèm cả văn bản gốc cùng với vector

---

## 8. Tham khảo
- Tài liệu Apache Spark
- Thư viện PySpark MLlib
- Thông tin về bộ dữ liệu C4
