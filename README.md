# [1] Lab 17 - Spark NLP Pipeline

## [2] Tổng quan
Dự án xây dựng **pipeline NLP** bằng **PySpark** để xử lý tập dữ liệu **C4** (`c4-train.00000-of-01024-30K.json.gz`).

Pipeline gồm:
1. Đọc **1000 bản ghi** từ `D:\c4-train.00000-of-01024-30K.json.gz`
2. Tiền xử lý: `RegexTokenizer`, `StopWordsRemover`
3. Vector hóa: `HashingTF` (`numFeatures=20000`), `IDF`
4. Chuyển vector TF-IDF thành chuỗi bằng **UDF**
5. Lưu kết quả: `C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt`
6. Ghi log: `log/lab17_pipeline.log`

---

## [3] Yêu cầu
- **Hệ điều hành:** Windows 10/11  
- **Phần mềm:**
  - Python 3.9  
  - PySpark 3.5.1  
    ```bash
    pip install pyspark==3.5.1
    ```
  - JDK 17
- **Dữ liệu:**  
  Tệp `c4-train.00000-of-01024-30K.json.gz` đặt tại `D:\`

---

## [4] Cài đặt

### [4.1] Cài PySpark
```bash
pip install pyspark==3.5.1
[4.2] Cài JDK 17
Tải: Oracle JDK 17

Kiểm tra:

bash
Sao chép mã
java -version
[4.3] Chuẩn bị dữ liệu
Đặt tệp dữ liệu vào D:\:

bash
Sao chép mã
dir D:\c4-train.00000-of-01024-30K.json.gz
[4.4] Cấp quyền cho thư mục kết quả
bash
Sao chép mã
icacls C:\Users\Quan\Desktop\results /grant Everyone:F
[4.5] Xóa tệp/thư mục sai
Nếu tệp kết quả bị sai định dạng (ví dụ bị ghi nhầm thành thư mục):

bash
Sao chép mã
rmdir C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
del C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
[5] Chạy mã
[5.1] Lưu mã
Lưu tệp lab17_pipeline.py vào:

makefile
Sao chép mã
C:\Users\Quan\PycharmProjects\PythonProject2\lab17_pipeline.py
[5.2] Chạy bằng PowerShell
powershell
Sao chép mã
cd C:\Users\Quan\PycharmProjects\PythonProject2
.\.venv\Scripts\python.exe lab17_pipeline.py
Lưu ý: Chạy PowerShell với quyền Administrator

[5.3] Chạy bằng PyCharm
Mở PyCharm bằng quyền Administrator

Chạy file lab17_pipeline.py

[5.4] Kiểm tra kết quả
Log:

lua
Sao chép mã
C:\Users\Quan\PycharmProjects\PythonProject2\log\lab17_pipeline.log
Kết quả:

swift
Sao chép mã
C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt
(~1000 dòng vector TF-IDF, ví dụ:

css
Sao chép mã
(20000,[0,1,2],[0.123,0.456,0.789])
```)

Đếm số dòng:

powershell
Sao chép mã
(Get-Content C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt).Length
[6] Mã nguồn
Tệp chính: lab17_pipeline.py

[7] Xử lý lỗi
[7.1] Lỗi PermissionError: [Errno 13] Permission denied
Giải pháp:

bash
Sao chép mã
icacls C:\Users\Quan\Desktop\results /grant Everyone:F
Chạy với quyền Administrator

Đóng trình soạn thảo nếu tệp bị khóa

[7.2] Lỗi tệp đầu ra là thư mục
Giải pháp:

bash
Sao chép mã
rmdir C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
del C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
Mã đã tự động xóa trước khi ghi lại

[7.3] Lỗi HADOOP_HOME
Giải pháp:
Mã sử dụng collect() và ghi bằng Python, không cần Hadoop
Nếu cần dùng write.text(), tải winutils.exe từ cdarlint/winutils

[7.4] Lỗi dữ liệu
Giải pháp:
Đảm bảo tệp c4-train.00000-of-01024-30K.json.gz nằm trong D:\

[8] Kết quả
Log: log/lab17_pipeline.log ghi lại các bước thực thi

Kết quả: C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt chứa khoảng 1000 dòng vector TF-IDF

Kiểm tra số dòng:

powershell
Sao chép mã
(Get-Content C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt).Length
[9] Tùy chọn: Lưu cả text và features_str
Chỉnh phần lưu kết quả trong mã:

python
Sao chép mã
from pyspark.sql.functions import concat_ws, col

result = result.withColumn("combined", concat_ws(" | ", col("text"), col("features_str")))
features_data = result.select("combined").coalesce(1).collect()

with open(output_path, "w", encoding="utf-8") as f:
    for row in features_data:
        f.write(f"{row['combined']}\n")
[10] Tham khảo
Apache Spark

C4 Dataset

PySpark MLlib
