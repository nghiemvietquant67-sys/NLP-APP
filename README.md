README.md: Lab 17 - Spark NLP Pipeline
Tổng quan
Dự án xây dựng pipeline NLP bằng PySpark để xử lý tập dữ liệu C4 (c4-train.00000-of-01024-30K.json.gz). Pipeline gồm:

Đọc 1000 bản ghi từ D:\c4-train.00000-of-01024-30K.json.gz.
Tiền xử lý: RegexTokenizer, StopWordsRemover.
Vector hóa: HashingTF (numFeatures=20000), IDF.
Chuyển vector TF-IDF thành chuỗi bằng UDF.
Lưu kết quả vào C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt.
Ghi log vào log/lab17_pipeline.log.


Yêu cầu

Hệ điều hành: Windows 10/11
Phần mềm:

Python 3.9
PySpark 3.5.1 (pip install pyspark==3.5.1)
JDK 17


Dữ liệu: c4-train.00000-of-01024-30K.json.gz tại D:\ 


Cài đặt

Cài PySpark:
bashpip install pyspark==3.5.1

Cài JDK 17:

Tải: Oracle JDK
Kiểm tra: java -version


Chuẩn bị dữ liệu:

Đặt c4-train.00000-of-01024-30K.json.gz vào D:\.
Kiểm tra: dir D:\c4-train.00000-of-01024-30K.json.gz


Cấp quyền:

Đảm bảo quyền ghi cho C:\Users\Quan\Desktop\results:
bashicacls C:\Users\Quan\Desktop\results /grant Everyone:F



Xóa tệp/thư mục sai:

Nếu lab17_pipeline_output.txt sai định dạng:
bashrmdir C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
del C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt





Chạy mã

Lưu mã:

Copy mã từ Mã nguồn vào C:\Users\Quan\PycharmProjects\PythonProject2\lab17_pipeline.py.


Chạy:

PowerShell:
bashcd C:\Users\Quan\PycharmProjects\PythonProject2
.\.venv\Scripts\python.exe lab17_pipeline.py
(Chạy với quyền Administrator)
PyCharm:

Mở PyCharm với quyền Administrator.
Chạy lab17_pipeline.py.




Kiểm tra:

Log: C:\Users\Quan\PycharmProjects\PythonProject2\log/lab17_pipeline.log
Kết quả: C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt (~1000 dòng vector TF-IDF, ví dụ: (20000,[0,1,2],[0.123,0.456,0.789]))
Đếm dòng: (Get-Content C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt).Length




Mã nguồn
Tệp lab17_pipeline.py:lab17_pipeline.pyx-python•
Xử lý lỗi

Lỗi PermissionError: [Errno 13] Permission denied:

Giải pháp:

Cấp quyền: icacls C:\Users\Quan\Desktop\results /grant Everyone:F
Chạy với quyền Administrator.
Đóng trình soạn thảo nếu tệp bị khóa.




Lỗi tệp đầu ra là thư mục:

Giải pháp:

Xóa:
bashrmdir C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
del C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt

Mã tự động xóa trước khi ghi.




Lỗi HADOOP_HOME:

Giải pháp: Mã dùng collect và ghi bằng Python, không cần Hadoop. Nếu cần write.text, tải winutils.exe từ cdarlint/winutils.


Lỗi dữ liệu:

Giải pháp: Đảm bảo c4-train.00000-of-01024-30K.json.gz ở D:\.




Kết quả

Log: log/lab17_pipeline.log ghi các bước thực thi.
Kết quả: C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt chứa ~1000 dòng vector TF-IDF.
Kiểm tra: (Get-Content C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt).Length


Tùy chọn
Để lưu cả text và features_str:

Thay phần lưu kết quả:
pythonfrom pyspark.sql.functions import concat_ws
result = result.withColumn("combined", concat_ws(" | ", col("text"), col("features_str")))
features_data = result.select("combined").coalesce(1).collect()
with open(output_path, "w", encoding="utf-8") as f:
    for row in features_data:
        f.write(f"{row['combined']}\n")



Tham khảo

Apache Spark
C4 Dataset
PySpark ML
