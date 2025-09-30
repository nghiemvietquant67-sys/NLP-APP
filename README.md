1. Các bước triển khai
Bài tập yêu cầu xây dựng một pipeline xử lý ngôn ngữ tự nhiên (NLP) bằng PySpark, sử dụng tập dữ liệu C4 và các công cụ như RegexTokenizer, StopWordsRemover, HashingTF, và IDF. Dưới đây là các bước triển khai cụ thể:
Bước 1: Đọc dữ liệu

Đọc tập dữ liệu c4-train.00000-of-01024-30K.json.gz từ đường dẫn D:\ vào Spark DataFrame bằng phương thức spark.read.json. Để tăng tốc xử lý, giới hạn 1000 bản ghi.
Mã:
pythondata_path = "file:///D:/c4-train.00000-of-01024-30K.json.gz"
df = spark.read.json(data_path).limit(1000)


Bước 2: Tiền xử lý văn bản

Sử dụng RegexTokenizer để phân tách văn bản từ cột text thành danh sách token, dựa trên regex \W+ (tách bởi các ký tự không phải chữ cái hoặc số).
Sử dụng StopWordsRemover để loại bỏ các từ dừng thông dụng (như "a", "the", "is") khỏi danh sách token.
Mã:
pythontokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")


Bước 3: Vector hóa dữ liệu

Sử dụng HashingTF để chuyển danh sách token đã lọc thành vector tần số thuật ngữ (TF) với numFeatures=20000.
Sử dụng IDF để tính trọng số ngược tần số tài liệu, ưu tiên các từ hiếm và quan trọng trong tập dữ liệu.
Mã:
pythonhashingTF = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=20000)
idf = IDF(inputCol="raw_features", outputCol="features")


Bước 4: Xây dựng pipeline

Tạo một pipeline kết hợp các giai đoạn: tokenizer, remover, hashingTF, và idf.
Fit pipeline trên dữ liệu và biến đổi dữ liệu để tạo ra vector đặc trưng TF-IDF.
Mã:
pythonpipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
model = pipeline.fit(df)
result = model.transform(df)


Bước 5: Chuyển vector thành chuỗi

Sử dụng hàm UDF (User-Defined Function) để chuyển cột features (vector TF-IDF) thành chuỗi văn bản (features_str) để lưu vào tệp text.
Mã:
pythondef vector_to_string(vector):
    return str(vector)
vector_to_string_udf = udf(vector_to_string, StringType())
result = result.withColumn("features_str", vector_to_string_udf(col("features")))


Bước 6: Lưu kết quả

Thu thập cột features_str bằng phương thức collect và ghi vào tệp C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt bằng Python, tránh lỗi quyền truy cập.
Kiểm tra quyền ghi trước khi lưu và xóa tệp/thư mục sai định dạng nếu tồn tại.
Mã:
pythonoutput_path = "C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt"
if os.path.exists(output_path):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    else:
        os.remove(output_path)
features_data = result.select("features_str").coalesce(1).collect()
with open(output_path, "w", encoding="utf-8") as f:
    for row in features_data:
        f.write(f"{row['features_str']}\n")


Bước 7: Ghi log

Sử dụng module logging để ghi lại thời gian bắt đầu, kết thúc, các bước thực thi, và lỗi (nếu có) vào tệp log/lab17_pipeline.log.
Mã:
pythonlogging.basicConfig(filename='log/lab17_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Job started at %s" % time.ctime())



2. Cách chạy mã và ghi log
Môi trường thực thi:

Phần mềm: Python 3.9, PySpark 3.5.1 (cài bằng pip install pyspark==3.5.1), JDK 17.
Tệp dữ liệu: Đặt c4-train.00000-of-01024-30K.json.gz tại D:\.

Cách chạy mã:

Lưu mã nguồn vào tệp lab17_pipeline.py trong thư mục C:\Users\Quan\PycharmProjects\PythonProject2.
Chạy mã bằng PowerShell với quyền Administrator:
powershellcd C:\Users\Quan\PycharmProjects\PythonProject2
.\.venv\Scripts\python.exe lab17_pipeline.py

Hoặc chạy từ PyCharm:

Nhấp chuột phải vào biểu tượng PyCharm → Run as administrator.
Mở dự án, chọn tệp lab17_pipeline.py, và nhấn Run.



Ghi log:

Tệp log được lưu tại C:\Users\Quan\PycharmProjects\PythonProject2\log/lab17_pipeline.log.
Nội dung log bao gồm thời gian bắt đầu, kết thúc, các bước thực thi (đọc dữ liệu, tiền xử lý, vector hóa, lưu kết quả), và thông báo lỗi nếu xảy ra.

Kiểm tra trước khi chạy:

Kiểm tra tệp dữ liệu:
powershelldir D:\c4-train.00000-of-01024-30K.json.gz

Kiểm tra quyền thư mục đầu ra:
powershellicacls C:\Users\Quan\Desktop\results /grant Everyone:F

Xóa tệp/thư mục sai định dạng nếu tồn tại:
powershellrmdir C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt
del C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt



3. Giải thích kết quả
Kết quả đầu ra:

Tệp C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt chứa các chuỗi biểu diễn vector TF-IDF, mỗi dòng tương ứng với một bản ghi trong tập dữ liệu. Ví dụ:
text(20000,[0,1,2],[0.123,0.456,0.789])

Số dòng trong tệp khoảng 1000, tương ứng với số bản ghi đã giới hạn.

Ý nghĩa các bước xử lý:

RegexTokenizer: Chia văn bản thành danh sách token (từ), loại bỏ các ký tự không cần thiết như dấu câu.
StopWordsRemover: Loại bỏ từ dừng để giảm nhiễu, tập trung vào các từ mang ý nghĩa.
HashingTF: Chuyển danh sách token thành vector tần số thuật ngữ, với không gian đặc trưng 20,000 chiều.
IDF: Điều chỉnh trọng số của các thuật ngữ, ưu tiên các từ hiếm xuất hiện trong ít tài liệu, giúp biểu diễn đặc trưng tốt hơn.

Kiểm tra kết quả:

Mở tệp C:/Users/Quan/Desktop/results/lab17_pipeline_output.txt để xem các vector TF-IDF.
Đếm số dòng để xác nhận:
powershell(Get-Content C:\Users\Quan\Desktop\results\lab17_pipeline_output.txt).Le
