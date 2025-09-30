from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import logging
import time
import os

# Thiết lập đường dẫn đầu ra
output_path = "results/lab17_pipeline_output.txt"

# Tạo thư mục log và results nếu chưa tồn tại
os.makedirs('log', exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Thiết lập logging
logging.basicConfig(
    filename='log/lab17_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("Lab17_SparkNLPPipeline") \
    .master("local[*]") \
    .getOrCreate()

logging.info("Job started at %s" % time.ctime())

# Hàm UDF để chuyển vector thành chuỗi
def vector_to_string(vector):
    return str(vector)
vector_to_string_udf = udf(vector_to_string, StringType())

try:
    # 1. Đọc dữ liệu từ đường dẫn trên ổ D:\
    data_path = "file:///D:/c4-train.00000-of-01024-30K.json.gz"
    logging.info(f"Checking data file at {data_path}...")
    if not os.path.exists('D:/c4-train.00000-of-01024-30K.json.gz'):
        raise FileNotFoundError(f"Data file not found at D:/c4-train.00000-of-01024-30K.json.gz")
    logging.info("Reading C4 dataset...")
    df = spark.read.json(data_path).limit(1000)

    # 2. Tiền xử lý văn bản
    logging.info("Setting up text preprocessing...")
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\W+"
    )

    # Loại bỏ từ dừng
    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )

    # 3. Vector hóa
    hashingTF = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=20000
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features"
    )

    # 4. Tạo pipeline
    logging.info("Building pipeline...")
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

    # Fit và transform
    logging.info("Fitting and transforming pipeline...")
    model = pipeline.fit(df)
    result = model.transform(df)

    # 5. Chuyển cột features thành chuỗi
    logging.info("Converting features to string...")
    result = result.withColumn("features_str", vector_to_string_udf(col("features")))

    # 6. Lưu kết quả bằng Python
    logging.info(f"Saving results to {output_path}...")
    # Kiểm tra quyền ghi
    test_file = os.path.join(os.path.dirname(output_path), "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except PermissionError as pe:
        raise PermissionError(f"Cannot write to {output_path}. Ensure folder has write permissions or run as administrator.")
    # Thu thập và ghi dữ liệu
    features_data = result.select("features_str").coalesce(1).collect()
    with open(output_path, "w", encoding="utf-8") as f:
        for row in features_data:
            f.write(f"{row['features_str']}\n")

    logging.info("Job completed successfully at %s" % time.ctime())

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")
    raise e

finally:
    spark.stop()