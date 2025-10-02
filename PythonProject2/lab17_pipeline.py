from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml.linalg import Vectors
import numpy as np
import logging
import time
import os

# ========================
# Thiết lập đường dẫn đầu ra
# ========================
output_path = "results/lab17_pipeline_output.txt"

os.makedirs('log', exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ========================
# Thiết lập logging
# ========================
logging.basicConfig(
    filename='log/lab17_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========================
# Khởi tạo Spark Session
# ========================
spark = SparkSession.builder \
    .appName("Lab17_SparkNLPPipeline") \
    .master("local[*]") \
    .getOrCreate()

logging.info("Job started at %s" % time.ctime())

# ========================
# Tham số cấu hình
# ========================
limitDocuments = 1000   # Số tài liệu đọc
data_path = "file:///D:/c4-train.00000-of-01024-30K.json.gz"

# ========================
# UDF chuyển vector thành chuỗi
# ========================
def vector_to_string(vector):
    return str(vector)
vector_to_string_udf = udf(vector_to_string, StringType())

try:
    # ========================
    # 1. Đọc dữ liệu
    # ========================
    logging.info(f"Checking data file at {data_path}...")
    if not os.path.exists('D:/c4-train.00000-of-01024-30K.json.gz'):
        raise FileNotFoundError(f"Data file not found at D:/c4-train.00000-of-01024-30K.json.gz")

    start_read = time.time()
    df = spark.read.json(data_path).limit(limitDocuments)
    end_read = time.time()
    logging.info(f"Time to read data: {end_read - start_read:.2f} seconds")

    # ========================
    # 2. Tiền xử lý văn bản
    # ========================
    logging.info("Setting up text preprocessing...")
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\W+"
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )

    # ========================
    # 3. Vector hóa
    # ========================
    hashingTF = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=20000
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features"
    )

    # ========================
    # 3b. Thêm Normalizer
    # ========================
    normalizer = Normalizer(
        inputCol="features",
        outputCol="normalized_features",
        p=2
    )

    # ========================
    # 4. Tạo pipeline
    # ========================
    logging.info("Building pipeline...")
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, normalizer])

    # Fit và transform
    start_fit = time.time()
    model = pipeline.fit(df)
    end_fit = time.time()
    logging.info(f"Time to fit pipeline: {end_fit - start_fit:.2f} seconds")

    start_transform = time.time()
    result = model.transform(df)
    end_transform = time.time()
    logging.info(f"Time to transform data: {end_transform - start_transform:.2f} seconds")

    # ========================
    # 5. Chuyển features thành chuỗi
    # ========================
    logging.info("Converting features to string...")
    result = result.withColumn("features_str", vector_to_string_udf(col("normalized_features")))

    # ========================
    # 6. Lưu kết quả
    # ========================
    logging.info(f"Saving results to {output_path}...")

    test_file = os.path.join(os.path.dirname(output_path), "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except PermissionError as pe:
        raise PermissionError(f"Cannot write to {output_path}. Ensure folder has write permissions or run as administrator.")

    features_data = result.select("features_str").coalesce(1).collect()
    with open(output_path, "w", encoding="utf-8") as f:
        for row in features_data:
            f.write(f"{row['features_str']}\n")

    logging.info("Result saved successfully.")

    # ========================
    # 7. Tìm top-5 tài liệu tương tự
    # ========================
    logging.info("Computing cosine similarity for top-5 similar documents...")

    reference_vector = result.select("normalized_features").first()[0]

    def cosine_similarity(v1, v2):
        return float(v1.dot(v2) / (Vectors.norm(v1, 2) * Vectors.norm(v2, 2) + 1e-10))

    all_rows = result.select("normalized_features").collect()
    similarities = [(i, cosine_similarity(reference_vector, row['normalized_features']))
                    for i, row in enumerate(all_rows)]

    top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    logging.info(f"Top 5 similar documents: {top5}")

    logging.info("Job completed successfully at %s" % time.ctime())

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")
    raise e

finally:
    spark.stop()
