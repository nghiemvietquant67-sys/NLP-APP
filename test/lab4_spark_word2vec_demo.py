"""PySpark Word2Vec demo (advanced task) - minimal skeleton that user can run.
This script demonstrates loading the C4 JSON, basic cleaning + tokenization, and
training a small Word2Vec model with Spark MLlib (Word2Vec).
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Word2Vec


def main(input_path: str = "data/c4-train.00000-of-01024-30K.json", limit: int = 1000):
    spark = SparkSession.builder.appName("lab4-word2vec").getOrCreate()
    df = spark.read.json(input_path).select("text").limit(limit)
    df_clean = df.withColumn("text", lower(col("text")))
    df_clean = df_clean.withColumn("text", regexp_replace(col("text"), "[^a-z0-9\\s]", " "))
    df_clean = df_clean.withColumn("tokens", split(col("text"), "\\s+"))

    # Train Word2Vec
    w2v = Word2Vec(vectorSize=100, minCount=2, inputCol="tokens", outputCol="result")
    model = w2v.fit(df_clean)
    synonyms = model.findSynonyms("computer", 5)
    print("Top synonyms for 'computer':")
    for row in synonyms.collect():
        print(row)
    spark.stop()


if __name__ == "__main__":
    main()
