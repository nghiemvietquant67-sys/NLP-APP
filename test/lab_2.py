"""Smoke test for Lab 2 pipeline.
Creates a small Spark session (local), constructs a DataFrame with a few texts,
runs the pipeline from src.lab_2 and verifies that 'features' column exists.
If pyspark is not installed, raises an informative exception so CI can install it.
"""
import sys

try:
    from pyspark.sql import SparkSession
except Exception as exc:
    raise RuntimeError("pyspark not available; install pyspark to run Lab 2 smoke test") from exc

from src.lab_2 import build_pipeline


def test_lab2_smoke():
    spark = SparkSession.builder.master("local[2]").appName("Lab2-SmokeTest").getOrCreate()

    data = [
        ("I love NLP and machine learning." ,),
        ("This is a short document for testing." ,),
        ("Spark ML pipelines are useful in production." ,)
    ]

    df = spark.createDataFrame(data, ["text"]) 

    pipeline = build_pipeline()
    model = pipeline.fit(df)
    result = model.transform(df)

    assert "features" in result.columns
    # ensure no nulls in features
    assert result.select("features").rdd.map(lambda r: r[0] is None).filter(lambda x: x).count() == 0

    print("Lab 2 smoke test passed: 'features' present and non-null for all rows")
    spark.stop()


if __name__ == "__main__":
    test_lab2_smoke()
    print("Done")
