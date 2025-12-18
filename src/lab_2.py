"""Lab 2 (provided as Lab 17) — Spark ML pipeline example

- Reads a JSON-lines C4 file into a Spark DataFrame (expects a 'text' field)
- Uses Spark ML: RegexTokenizer, StopWordsRemover, HashingTF, IDF
- Fits pipeline and transforms a sample or the full dataset
- Saves transformed features as Parquet and logs process

Usage:
    python src/lab_2.py \
        --input data/c4-train.00000-of-01024-30K.json \
        --output data/outputs/lab2_parquet \
        --sample 1000

If Spark is not available locally, the script detects it and raises an informative error.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "lab2") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


LOGGER = get_logger()


def build_pipeline(input_col: str = "text", output_col: str = "features"):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF

    tokenizer = RegexTokenizer(inputCol=input_col, outputCol="tokens", pattern="\\w+|[^\\w\\s]")
    stop = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    hashing = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=2 ** 18)
    idf = IDF(inputCol="rawFeatures", outputCol=output_col)

    pipeline = Pipeline(stages=[tokenizer, stop, hashing, idf])
    return pipeline


def run(input_path: str, output_path: str, sample: Optional[int] = None, spark_master: str = "local[*]") -> None:
    LOGGER.info("Starting Lab 2 pipeline")
    try:
        from pyspark.sql import SparkSession
    except Exception as exc:  # pragma: no cover
        LOGGER.error("pyspark is not installed or not importable. Install pyspark to run this script.")
        raise

    spark = SparkSession.builder.master(spark_master).appName("Lab2-Pipeline").getOrCreate()
    LOGGER.info("Spark session started: %s", spark.sparkContext.applicationId)

    # Read JSON-lines — infer schema
    LOGGER.info("Reading input: %s", input_path)
    df = spark.read.json(input_path)
    LOGGER.info("Read %d records", df.count())

    if sample:
        LOGGER.info("Sampling %d records for quicker run", sample)
        df = df.limit(sample)
        LOGGER.info("Sampled %d records", df.count())

    if "text" not in df.columns:
        # try common alternatives
        for c in ["content", "body", "text_raw"]:
            if c in df.columns:
                df = df.withColumnRenamed(c, "text")
                LOGGER.info("Renamed column %s to 'text'", c)
                break

    # Basic filter to drop null/empty
    df = df.filter(df.text.isNotNull())

    pipeline = build_pipeline(input_col="text", output_col="features")
    LOGGER.info("Fitting pipeline on data")
    model = pipeline.fit(df)
    LOGGER.info("Pipeline fit complete")

    LOGGER.info("Transforming data")
    result = model.transform(df).select("text", "features")

    LOGGER.info("Writing results to %s", output_path)
    os.makedirs(output_path, exist_ok=True)
    result.write.mode("overwrite").parquet(output_path)

    LOGGER.info("Results saved. Stopping Spark session")
    spark.stop()
    LOGGER.info("Done")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run Lab 2 Spark ML pipeline (Lab 17 in submission)")
    p.add_argument("--input", required=True, help="Input JSON lines file (C4).")
    p.add_argument("--output", required=True, help="Output directory (Parquet).")
    p.add_argument("--sample", type=int, default=None, help="Limit to N rows for quick run")
    p.add_argument("--master", default="local[*]", help="Spark master (default local[*])")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args.input, args.output, sample=args.sample, spark_master=args.master)
