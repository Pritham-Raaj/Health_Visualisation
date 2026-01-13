import os
os.environ.pop('CONTAINER_ID', None)

from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env file

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, coalesce, lit, trim, upper, avg,
    sum as spark_sum,
    round as spark_round
)
from pyspark.sql.types import BooleanType, DoubleType
# Setting up Hadoop and AWS sdk packages to connect to an S3 instance and load data into the local system
spark = SparkSession.builder \
    .appName("S3DataAnalysis") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
    .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.getenv('AWS_REGION', 'ap-south-1')}.amazonaws.com") \
    .getOrCreate()
# Load data from S3
raw_df = spark.read.format('csv') \
    .option('header', True) \
    .option('inferSchema', True) \
    .load('s3a://pritham-heartdata/heart_disease_uci.csv')

print("=== EXTRACT: Raw Data ===")
print(f"Total Records: {raw_df.count()}")
print(f"Total Columns: {len(raw_df.columns)}")
raw_df.printSchema()