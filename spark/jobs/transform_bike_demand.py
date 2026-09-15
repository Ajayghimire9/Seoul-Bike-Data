from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("StreamForgeBikeDemand").getOrCreate()

source = "Dataset/SeoulBikeData.csv"
out = "artifacts/spark/bike_demand"

df = spark.read.option("header", True).option("inferSchema", True).csv(source)

fact = (
    df.withColumn("timestamp", F.to_timestamp("Date", "dd/MM/yyyy"))
      .withColumn("timestamp", F.expr("timestamp + make_interval(0, 0, 0, 0, Hour, 0, 0)"))
      .withColumn("day_of_week", F.dayofweek("timestamp"))
      .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]))
      .withColumnRenamed("Rented Bike Count", "rented_bike_count")
      .select("timestamp", "day_of_week", "is_weekend", "rented_bike_count", "Hour", "Seasons", "Holiday")
)

fact.write.mode("overwrite").parquet(out)
spark.stop()
