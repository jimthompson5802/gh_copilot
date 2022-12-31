# initialize spark session with hadoop configuration and hive support
# dynamic allocation is enabled by default with minimum 2 executors and maximum 8 executors
# spark.executor.cores is set to 2 by default
# spark.executor.memory is set to 2G by default
# spark.driver.memory is set to 1G by default
# spark.driver.cores is set to 1 by default

# add spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("demonstration") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "8") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "2G") \
    .config("spark.driver.memory", "1G") \
    .config("spark.driver.cores", "1") \
    .enableHiveSupport() \
    .getOrCreate()

# add spark context
sc = spark.sparkContext
