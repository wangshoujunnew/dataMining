# encoding=utf-8
"""spark环境"""

from pyspark.sql import SparkSession, dataframe
from pyspark import SparkConf, SparkContext, HiveContext
from pyspark.sql.types import *
from pyspark.sql import functions as F
import os

if os.getcwd().startswith("C:") or os.getcwd().startswith("D:"):
    print("[本地spark环境]")
    os.environ["SPARK_HOME"] = "D:\\Applications\\spark-2.3.4-bin-hadoop2.6"
    sparkConf = (
        SparkConf()
            .set("spark.master", "local[*]")
            .set("spark.sql.execution.arrow.enabled", "true")
            .set("spark.sql.crossJoin.enabled", "true")
            .set("spark.default.parallelism", 6)
    )
    # .set("spark.submit.deployMode", "local")

else:
    print("[hadoop 集群spark环境]")
    sparkConf = (
        SparkConf()
            .set("spark.master", "yarn")
            .set("spark.sql.crossJoin.enabled", "true")
            .set("spark.sql.execution.arrow.enabled", "true")
            .set("spark.sql.warehouse.dir", "/home/data/hive/warehouse")
            .set("spark.submit.deployMode", "client")
            .set("spark.yarn.isPython", "true")
            .set("spark.yarn.queue", "data_ai")
            .set("spark.default.parallelism", 300)
    )

spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
sc = SparkContext.getOrCreate()


task_name = "pyspark"
