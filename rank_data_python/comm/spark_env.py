# encoding=utf-8
from pyspark.sql import SparkSession, dataframe
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
from pyspark.sql import functions as F
import os

# 设置环境变量 python_python, spark_home, pyspark_path等 todo 
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python"
# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"

from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType

schema = [("name", ""), ("age", 0.0)]
data = [("a", 1), ("b", 2)]


def _build_field(name_type):
    type_map = {
        str: StructField(name_type[0], StringType(), True),
        int: StructField(name_type[0], IntegerType(), True),
        float: StructField(name_type[0], DoubleType(), True)
    }
    return type_map.get(type(name_type[1]))


# 构建schema
def build_df_by_schema(rdd, schemas, tb_name=None):
    # spark = SQL.spark
    df_schema = StructType([_build_field(x) for x in schemas])
    df_rdd = spark.createDataFrame(rdd, df_schema)
    if tb_name:
        df_rdd.createOrReplaceTempView(tb_name)
    else:
        return df_rdd


env = "qunar"


def shell(cmd):
    print(cmd)
    return os.system(cmd)


# 172.31.84.108
def switch_env():
    import socket
    name = socket.getfqdn(socket.gethostname())
    addr = socket.gethostbyname(name)
    print(addr)
    if "addr" == "108":
        env = "tujia"


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
) if env == "tujia" else (
    SparkConf().set(
        "warehouselocation",
        "",
    ).set(
        "spark.sql.warehouse.dir",
        "",
    ).set("hive.metastore.uris", "")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("hive.exec.dynamic.partition", "true")
        .set("hive.exec.dynamic.partition.mode", "nonstrict")
        .set("hive.exec.max.dynamic.partitions", "5000")
        .set("hive.exec.max.dynamic.partitions.pernode", "5000")
        .set("spark.kryoserializer.buffer.max", "2047M")
        .set("spark.broadcast.compress", "true")
)
spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
sc = SparkContext.getOrCreate()

task_name = "pyspark"
