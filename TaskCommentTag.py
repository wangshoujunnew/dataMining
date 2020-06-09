from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext


class SparkEnv:
    @staticmethod
    def import_env():
        conf = SparkConf().setMaster('yarn')
        conf.set('hive.metastore.uris', 'thrift://l-hadoop-around2.data.cn2:9083')
        sc = SparkContext(conf=conf)
        hc = HiveContext(sc)
        return sc, hc


sc, hc = SparkEnv.import_env()

data_rdd = hc.sql("select * from warehouse.mar_order limit 1").rdd.map(lambda x: x[0])
data_rdd.saveAsTextFile()
print(data_rdd.take(10))