#encoding=utf-8
"""
自定义累加器
"""
from pyspark import SparkContext, AccumulatorParam

class StrAccumulator(AccumulatorParam):

    def zero(self, value):
        return ""

    def addInPlace(self, value1, value2):
        if value1 == "":
            return value2

        if value2 != "":
            return "{}\n{}".format(value1, value2)
        else:
            return value1
