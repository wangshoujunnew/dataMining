"""
用户行为Task
"""

from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from houseEmbed.UserSession import *

sc = SparkContext()

keys = ["sid", "deviceID", "userID", "unitId", "actTime", "channel", "platform", "location", "refPage", "curPage", "searchConditonStr", "distance", "unitPrice", "pos", "advertUnit", "adLabel",
        "click", "book", "order", "orderStr"]


def _parse_line(line: str):
    arr = line.split("\t")
    data_dict = {}
    for index, key in enumerate(keys[:17]):  # 17 为click位置
        data_dict[key] = arr[index]

    if len(arr) >= 20:
        for index, key in enumerate(keys[17:]):
            data_dict[key] = arr[17 + index]

    return data_dict



save_text_file = "/home/shoujunw/embed"
input_text_file = "/data/rankdata/user_action_on_unit/tujia/20200526/part-r-00099"
sc.textFile(input_text_file).map(lambda x: _parse_line(x)).map(lambda x:
                                                (x["sid"], x)).groupByKey().mapValues(lambda x: _parse_session(x))\
                                                .filter(lambda x: x.is_book).map(lambda x: x.__str__).saveAsTextFile(save_text_file)

input_text_file = "/data/rankdata/user_action_on_unit/tujia/20200526"
key_rdd = sc.textFile(input_text_file).map(lambda x: UserSession.parse_line(x)).map(lambda x: UserSession.user_group(x))
