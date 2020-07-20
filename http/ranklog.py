# python3 分析rank日志,简单分析


""" todo
1. Error类型的日志
2. INFO类型的日志
"""
import pandasql
from pandas import DataFrame

from utilself import *
import pandas as pd

pysqldf = lambda sql: pandasql.sqldf(sql, globals())  # sql 查询引擎


def get_list(f):
    table = [line.strip().split("]")[:2] for line in open_new(f)]
    return pd.DataFrame(table, columns=["thread", "info"])


df = get_list("/home/tujia/www/rank_exp/logs/checkLog.2020-07-11-22.log")
# 以INFO]结尾, 查看日志信息, 日志长度小于100的
run_record_df: DataFrame = pysqldf("select info from df where length(info) <= 100 and thread like '%INFO'")
run_record_df.head(10)

last = None


def check_distinct(lis):
    # 检测重复
    global last
    tmp = []
    for l in lis:
        if last == None:
            tmp.append(l)
        else:
            if last[:5] == l[:5]:
                pass
            else:
                tmp.append(l)
        last = l
    return tmp


check_distinct(list(filter(lambda x: not x.startswith(" 位图:"), run_record_df["info"])))[:10]
"""
发现的问题: IDC is set to [cn2' ? 
cacheDir is set to [/home/tujia/www/rank_exp/cache',: 缓存目录用来干嘛
start load remote coordFile(http://nameserver.fvt.tujia.com/xdriver-etcd/cn2/coord)': 插件文件? 用来干嘛
Socket connection established to 172.16.81.70/172.16.81.70:2181, initiating session'
create pool max total : [20 
abtest env:beta',
start init the rankService',
"""
