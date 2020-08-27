# encoding=utf-8
"""常用的sql"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType, LongType, StructType
import pandasql
import numpy as np
import pandas as pd
import os
schema = [("name", ""), ("age", 0.0)]
data = [("a", 1), ("b", 2)]

pysqldf = lambda sql: pandasql.sqldf(sql, globals()) # sql 查询引擎
test_df = pd.DataFrame(np.array([[1,2], [3,4]]))
test_df.columns = ["a", "b"]
pysqldf("select * from test_df") # test_df为df的变量名称

def _build_field(name_type):
    type_map = {
        str: StructField(name_type[0], StringType(), True),
        int: StructField(name_type[0], IntegerType(), True),
        float: StructField(name_type[0], LongType(), True)
    }
    return type_map.get(type(name_type[1]))


def build_df_by_schema(rdd, schemas, tb_name=None):
    spark = SQL.spark
    df_schema = StructType([_build_field(x) for x in schemas])
    df_rdd = spark.createDataFrame(rdd, df_schema)
    if tb_name:
        df_rdd.createOrReplaceTempView(tb_name)
    else:
        return df_rdd



# 自定义udaf函数, 返回一个对象, 返回的结果不管是什么类型得到的临时表的字段都是字符串类型
def udaf_max(items):
    max_v = max(items)
    return max_v #注册方式同样是 spark.udf.register

class SQL:
    spark = None

    @staticmethod
    def order_effect():  # 订单有效
        return " is_paysuccess_order = 1 and is_effect_order = 1 "

    @staticmethod
    def dx(start, end):  # 动销, 根据动销, 一方面改进营销策略, 一方面让卖的不好的商品少进,或者下架, 将卖的好的增加库存
        # 已经销售的 / 可以销售的

        # 在此段时间房屋每天的售卖间夜
        has_sale = """
        select 
            house_id, checkin_data, sum(order_room_night_count) as order_room_night_count
        from warehouse.mar_order 
        where checkin_date between "{start}" and "{end}" and {effect} 
        group by house_id, chekin_date 
        """.format(start=start, end=end, effect=SQL.order_effect())
        can_sale = """
        -- 拿到当前早上的库存
        select 
            
        from 库存
        group by house_id
        """
        # 在此段时间可以销售的间夜量

        pass

    @staticmethod  # 用户留存
    def user_save():
        # 每个用户的登陆时间, 得从日志获取, dws_user 只存储用户的最后一次登陆时间
        """
        user_id login_time by_day:指的用户登陆了多少天了
        """
        user_login_time = """
        
        """
        # hive sql 窗口函数, 得到第一个 first_value
        case_when_by_day = list(map(lambda x: "when by_day = {0} then by_day_{0}".format(x), [0, 1, 2, 3, 4, 5, 6]))
        case_when_by_day = " \n".join(case_when_by_day)
        case_when_by_day = "case {} else 6plus end as by_day".format(case_when_by_day)

        user_save_sql = """
        with get_first(
            select 
                user_id, login_time,
                first_value() over(partition by user_id order by login_time asc) first_day
            from user_login_time 
            group by user_id
        )
        ,
        get_by_day as (
            select 
                user_id,
                datediff(login_time, first_day) by_day -- 当前已经登陆了多少天了
            from get_first
        )
        
        select 
            {0} , count(1) as user_count -- 登陆了by_day的用户有多少天
        from get_by_day
        group by {0}
        """.format(case_when_by_day)
        print(user_save_sql)
