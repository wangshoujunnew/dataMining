# encoding=utf-8
from spark_env import *
import json
import os
import datetime

# 应对疫情措施
# run_date = "2020-01-20"
save_date = datetime.datetime.now().strftime("%Y-%m-%d")
run_date = save_date.replace("-", "")
config = {
    "save_path": "/user/tujiadev/data/rankdata_v2/price_range/{}".format(save_date)
}


class PriceRange:
    """
    价格区间订单占比
    """

    def sql(self):
        sql = """
        with cityOrder as (
            select 
                city_id,count(1) as c 
            from appd_common.appd_order where is_success_order = 1 and create_date > date_add("{0}",-60)
            group by city_id
        ) 
        select 
            T.city_id,
            price_range,
            round(price_range_count / co.c,7) as rate
        from (
            select 
                city_id, price_range, 
                count(1) as price_range_count
            from 
            (
            select 
                city_id,
                if(cast((ld_gmv / order_room_night_count) / 100 as int) >100, 100, cast((ld_gmv / order_room_night_count) / 100 as int)) as price_range
            from appd_common.appd_order
            where is_success_order = 1 and create_date > date_add("{0}",-60)
            ) order_t 
            group by city_id, price_range
        ) T 
        left join cityOrder co on T.city_id = co.city_id
        """.format(run_date)
        print(sql)
        order_rdd = spark.sql(sql).rdd.map(lambda e: (e[0], e))

        def generate_json(lines):
            for line in lines:
                result = {}
                (city_id, price_range_list) = line
                for row in price_range_list:
                    result[str(row[1])] = row[2]
                yield '\t'.join([city_id, json.dumps(result)])

        order_rdd.groupByKey().mapPartitions(generate_json).repartition(1).saveAsTextFile(config['save_path'])

    def run(self):
        self.sql()


mainRun = PriceRange()
mainRun.run()
