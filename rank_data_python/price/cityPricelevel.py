#encoding=utf-8
"""
城市价格指数
"""
from pyspark import StorageLevel
from spark_env import *
import json
import sys

config = {}
hdfsPrefix = "/user/tujiadev/" if env == "qunar" else ""

# 跑哪天的roombaseinfo文件
origin_date=sys.argv[1]
config['run_data'] = origin_date.replace("-", "")
# roombaseinfo hdfs的路径
config['roombaseinfo'] = "%s/data/rankdata_v2/house_base_info/%s" % (hdfsPrefix, config['run_data'])
# 数据保存路径
config['save_path'] = "%s/data/rankdata_v2/city_price_level/%s" % (hdfsPrefix, config['run_data'])
# config['house_rdd_save_path']

# 广播变量
broadcast_list = {}

def house_year_order():
    """
    房屋一年的订单
    """
    year_orders = spark.sql(
        """
        select
            house_id,
            avg(ld_gmv / order_room_night_count) as avg_ld_gmv
        from
            appd_common.appd_order
            where
            create_date > date_add(current_date, -365)
            and is_success_order = 1
            group by house_id
        """
    ).rdd.map(lambda e: (int(e[0]), float(e[1]))).collectAsMap()

    broadcast_list['year_orders_any'] = sc.broadcast(year_orders)

def load_showprice():
    sql = "select unitId, avg_f from dw_algorithm.list_price_day where create_date = '{}'"\
        .format(origin_date)

    print(sql)
    finalPrice_map = spark.sql(sql).rdd.collectAsMap()

    print("[广播finalPrice]")
    print(len(finalPrice_map))
    broadcast_list["finalPrice"] = sc.broadcast(finalPrice_map)


def get_house_info():
    lines = sc.textFile(config['roombaseinfo']).repartition(500)
    load_showprice() # 加载展现价格 
    order_map = broadcast_list['year_orders_any'].value
    finalPrice = broadcast_list['finalPrice'].value
    def extract_base_info(houses):
        for house in houses:
            # try:
            house_id, info = house.strip().split('\t')
            house_id = int(house_id)
            info = json.loads(info)
            need_info = {
                'house_id': house_id,
                'avg_price': info['avgPrice'],
                'city_id': info['cityID'],
                'area': info['area'],
                'persons': info['checkInNum'],
                'is_haiwai': int(info['ifHaiwai']),
                'avg_ld_gmv': order_map.get(house_id, 0.0)
            }
            now_finalPrice = finalPrice.get(house_id, None)
            if now_finalPrice:
                need_info["avg_price"] = now_finalPrice
            # [need_info, 0, 0] 房屋信息, 价格 ,面积价格, 人数价格, reduce 结果
            if need_info['is_haiwai'] == 1:
                need_info['persons'] += info['addPeopleNum']
            if need_info['area'] > 0 and need_info['persons'] > 0:
                area_price = need_info['avg_price'] * 1.0 / need_info['area']
                person_price = need_info['avg_price'] * 1.0 / need_info['persons']
                yield (need_info['city_id'], [need_info, need_info['avg_price'], area_price, person_price])

    return lines.mapPartitions(extract_base_info)


def price_filter(price_obj):
    if price_obj['price_area'] <= 10 and price_obj['price_person'] <= 10:
        return price_obj
    else:
        return None
    # 是否只跑监控
    if self.config.get('monitor',False) is True:
        file_rdd = self.sc.textFile(self.config['save_path']).map(lambda e:eval(e))

        df_rdd = file_rdd.map(lambda r: price_filter({'price_area': r[1][0],'price_person': r[1][1]})).filter(lambda e:e is not None)
        spark.createDataFrame(df_rdd).describe(['price_area','price_person']).show()
        return

def local_aboard(house):
    info = house[0]
    ld_gmv = info['avg_ld_gmv']
    if info['is_haiwai'] == 0: # 国内
        if house[1] >= 20 and house[3] <= 800 and house[2] <= 200: # 每人价格<=800
            return True
        elif 800 <= house[3] <= 2000 and (house[1] <= 8 * ld_gmv and ld_gmv > 0): # 间夜lggmv
            return True
        else:
            return False
    else:
        if house[1] >= 20 and house[3] <= 5000 and house[2] <= 200:
            return True
        else:
            return False

def price_sum(lines):
    for line in lines:
        (city_id, houses) = line
        house_count = 0
        price_area, price_person = [0.0, 0.0]
        no_filter_price_area, no_filter_price_person = [0.0, 0.0]
        filter_houses = map(lambda e: (e[2], e[3], e[1]),filter(local_aboard, houses)) # 取没平米价格, 和没人价格
        filter_rest_len = len(filter_houses)
        if filter_rest_len > 0:
            [price_area, price_person, avg_price_all] = list(reduce(lambda x,y: [ x[0] + y[0], x[1] + y[1], x[2] + y[2] ], filter_houses))
            yield (city_id, [price_area / filter_rest_len, price_person / filter_rest_len, '{}_filter'.format(filter_rest_len), avg_price_all / filter_rest_len])
        else:
            no_filter_houses = map(lambda e: (e[2], e[3], e[1]),houses)
            no_filter_houses_len = len(no_filter_houses)
            [price_area, price_person , avg_price_all] = list(reduce(lambda x,y: [ x[0] + y[0], x[1] + y[1], x[2] + y[2] ], no_filter_houses))
            yield (city_id, [price_area / no_filter_houses_len, price_person / no_filter_houses_len, '{}_nofilter'.format(no_filter_houses_len), avg_price_all / no_filter_houses_len])


house_year_order()
house_rdd = get_house_info()
# ==============
#house_rdd.saveAsTextFile(self.config['house_rdd_save_path'])
# ==============
house_rdd.persist(StorageLevel.MEMORY_AND_DISK)
# city_data = house_rdd.reduceByKey(price_sum).mapPartitions(price_select).collectAsMap()
city_data = house_rdd.groupByKey().mapPartitions(price_sum).collectAsMap()

# 存储到文件中
broadcast_list['city_data_any'] = sc.broadcast(city_data)

city_datas = broadcast_list['city_data_any'].value
def compare_city(lines):
    for line in lines:
        (city_id, [info, avg_price, area_price, person_price]) = line
        city_data = city_datas.get(city_id, None)
        if city_data is not None and city_data[0] > 0 and city_data[1] > 0:
            yield '\t'.join([str(info['house_id']), str(area_price / city_data[0]), str(person_price / city_data[1]), str(avg_price)])


print("saveAsTextFile ... ")
shell("hadoop fs -rm -r -f %s" % config["save_path"])
house_rdd.mapPartitions(compare_city).repartition(10).saveAsTextFile(config['save_path'])
