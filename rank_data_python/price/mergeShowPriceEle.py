# encoding=utf-8
# 融合展现价格因子

from spark_env import *
import json
import sys

create_date = "2019-11-01"
create_date = sys.argv[1]
createdate = create_date.replace("-", "")
local_sc = sc

hdfsPrefix = "/user/tujiadev" if env == "qunar" else ""

roombaseHdfsDir = "%s/data/rankdata_v2/house_base_info/%s" % (hdfsPrefix, createdate)
cityprice_path = "%s/data/rankdata_v2/city_price_level/%s" % (hdfsPrefix, createdate)
regionprice_path = "%s/data/rankdata_v2/regionpriceLevelfinalPrice/%s" % (hdfsPrefix, create_date)
savepath = "%s/data/rankdata_v2/house_base_info_forshowprice/%s" % (createdate, createdate)

print("======", roombaseHdfsDir, cityprice_path, regionprice_path, savepath)


def load_baseInfo():
    return local_sc.textFile(roombaseHdfsDir).map(lambda x: x.split("\t")).map(lambda x: (x[0], json.loads(x[1])))


def load_finalPrice():
    df = spark.sql("select cast(unitId as string) as unitId, avg_f from dw_algorithm.list_price_day where create_date = '{}'".format(create_date))
    return df.rdd


def load_pricelevel():
    def type_check(lis, types):
        new_lis = []
        for t, number in zip(types, lis):
            try:
                if t == float:
                    new_lis.append(float(number))
                elif t == int:
                    new_lis.append(int(number))
                else:
                    raise Exception("no hand this data type")
            except:
                raise Exception("type error {} ===== ".format(lis))

        return new_lis

    cityprice = local_sc.textFile(cityprice_path).map(lambda x: x.split("\t")).filter(lambda x: len(x) >= 3).map(lambda x: (x[0], type_check([x[1], x[2]], [float, float])))
    regionprice = local_sc.textFile(regionprice_path).map(lambda x: x.split("\t")).filter(lambda x: len(x) >= 3).map(lambda x: (x[0], type_check([x[1], x[2]], [float, float])))
    return cityprice, regionprice


def check_count(*lis):
    for i in lis:
        print(type(i))
        count = i.count()
        print("======count", count)
        if count <= 0:
            print("debug output: rdd 数量为0")
            sys.exit(0)


def line_hand(x):
    (house_id, (((baseinfo, f_price), c_price), r_price)) = x
    if f_price:
        baseinfo["showPrice"] = f_price
    if c_price:
        if baseinfo["priceLevel"] is None:
            baseinfo["priceLevel"] = {}

        baseinfo["priceLevel"]["cityPerIndex"] = c_price[1]
        baseinfo["priceLevel"]["cityPerMeterIndex"] = c_price[0]

    if r_price:
        baseinfo["regionPrice"] = r_price[0]
        baseinfo["regionPricePerson"] = r_price[1]

    return "{}\t{}".format(house_id, json.dumps(baseinfo))


def main():
    roombase = load_baseInfo()
    finalPrice = load_finalPrice()
    cityprice, regionprice = load_pricelevel()

    check_count(finalPrice, cityprice, regionprice)
    join_rdd = roombase.leftOuterJoin(finalPrice).leftOuterJoin(cityprice).leftOuterJoin(regionprice)
    join_rdd.map(lambda x: line_hand(x)).saveAsTextFile(savepath)

main()