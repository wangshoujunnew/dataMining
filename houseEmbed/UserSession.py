# encoding=utf-8
import json
import random

# 日志的含义
from pyspark import SparkContext
from pyspark.sql import SparkSession

user_action_on_unit_fields = ["sid", "deviceID", "userID", "unitId", "actTime", "channel",
                              "platform", "location", "refPage", "curPage",
                              "searchConditonStr", "distance", "unitPrice", "pos",
                              "advertUnit", "adLabel",
                              "click", "book", "order", "orderStr"]


class LineBean:
    def parse_line(self, line):
        """
        解析一行成为对象
        """
        arr = line.split("\t")
        self.sid, self.deviceID, self.userID, self.unitId, self.actTime, self.channel, self.platform, \
        self.location, self.refPage, self.curPage, self.searchConditonStr, self.distance, self.unitPrice, \
        self.pos, self.advertUnit, self.adLabel, self.click, self.book, self.order = arr[:19]
        if arr[18] == "true":
            self.orderStr = arr[19]
        else:
            self.orderStr = None

    @staticmethod
    def parse_cond(cond_str):  # 解析请求条件
        cond_bean = {}
        try:
            cond_str = json.loads(cond_str)
            cond_bean["sence"] = "land" if "landmark" in cond_str.keys() else "city"
        except:
            return None


class UserSession(object):
    """用户日志结构"""
    show, click, book = 0, 1, 4
    # 采样类型
    pos, neg_session, neg_city, neg_other_city = 1, -3, -2, -1
    # 城市对应的房屋id, 需要广播
    city_house_id = {}
    sc = None
    spark = None
    error_ac = None # 错误累加器

    @staticmethod
    def show_sql(sql):
        print("==" * 50)
        print(sql)
        print("==") * 50
        return sql

    @staticmethod
    def allcity_house():  # 广播城市下的所有房子, 做城市负采样
        sql = "select house_city_id, house_id from warehouse.mar_house where house_is_active =1 and hotel_is_active =1"
        result = UserSession.spark.sql(UserSession.show_sql(sql)).rdd.groupByKey().collectAsMap()
        print("[广播所有城市房屋ID]")
        broad_value = UserSession.sc.broadcast(result)
        #for city in broad_value.keys():
        #    broad_value[city] = list(broad_value[city])
        return broad_value

    @staticmethod
    def order_city(start, end):  # 订单对应的城市
        sql = "select cast(order_no as string) as order_no, city_id from warehouse.mar_order where create_date between '{}' and '{}'".format(start, end)
        result = UserSession.spark.sql(UserSession.show_sql(sql)).rdd.collectAsMap()
        print("[广播订单id对应的城市]")
        broad_value = UserSession.sc.broadcast(result)
        return broad_value

    # 通过用户的session分组
    @staticmethod
    def user_group(data_dict):
        if type(data_dict) == str:
            data_dict = json.loads(data_dict)
        return (data_dict["sid"], data_dict)

    # 将一行解析成我user_action_on_unit_fileds字段对应的值
    @staticmethod
    def parse_line(line):
        arr = line.split("\t")
        #user_action_on_unit_fields = ["sid", "deviceID", "userID", "unitId", "actTime", "channel","platform", "location", "refPage", "curPage","searchConditonStr", "distance", "unitPrice", "pos","advertUnit", "adLabel","click", "book", "order", "orderStr"]
        data_dict = dict(zip(user_action_on_unit_fields[:17], arr[:17]))
        #return arr
        #for index, key in enumerate(user_action_on_unit_fields[:17]):  # 17 为click位置
        #    data_dict[key] = arr[index]
        if len(arr) >= 20:
            for key, value in dict(zip(user_action_on_unit_fields[17:], arr[17:])).items():
                data_dict[key] = value
        return data_dict

    # 将用户分组后的list组装成用户数据结构
    @staticmethod
    def parse_session(items, order_city_dict):
        # return items
        order_city_dict = order_city_dict.value

        userSession = UserSession(items[0]["userID"], items[0]["actTime"], items[0]["sid"])
        userSession.user_id = items[0]["userID"]
        userSession._parse_list_to_session(items, order_city_dict)
        return userSession

    # 随机选取n个房子
    @staticmethod
    def random_ext_id(ids, k=10):
        if len(ids) < 10:
            return ids
        else:
            return random.sample(ids, k)

    # 校验数据类型的正确性
    @staticmethod
    def type_ok(data_dict):
        try:
            data_dict["unitId"] = int(data_dict["unitId"])
            return data_dict
        except:
            return None

    def __init__(self, user_id, acttime, sid):
        self.user_id = user_id
        self.acttime = acttime
        self.houseIds = []  # [(id, 样本类型, pos)]
        self.conditions = []  # 搜索条件
        self.is_book = False
        self.sid = sid
        self.order_info = None
        self.book_house = None
        self.city_id = None
        self.book_pos = -1

    def _parse_list_to_session(self, items, order_city_dict):
        # 将按照用户groupby之后的日志解析到Session对象中
        for item in items:
            item = UserSession.type_ok(item)

            if item is None:
                continue

            if "pos" not in item.keys():
                continue
            if item["pos"] == "-1":
                self.conditions.append(item["searchConditonStr"])
            else:
                # , item["pos"], UserSession.book if item.get("book", None) == "true" else UserSession.click)
                # 如果有多个订单, 我只认最后一个订单
                if item.get("order") == "true":
                    # sample.append(UserSession.pos)
                    self.is_book = True
                    self.order_info = json.loads(item["orderStr"])
                    self.city_id = order_city_dict.get(self.order_info["orderNo"])
                    self.book_house = item["unitId"]
                    self.book_pos = item["pos"]
                else:
                    self.houseIds.append((item["unitId"], UserSession.neg_session, item["pos"]))
        if self.book:
            self.houseIds.append((self.book_house, UserSession.pos, self.book_pos))

    def _user_show_house(self):
        return "[userId: {}, bookId: {}, showHouse: {}]".format(self.user_id, self.book_house, self.houseIds)

    def to_str(self):
        return "userId: {}, session: {}, condition: {}".format(self.user_id, self.houseIds, self.conditions)

    def train_data(self, all_city_house_dict):
        # Sesion 负采样
        all_city_house_dict = all_city_house_dict.value
        negs_session = filter(lambda x: x[1] != UserSession.pos, self.houseIds)
        session_sample = UserSession.random_ext_id(list(negs_session), k=10)
        # print(session_sample)
        # session_sample = list(map(lambda x: x, ))
        # 同城负采样
        city_sample = []
        other_city_sample = []
        if self.city_id or True:  # 要得到城市需要用底单表关联之后好些, 从日志得到城市不好, 但是我需要通过orderInfo知道订单ID
            city_houseid_list = list(all_city_house_dict[self.city_id])
            city_sample = UserSession.random_ext_id(city_houseid_list, 10)

            # 从随机的10个城市随机抽取1个id
            for city in UserSession.random_ext_id(list(filter(lambda x: x != self.city_id, all_city_house_dict.keys())), 10):
                other_city_sample.append(UserSession.random_ext_id(list(all_city_house_dict[city]), 1)[0])

        data_list = []
        if self.is_book:
            data_list.append((self.book_house, UserSession.pos, self.book_pos))

        for i in session_sample:
            data_list.append(i)

        return data_list + session_sample + [(x, UserSession.neg_city, -1) for x in city_sample] + [(x, UserSession.neg_other_city, -1) for x in other_city_sample]

    def __str__(self):
        return self._user_show_house()
