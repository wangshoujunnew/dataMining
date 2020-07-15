# encoding=utf-8
from unittest import TestCase
import requests
import json


# 点评标签展现服务
class comment_listshow(TestCase):
    """当前测试ip, 所有可测试的ip, 所有测试的接口(参数提示)"""
    ip = "172.29.24.207"
    ip = "localhost"
    ips = ["172.29.24.207", "172.31.135.67", "10.95.156.90", "10.95.156.99"]
    interface = ["getCommClauseByHouseIds"]

    def test_getCommClauseByHouseIds(self):
        # assert data, "data is not None: ps: {}".format({"houseIds": [100768, 10090480, 10156490, 103050]})
        data = {"houseIds": [100768, 10090480, 10156490, 103050]}

        context = "commtag_listshow/" if comment_listshow.ip == "localhost" else ""
        r = requests.post(url="http://%s:8080/%sgetCommClauseByHouseIds" % (comment_listshow.ip, context), json=data)
        r


# 点评标签
class Comtag:
    ips = ["10.95.157.64", "10.95.157.65"]
    ip = ips[0]

    @staticmethod
    def post(data=None):
        url = url = "http://{}:8016".format(Comtag.ip)
        r = requests.post(url=url, json={"data": ["服务态度很好呀哈哈哈" if data is None else data]})
        return r


# 排序,线上
class unitRank:
    def testDemo(self):
        print("this is a test")

    url1 = "http://172.31.84.110:6070"
    # 1. b中的reqTime是用来干嘛的, 如果没有则给当前的请求时间, 知道了: 是用来提供模型训练的离线服务的
    # distance 的作用, 可能输入的distance > 5km的时候用, 为啥要这样呢? >5km的时候为啥要设置地标名称为空呢
    # 入住日期和离店日期为空的话就默认今住明离(任意为空时), 入住时间,当天23点59分, 并计算一个入住时间到请求时间的天数
    # chanelWeight 是干嘛的
    # roomChannelWeight 干嘛的
    # 城市地标场景下的默认热配置 CityLandmarkConfig
    # List<Integer> sortUnitIds = rankService.getSortCityIds(bParam.getChannel(), cityId) 这是什么排序
    b = {
        "cityId": 8,
        "coord": "",
        "landmarkName": "",
        "sort": "recommend",
        "manual": 1,
        "bucket": "B",
        # "checkinDate": "2020-06-18",
        # "checkoutDate": "2020-06-19",
        "peopleNum": 0,
        "bedNum": 0,
        "channel": 1,
        "originchannel": 1
    }
    c = {"userId": 121534595}
    url = url1

    @staticmethod
    def test_rankv2():
        debug = "true"
        param = "?b={}&c={}&debug={}".format(json.dumps(unitRank.b), json.dumps(unitRank.c), debug)
        r = requests.get("{}/rankv2{}".format(unitRank.url, param))
        print(r.status_code)
        return r
