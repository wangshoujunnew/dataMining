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
class unitRank(TestCase):
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
        "bucket": "G",
        "checkinDate": "2020-06-18",
        "checkoutDate": "2020-06-19",
        "peopleNum": 0,
        "bedNum": 0,
        "channel": 8,
        "originchannel": 8
    }
    # c = {"userId": 121534595}
    c = {}
    url = url1

    def test_rankv2(self):
        debug = "true"
        param = "?b={}&c={}&debug={}".format(json.dumps(unitRank.b), json.dumps(unitRank.c), debug)
        r = requests.get("{}/rankv2{}".format("http://172.31.90.46:6070", param))

        print(r.status_code)
        j = json.loads(r.text)
        j
        # print(r.text)

    def testBaidu(self):
        requests.get("http://www.baidu.com")

    def test_epSelect(self):
        """
        解释性服务 rankv2 接口
        :return:
        """
        param = "?sessionList={}".format('{"sessionId": "2345","unitIds": [114906]}')
        # param = "?sessionList={}".format('{"sessionId":"b5574edd-ff80-4b8d-9024-568bc368fdff","unitIds":[8898086,38908213,3097178,35374088,11700627,44997532,7758185,10137912,18735312,44682499,81416,71198,18113684,8812784,44107498,39211404,1604778,15380807,19728997,45032551,103841,10544521,14627488,10701797,22759479,303882,10676730,45116329,170259,16893311,1003267,44048110,12414473,16452668,43499590,8481705,14627866,43556997,8813029,44799751,11200743,45176839,1045345,7660514,44981848,5207162,5968846,43994917,16927674,202115,9584597,9585010,14627285,61459,173395,44106560,10806363,10013368,6823224,21749323,45032971,45017260,10806300,1117916,40718238,1082212,45036913,355377,39547299,44442766,43541870,22160321,1073138,16748866,10479533,1162693,44962705,12299582,2468270,9290632,44271898,171841,11991813,15599326,12490003,43393540,9585234,17111970,44184604,42844740,1184265,44981539,9378188,14785982,9950907,10717001,1213511,9926197,8825517,22073493,17204440,43967134,44981878,16893045,10197279,20094999,31673475,9584933,45034033,10091425,15480158,16739787,44292496,10912105,16887984,44871670,45100414,41122803,43369488,9585241,3865073,13998398,43519827,1081526,10137681,80577,13504457,19779894,13580596,301843,45162697,11253474,16739045,45165796,45162910,11042788,12904543,170829,199561,1077992,9495851,45126205,8939603,8316372,16893409,7491024,448411,9926162,6563573,2841699]}')
        req = "{}/epSelect{}".format("http://172.31.90.46:6070", param)
        print(req)
        r = requests.get(req)
        print(r.status_code)
        print(r.text)
        # return r

    def test_epChange(self):
        """
        解释性服务 rankv2 接口
        :return:
        """
        sessionDetail = {
            "sessionId": "e42b5312-5c16-4d8f-9a2e-a0870137593a",
            "scence": "city",
            "bucket": "C",
            "channel": 8,
            "unitId": 11199665,
            "featureNum": 101
            # """
            # public String sessionId;
            # public String sence;
            # public String bucket;
            # public String channel;
            # public Integer unitId; 点击率
            # public Integer featureNum;
            # """
        }
        param = "?sessionDetail={}".format(json.dumps(sessionDetail))
        req = "{}/epChange{}".format("http://172.31.90.46:6070", param)
        print(req)
        r = requests.get(req)
        print(r.status_code)
        print(r.text)