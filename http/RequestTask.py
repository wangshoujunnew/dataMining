# encoding=utf-8
from unittest import TestCase
import requests
class Task:
    task_list = [("comment_listshow", "点评标签区域热度服务")]

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
        r = requests.post(url="http://%s:8080/%sgetCommClauseByHouseIds"%(comment_listshow.ip, context), json=data)


        r