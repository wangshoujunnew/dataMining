"""
常用的http请求
"""
import requests


class HTTP:

    @staticmethod
    def split_bucket(param=None):
        if param == None:
            pass
        else:
            param = {
                "bucketItemList": [
                    {
                        "lab_key": "rankOperate",
                        "rateMap":
                            {
                                "B": 0.33,
                                "C": 0.67
                            },
                        "specialMap": {
                            "0315dcc7-8586-3c2a-b9e5-a4ead08ddab5": "E",
                            "03942079C398A8917FC472ECEEC31EF0": "E",
                            "0D2098ED96FA966E60765642E95D717A": "B",
                            "12001153810182033194": "B"
                        },

                        "comment": "dy_0605"
                    }
                ]
            }

        host_url = "http://abtest.corp.tujia.com/abtest/modifybucketconfig1"

        content = requests.post(host_url, data=param)
        print(f"request param {param} \n result: {content}")

