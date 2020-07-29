# encoding=utf-8
import json
import datetime
import os

"""
房屋基础信息处理
"""
data_dir = "/home/tujia/rank_data_v2.0/data/base"


def date_add(current_date, add_n, format='%Y%m%d'):
    if current_date is not None:
        format_str = '%Y-%m-%d' if '-' in current_date else '%Y%m%d'
        now = datetime.datetime.strptime(current_date, format_str)
    else:
        now = datetime.datetime.now()
    delta = datetime.timedelta(days=add_n)
    n_days = now + delta
    return n_days.strftime(format)


setting = {
    "tmp_delete": False,
    "run": ['price_range',
            'region_price_level',
            'city_price_level',
            'comment_tag',
            'yunyin_score',
            'md_online2date',
            "house_show_price_threshold_new",
            'info_score',
            'u_add',
            'orderPrice',
            'showPrice'],
    "origin_room_path": "{}/roombaseinfoTmp".format(data_dir.replace('/base', '')),
    "save_path": "{}/roombaseinfo".format(data_dir.replace('/base', '')),

    # ======================
    # 价格区间
    "price_range_path": "{}/price_range.res".format(data_dir),
    # 评论标签
    "comment_tag_path": "{}/comment_tag.res".format(data_dir),
    # 区域价格指数
    "region_price_path": "{}/region_price_level.res".format(data_dir),
    # 城市价格指数
    "city_price_path": "{}/city_price_level.res".format(data_dir),
    # 运营分
    "yunying_score_path": "{}/yunying_score.res".format(data_dir),

    # 信息分(运营维护)
    "infoscore_path": "{}/infoscore.res".format(data_dir),

    # 房屋周围的价格倍率
    "house_show_price_threshold_new_path": "{}/house_show_price_threshold.res".format(data_dir),

    # 房屋优+
    "uadd_path": "{}/uAdd.res".format(data_dir),

    # 房屋最近14天的订单价格
    "orderPrice_path": "{}/orderpriceLast14.res".format(data_dir),

    "showPrice_path": "{}/finalPrice.res".format(data_dir),

    "pull_data": {
        # 优选pro和运营分, hdfs位置, 拉取之后执行的命令, 存储到108的位置
        "perfer_pro_and_yunying_score": ("/home/data/hive/warehouse/dw_algorithm.db/v2_additional/*", " sed 's//\t/g'  ", "{}/yunying_score.res".format(data_dir)),
        # 价格区间订单占比
        "price_range": ("/data/rankdata_v2/price_range/{}/*".format(date_add(None, -1)), None, "{}/price_range.res".format(data_dir)),
        # 区域价格指数
        "region_price_leve": ("/data/rankdata/regonpriceLevel/part1/*", " grep -v NaN ", "{}/region_price_level.res".format(data_dir)),
        "city_price_leve": ("/data/rankdata_v2/city_price_level/{}/*".format(date_add(None, -1)), None, "{}/city_price_level.res".format(data_dir)),
        # "comment_tag": ("/data/rankdata_v2/commenttag/{}/*".format(date_add(None, -1)), None, "{}/comment_tag.res".format(data_dir)),
        "comment_tag": ("/data/rankdata_v2/commenttag/{}/*".format("20200420"), None, "{}/comment_tag.res".format(data_dir)),

        # 拉去陈志田的数据不在python中做调用, 而是放在crontab中进行, crontab需要注意拉去的时候是否当前正在插入数据, 拉去之后需要判断一下数据量,
        # 将数据房屋到data_dir目录下, 不能采用text的方式获取, 因为是paquent文件
        # "house_show_price_threshold_new": ("/home/data/hive/warehouse/app_bigdata.db/house_show_price_threshold_new/*", " sed 's//\t/g'  ", "{}/house_show_price_threshold_new.res".format(data_dir))
    }
}


class PushData:
    def __init__(self):
        pass

    def exe_shell(self, cmds):
        for cmd in cmds:
            print('shell 命令执行: {}'.format(cmd))
            status = os.system(cmd)
            if status > 0:
                raise Exception("shell 命令执行失败")

    def pull_data(self):
        """
        从hdfs上拉取数据
        """
        for k, last_cmd in setting['pull_data'].items():
            print('start pull data {} from {}'.format(k, last_cmd[0]))
            if last_cmd[1] == None:
                self.exe_shell(["hadoop fs -text {} > {}".format(last_cmd[0], last_cmd[2])])
            else:
                self.exe_shell(["hadoop fs -text %s | %s > %s" % last_cmd])


class BaseInfo(PushData):
    """
    setting[run] 需要加入那些数据, 数据处理完得销毁, 否则内存受不了
    房屋ID全部都用字符串
    """

    def __init__(self):
        self.roombases = []

    def load_to_map(self, f_path, type_='list', names=None):
        """
        加载文件成一个map, 注意这里都是字符串类型的，如需数字类型，自己转换
        names: 每行对应的名称
        """
        result = {}
        for line in open(f_path, 'r', encoding='utf-8').readlines():
            try:
                key, *info = line.strip().split('\t')
                if type_ == 'list':
                    result[key] = info
                elif type_ == 'map':
                    result[key] = dict(zip(names, info))
                else:
                    raise Exception('无效的加载类型')
            except Exception as e:
                print('error: {}, reason: {}'.format(line, e))
        return result

    def info_score(self):
        """信息分加载"""
        self.infoscore_map = {}
        infoscore_map = self.load_to_map(setting["infoscore_path"], type_="map", names=["infoScore"])
        for k, v in infoscore_map.items():
            if v["infoScore"].isdigit():
                self.infoscore_map[k] = {}
                self.infoscore_map[k]["infoScore"] = int(v["infoScore"])
            else:
                pass

        del infoscore_map
        print("信息分数量: {}".format(len(self.infoscore_map)))

    def info_score_set(self, room):
        houseId = str(room["houseId"])
        room["infoScore"] = self.infoscore_map.get(houseId, {}).get("infoScore", None)
        return room

    def showPrice(self):
        print("[final price load]")
        finalPrice_map = self.load_to_map(setting["showPrice_path"], type_="map", names=["showPrice"])
        print(list(finalPrice_map.items())[:10])
        for k, v in list(finalPrice_map.items()):
            finalPrice_map[k]["showPrice"] = float(v["showPrice"])
        self.finalPrice_map = finalPrice_map
        print("[final price size {}".format(len(self.finalPrice_map)))

    def showPrice_set(self, room):
        houseId = str(room["houseId"])
        if houseId in self.finalPrice_map.keys():
            room["showPrice"] = self.finalPrice_map[houseId]["showPrice"]
        return room

    def u_add(self):
        print("u+房屋加载")
        u_add_map = self.load_to_map(setting["uadd_path"], type_="map", names=["isuPlus"])
        for k, v in list(u_add_map.items()):
            u_add_map[k]["isuPlus"] = 1
            # if v["isuPlus"] == "21":
            #    self.u_add_map[k] = {"isuPlus": 1}
            # else:
            #    pass

        self.u_add_map = u_add_map
        print("u+数量: {}".format(len(self.u_add_map)))

    def orderPrice(self):
        print("[房屋订单价格加载]")
        orderPrice_map = self.load_to_map(setting["orderPrice_path"], type_="map", names=["orderPrice"])
        for k, v in list(orderPrice_map.items()):
            orderPrice_map[k]["orderPrice"] = float(v["orderPrice"])
        self.orderPrice_map = orderPrice_map
        print("[数量: {}]".format(len(self.orderPrice_map)))
        print(self.orderPrice_map.get("40253081", "没有这个房屋id 40253081"))

    def orderPrice_set(self, room):
        houseId = str(room["houseId"])
        if houseId in self.orderPrice_map.keys():
            room["orderPrice"] = self.orderPrice_map[houseId]["orderPrice"]
        return room

    def u_add_set(self, room):
        houseId = str(room["houseId"])
        room["isuPlus"] = self.u_add_map.get(houseId, {}).get("isuPlus", 0)
        return room

    def yunyin_score(self):
        """
        运营分添加
        """
        yunyin_score_map = self.load_to_map(setting['yunying_score_path'], type_='map', names=['tujiaChoosePro', 'opeScore'])
        # 类型转换
        for k, v in yunyin_score_map.items():
            v['tujiaChoosePro'] = int(v['tujiaChoosePro']) if v['tujiaChoosePro'].isdigit() else None
            try:
                v['opeScore'] = float(v['opeScore'])
            except:
                v['opeScore'] = None
        self.yunyin_score_dict = yunyin_score_map
        print('yunyin_score_dict.size: {}'.format(len(self.yunyin_score_dict.keys())))
        if len(self.yunyin_score_dict.keys()) < 2000000:
            raise Exception('运营分量级异常')

    def yunyin_score_set(self, room):
        houseId = str(room['houseId'])
        room['tujiaChoosePro'] = self.yunyin_score_dict.get(houseId, {}).get('tujiaChoosePro', None)
        room['opeScore'] = self.yunyin_score_dict.get(houseId, {}).get('opeScore', None)
        return room

    def price_range(self):
        """
        价格区间添加
        """
        result_info = {}
        for line in open(setting['price_range_path'], 'r', encoding='utf-8').readlines():
            try:
                city_id, price_str = line.strip().split('\t')
                price_dict = json.loads(price_str)
                result_info[int(city_id)] = price_dict
            except Exception as e:
                print('error: {}, reason: {}'.format(line, e))
        self.price_range_dict = result_info
        # print('cityId=48: {}'.format(json.dumps(result_info.get(48, None), indent=4)))
        print('price_range.size: {}'.format(len(self.price_range_dict)))
        if len(self.price_range_dict) < 500:
            raise Exception('价格区间量级异常')
        """
        设置roombase的价格区间属性
        """

    def price_range_set(self, room):
        this_room_price_range = str(min(100, int(room['avgPrice'] / 100)))
        range_value = self.price_range_dict.get(room['cityID'], {}).get(this_room_price_range, 0.0)
        room['priceRange'] = round(range_value, 5)
        return room

    def region_price_level(self):
        """
        区域价格指数
        """
        region_price_level = {}
        for line in open(setting['region_price_path'], 'r', encoding='utf-8').readlines():
            try:
                houseid, area_price_level, person_price_level = line.strip().split('\t')
                area_price_level = float(area_price_level)
                person_price_level = float(person_price_level)
                region_price_level[houseid] = {
                    'area_price_level': round(area_price_level, 5),
                    'person_price_level': round(person_price_level, 5)
                }
            except Exception as e:
                print('区域价格指数error: {}, reason: {}'.format(line, e))
        self.region_price_level_dict = region_price_level
        print('region_price_level_dict.size: {}'.format(len(self.region_price_level_dict.keys())))
        if len(self.region_price_level_dict.keys()) < 1000000:
            raise Exception('区域价格水平量级异常')

    def region_price_level_set(self, room):
        houseId = str(room['houseId'])
        room['regionPrice'] = self.region_price_level_dict.get(houseId, {}).get('area_price_level', None)
        if room['regionPrice'] is not None and room['regionPrice'] > 100:
            room['regionPriceRaw'] = room['regionPrice']
            room['regionPrice'] = 100
        room['regionPricePerson'] = self.region_price_level_dict.get(houseId, {}).get('person_price_level', None)
        if room['regionPricePerson'] is not None and room['regionPricePerson'] > 100:
            room['regionPricePersonRaw'] = room['regionPricePerson']
            room['regionPricePerson'] = 100
        if 'regonPricePerson' in room.keys():
            del room['regonPricePerson']

        if room['regionPrice'] is None:
            room['regionPrice'] = 0.0
        if room['regionPricePerson'] is None:
            room['regionPricePerson'] = 0.0
        return room

    def city_price_level(self):
        """
        城市价格指数
        """
        city_price_level = {}
        for line in open(setting['city_price_path'], 'r', encoding='utf-8').readlines():
            try:
                houseid, area_price_level, person_price_level = line.strip().split('\t')
                area_price_level = float(area_price_level)
                person_price_level = float(person_price_level)
                city_price_level[houseid] = {
                    'area_price_level': round(area_price_level, 5),
                    'person_price_level': round(person_price_level, 5)
                }
            except Exception as e:
                print('城市价格指数error: {}, reason: {}'.format(line, e))
        self.city_price_level_dict = city_price_level
        print('city_price_level_dict.size: {}'.format(len(self.city_price_level_dict.keys())))
        if len(self.city_price_level_dict.keys()) < 1000000:
            raise Exception('城市价格水平量级异常')

    def city_price_level_set(self, room):
        houseId = str(room['houseId'])
        if self.city_price_level_dict.get(houseId, None) is not None:
            obj = {
                "cityPerIndex": self.city_price_level_dict[houseId].get('person_price_level', None),
                "cityPerMeterIndex": self.city_price_level_dict[houseId].get('area_price_level', None)
            }
            if obj['cityPerIndex'] is not None and obj['cityPerIndex'] > 100:
                obj['cityPerIndexRaw'] = obj['cityPerIndex']
                obj['cityPerIndex'] = 100
            if obj['cityPerMeterIndex'] is not None and obj['cityPerMeterIndex'] > 100:
                obj['cityPerMeterIndexRaw'] = obj['cityPerMeterIndex']
                obj['cityPerMeterIndex'] = 100
            room['priceLevel'] = obj
        else:
            room['priceLevel'] = None
        return room

    def comment_tag(self):
        """
        评论标签
        """
        comment_tag_map = {}
        for line in open(setting['comment_tag_path'], 'r', encoding='utf-8').readlines():
            try:
                houseid, taginfo = line.strip().split('\t')
                taginfo = eval(taginfo)
                # 对其保留2位小数
                if taginfo is not None:
                    for k, v in taginfo.items():
                        taginfo[k] = round(v, 2)
                comment_tag_map[houseid] = taginfo
            except Exception as e:
                print('评论标签error: {}, reason: {}'.format(line, e))

        self.comment_tag_dict = comment_tag_map
        print('comment_tag_size: {}'.format(len(self.comment_tag_dict.keys())))
        if len(self.comment_tag_dict.keys()) < 150000:
            raise Exception('评论标签量级异常')

    def comment_tag_set(self, room):
        houseId = str(room['houseId'])
        room['comTag'] = self.comment_tag_dict.get(houseId, None)
        return room

    def md_online2date(self):
        """处理上线日期异常数据"""
        pass

    def md_online2date_set(self, room):
        if room['activeDate'] < '20111201':
            room['activeDate'] = '20111201'
            room['activeTime'] = '2011-12-01 00:00:00'
        return room

    # 加载house_show_price_threshold_new数据
    def house_show_price_threshold_new(self):
        self.house_show_price_threshold_new_dict = {}
        lines = open(setting['house_show_price_threshold_new_path'], 'r', encoding='utf-8').readlines()
        if len(lines) <= 10000:
            pass
            # raise Exception("房屋周围的价格倍率阈值数量<1W")

        for line in lines:
            items = line.strip().split("\t")
            houseId, adjustPriceRange = items[0], items[2]
            try:
                adjustPriceRange = float(adjustPriceRange)
                self.house_show_price_threshold_new_dict[houseId] = adjustPriceRange
            except Exception as identifier:
                print("houseId {} adjustPriceRange ERROR {}".format(houseId, items[2]))

    def house_show_price_threshold_new_set(self, room):
        houseId = str(room['houseId'])
        adjustPriceRange = self.house_show_price_threshold_new_dict.get(houseId, None)
        if adjustPriceRange is not None:
            room["adjustPriceRange"] = adjustPriceRange
        return room

    def to_file(self):
        """
        将房屋对象写入到文件中, 对象太大, 写不了 放弃治疗
        """
        fw = open(setting['save_path'], 'w', encoding='utf-8')
        for room in map(lambda e: json.dumps(e, ensure_ascii=False)):
            fw.writelines(room)
        fw.close()

    def run(self):
        # funcs = []
        """
        加载对象并得到处理对象的函数
        """
        for func in setting['run']:
            print('{} exe start load data'.format(func))
            eval('self.{}()'.format(func))
            print('{} exe end'.format(func))

        # 采取读文件的方式, 一下子加载全部对象内存吃不消
        fw = open(setting['save_path'], 'w', encoding='utf-8')
        for line in open(setting['origin_room_path'], 'r', encoding='utf-8'):
            house_id, room_obj_s = line.strip().split('\t')
            room_obj = json.loads(room_obj_s)
            """
            每个对象赋值操作
            """
            for func_name in setting['run']:
                func = eval('self.{}_set'.format(func_name))
                room_obj = func(room_obj)
            if room_obj.get("priceLevel", {}) != {}:
                fw.writelines('{}\t{}\n'.format(house_id, json.dumps(room_obj, ensure_ascii=False)))
        fw.close()


if __name__ == "__main__":
    base_info = BaseInfo()
    base_info.pull_data()
    base_info.run()