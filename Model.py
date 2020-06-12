# 自己测试的一些model
from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np
from pandas import DataFrame
from pyecharts import Line
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

plt.figure(figsize=(8, 6), dpi=80)  # 生成画布
plt.ion()  # 打开交互模式 .clf 清除figure对象, cla 清除axes对象, pause 暂停功能

# top3曝光情况
# 值, 速度, 加速度
class SVMModel:
    feature = {x.strip().split()[0]: x.strip().split()[1] for x in open("v2_feature.txt", "r", encoding="utf-8").readlines()}

    @staticmethod
    def load_libsvm(f, n_feature): # 加载libsvm文件, skitlean自带的转为稠密向量的时候会默认填充空位0, 不可取
        # df = [[] for x in range(n_feature)]
        df = []
        for i, line in enumerate(open(f, "r", encoding="utf-8").readlines()):
            # 去除#后面的内容
            line = line.split("#")[0]
            label, *feature = line.strip().split("\t")
            label = float(label)
            sample = [None] * n_feature
            for f in feature:
                index, value = f.split(":")
                index, value = int(index), float(value)
                if index > n_feature:
                    continue
                sample[index - 1] = value
            # 添加label
            sample.append(label)
            df.append(sample)
        df = DataFrame(np.array(df), columns=[str(x + 1) for x in range(n_feature)] + ["target"], dtype=float)
        return df

    @staticmethod  # 动态图的回调
    def plt_dy_call(fn, data):
        # 阻塞模式, show, 图片关闭前阻塞
        # 交互模式, .plot 后立马显示图片
        plt.clf()  # 必须前清空之前的绘图
        fn(data)
        plt.pause(0.5)

    # 准确率计算
    @staticmethod
    def acc(label: list, predict: list):
        assert len(label) == len(predict), "label predict length iss not equals"
        label, predict = map(lambda x: np.array(x).reshape([-1, 1]), [label, predict])  # 为了打印预测错误的数据
        tuple_data = np.concatenate([label, predict], axis=1)

        # 参数需要为list类型
        is_equals = np.equal(list(tuple_data[:, 0]), list(tuple_data[:, 1]))
        result = sum(is_equals) / len(is_equals)
        # 打印预测错误的数据, 如何对一个一个列表取反
        error_data = tuple_data[[not x for x in is_equals]]
        print(error_data)
        print("data size %d: acc %.4f" % (len(label), result))
        return result

    @staticmethod  # 组装predict和label, 通过predict的计算值对tuple排序
    def predict_sort(label: list, predict: list):
        tuple_items = zip(label, predict)
        tuple_items = sorted(tuple_items, key=lambda x: -1 * x[1])
        return tuple_items

    # 排序map指标, 在排序算法中是无法引入只关注topk的排序的, 因此将此元素位置,引入到损失函数中
    @staticmethod
    def rank_map(label: list, predict: list, k):
        # 先计算topk 的 ap label 要么是1 要么是0
        # 将label和predict绑定在一起, 通过predict排序
        tuple_items = SVMModel.predict_sort(label, predict)
        # 然后通过label来计算ap
        ap = []  # 平均准确率
        for i in range(k):
            li_ap = sum(map(lambda x: x[0], tuple_items[:i + 1])) / (i + 1)
            ap.append(round(li_ap, 4))

        print(f"data: {tuple_items} \ntok {k} ap: {ap}, map: {sum(ap) / len(ap) : .4f}")

    # 排序ndcg指标, 可以度量更多分级, 不止是0/1,
    # 因为把第一个位置排好了特别重要, 所以ndcg可能大于map也可能小于map, 取决于前几个排的咋样
    @staticmethod
    def rank_ndcg(label: list, predict: list, k):
        def score(x):
            # 得分计算, 我们使用对目标label进行去指数, 相关性分数
            return math.exp(x)

        tuple_items = SVMModel.predict_sort(label, predict)

        def dcg_get(tuple_items):
            dcg = []
            for i in range(k):
                item = tuple_items[i][0]
                dcg.append(round(score(item) / math.log(i + 2, 2), 4))
            return dcg

        # 计算带折扣的累计增益 dcg # 折扣是引入位置因素
        dcg = dcg_get(tuple_items)
        dcg_sum = sum(dcg)

        # 不同用户的dcg相比没有意义, 因此需要归一化
        # 计算用户的idcg, 通过label排序
        tuple_items.sort(key=lambda x: -1 * x[0])
        idcg = dcg_get(tuple_items)
        idcg_sum = sum(idcg)

        print(f"dcg: {dcg}, dcg_sum: {dcg_sum}, idcg_sum: {idcg_sum} , ndcg: {dcg_sum / idcg_sum: .4f}")

    # 制作pair对
    @staticmethod
    def make_pair(label: list):
        pos = list(filter(lambda x: x[1] == 1, enumerate(label)))  # 正样本索引
        neg = list(filter(lambda x: x[1] == 0, enumerate(label)))
        pairs = []
        for p in pos:
            for n in neg:
                pairs.append((p[0], n[0]))
        return pairs

    @staticmethod
    def show_image(data):
        # 展示原始数据, 看来这个不行了, 这个无法做动态图
        line = Line(title="模型数据图")  # 折线图
        line.add(name="原始数据", x_axis=list(data[:, 0]), y_axis=list(data[:, 1]))
        line.render("data.html")
        line

    @staticmethod  # 动态图
    def show_image_dy(data):
        # plt.plot()
        plt.scatter(data[:, 0], data[:, 1])

    @DeprecationWarning
    @staticmethod # 特征的相关系数绘制, 数据缺失的情况下无法计算相关性, 需要填充或者过滤掉缺失的值
    def feature_pearson_image(df: DataFrame, index_dict=None): # 皮尔逊相关系数 x,y协方差 / (x的标准差*y的标准差), 协方差=mean( (x-x_mean) * (y-y_mean) )
        """
        index_dict: 特征索引含义
        """
        def is_null(x): return x == np.NaN or x == None
        infos = []
        target = df.columns[-1]
        for col in df.columns[: -1]:
            is_null_df = df[df[col].map(lambda x: is_null(x))]
            is_not_null_df = df[df[col].map(lambda x: not is_null(x))]
            pearson_df = is_not_null_df[[col, target]]
            pearson_df[[col, target]] = pearson_df[[col, target]].astype(float)
            # print(pearson_df.dtypes)
            pearson = pearson_df.corr()
            # print(pearson)
            if index_dict != None:
                col_name = index_dict.get(col, None)
                if col_name:
                    infos.append((f"col:{col_name}:{col}, 空值率: {len(is_null_df) / len(df[target]): .4f} 相关系数: {pearson.iloc[0, 1]: .4f}", pearson.iloc[0, 1]))
        infos.sort(key=lambda x: math.fabs(x[1]) * -1)
        for info in infos:
            print(info)


    @staticmethod
    def load_data(data_path): # 加载数据
        feature_df = SVMModel.load_libsvm(data_path, 200)
        print(feature_df.head(10))
        print(feature_df.dtypes)
        # SVMModel.feature_pearson_image(feature_df.head(10), SVMModel.feature)
        return feature_df


    @staticmethod
    def select_from_cart(df): # 根据cart数来选择重要的特征
        pass

    @staticmethod
    def _filter_feature(df):
        # df的特征和所需要的的特征做交集
        # 如果当前的特征没有任何的区分性, 则得到的方差为0, 则和其他的计算出来的协方差为0
        jiao_set = set(SVMModel.feature.keys()).intersection(set(df.columns))
        df = df[list(jiao_set) + ["target"]]
        return df

class CorrLook:
    """
    查看数据集的特征的相关性
    """
    pairs = []
    index_featurename = SVMModel.feature

    @staticmethod
    def _get_feature_xiangguan_pair(df: DataFrame):
        """
        为了计算特征和特征的相关性, 先得到所有特征组合的pair
        """
        result = []
        cols = df.columns
        # 下标组队
        for i_1, col in enumerate(cols):
            for i_2, c in enumerate(cols):
                if i_1 < i_2:
                    result.append((col, c))

        return result

    @staticmethod
    def show(df): # 此corr_df计算的时候只取了非空的
        print("数据分布情况, 保存到csv中")

        curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        describe = df.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).applymap(lambda x: round(x, 4))
        describe.columns = [SVMModel.feature.get(x, x) for x in describe.columns]
        describe.to_excel("describe{}.xlsx".format(curtime))


        pairs = CorrLook._get_feature_xiangguan_pair(feature_df)
        CorrLook.pairs = pairs

        corr_df = df.corr()

        result = []
        for pair in CorrLook.pairs:
            # , 前面为行, 行可以多个, 如果没有, 则默认是所有的列, loc 不是() , 而是 [] df.loc["0"]["0"] 第0行第0列, iloc 通过数字来取
            result.append((pair, corr_df.loc[pair[0]][pair[1]]))
        result = list(filter(lambda x: not pd.isna(x[1]), sorted(result, key=lambda x: x[1] * -1)))
        for x in result:
            print("%s#%s#%s" % (CorrLook.index_featurename.get(x[0][0], x[0][0]),
                                  CorrLook.index_featurename.get(x[0][1], x[0][1]), x[1]))

        return result

feature_df = SVMModel.load_libsvm("d:/data/houseData/tujia_20191212.land.valid.data", 200)
feature_df = SVMModel._filter_feature(feature_df)

CorrLook.show(feature_df)