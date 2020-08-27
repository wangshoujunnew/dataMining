# Title: 数据预处理
from sklearn.preprocessing import MinMaxScaler

# 线性无量钢化
"""
center: scale => 
x - center 
----------
scala
数据移动了最小值个单位 
x* - x_min
---------
x_max - 

obj: 
scale_: 每一列的缩放值, 数据中的最大值-最小值 / 缩放目标最大值减去最小值
min_: 每列移动的中心
data_max_: 原始数据中每列的最大值
data_range: data_max - data_min

计算结果： X*scala_ + min_

公式
X(当前) - X(min)   X(新) - X(新min)
--------------- = ---------------
X(max) - X(min)   X(新max) - X(新min)

转换得到X(新)
"""


def self_minmax(data, range):
    """

    :param data: list
    :param range: tuple
    :return:
    """
    data_min = min(data)
    data_max = max(data)
    tmp = []
    for x in data:
        tmp.append((x - data_min) / (data_max - data_min) * (range[1] - range[0]) + range[0])
    return tmp


import numpy as np

# X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
line_scala = MinMaxScaler(
    feature_range=(0, 1)  #
)  # 对异常值非常铭感, 所以通常使用标准化
# MinMax 在不涉及到距离度量, 梯度, 协方差以及数据需要被压缩到特定区间的时候使用广泛
line_scala.fit(data)
print(np.array(data))
print("=========")
transform_result = line_scala.transform(data)
print(transform_result)
line_scala.inverse_transform(transform_result)  # 逆转缩放
line_scala

print(self_minmax(list(np.array(data)[:, 0]), (0, 1)))  # ok

# 当不想影响数据的稀疏性的时候, 只缩放不中心化 使用 MaxAbsScaler
# 当标准化的时候, 异常点较多, 使用 RobustScaler, 平移的是中位数, 缩放的是IQR(样本的四分位距) IQR = Q3 − Q1, Q2是中位数
from sklearn.preprocessing import RobustScaler

# Title: 缺失值处理impute.SimleImputer

# Title: 分类性特征处理, 编码与哑变量(One-hot), 变量类型: 名义变量One-hot, 有序变量OrdinalEncoder, 有距变量(连续性变量)
from sklearn.preprocessing import LabelEncoder  # 对y进行编码, 允许一维度数据
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

labelEncoder = LabelEncoder()
label = labelEncoder.fit_transform(['B', 'A'])
print(label)
print(labelEncoder.classes_)  # 索引对应的中文

OrdinalEncoder().categories_
# OneHotEncoder().get_feature_names()

# 连续性变量的处理
from sklearn.preprocessing import Binarizer, KBinsDiscretizer  # 二值化, 分箱

Binarizer(threshold=30).fit_transform([[1, 2]])
KBinsDiscretizer(n_bins=5, strategy="quantile")  # 每个特征分成5个箱子, uniform等宽, quantile等位, kmeans按照聚类分箱
