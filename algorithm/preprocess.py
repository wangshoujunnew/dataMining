# 数据预处理
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
)
line_scala.fit(data)
print(np.array(data))
print("=========")
print(line_scala.transform(data))
line_scala

print(self_minmax(list(np.array(data)[:, 0]), (0, 1))) # ok