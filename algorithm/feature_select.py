# 特征选择, 1: 特征提取(文字图像等特征数字化), 2: 特征创造 3: 特征选择
# 1. 过滤法: 方差过滤
from sklearn.feature_selection import VarianceThreshold
# 相关性过滤, K方值过滤
# 卡方检验, 推测两组数据的差异, 主要检测离散型数据
# 卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，如果卡方值越大，二者偏差程度越大；反之，二者偏差越小；若两个值完全相等时，卡方值就为0，表明理论值完全符合。
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif as MIC
# F检验, 捕捉特征和y之间的线性关系
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectFromModel  # 嵌入法选择特征, 使用一个模型在这个数据集上的特征重要性选择特征

# 协方差, 皮尔逊相关系数,卡方检验, f检验 的区别, 研究一下p>=0.05的关系 ?? todo

# 互信息法, 检测标签和特征的 线性,非线性关系(可以找出任意关系) F检验只能找到线性关系 MIC

varianceThreshold = VarianceThreshold()
varianceThreshold.variances_  # 先找出方差的中位数, 然后把这个中位数当做阈值参数输入

# 通过学习曲线来选择k的大小
SelectKBest(chi2, k=300).fit_transform([[1, 2]], [1])  # 选取前k个和y先关的特征, 相关量采用k方

from sklearn.ensemble import RandomForestClassifier
import numpy as np

RFC = RandomForestClassifier(n_estimators=10)
RFC.fit([], [])

X_embedding = SelectFromModel(RandomForestClassifier(n_estimators=10), threshold=np.linspace(0, max(RFC.feature_importances_), 20)).fit_transform([], [])

# 将x_embedding在新模型上进行交叉验证
