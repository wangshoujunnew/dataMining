# 特征选择, 1: 特征提取(文字图像等特征数字化), 2: 特征创造 3: 特征选择
# 1. 过滤法: 方差过滤
from sklearn.feature_selection import VarianceThreshold

varianceThreshold = VarianceThreshold()
varianceThreshold.variances_  # 先找出方差的中位数, 然后把这个中位数当做阈值参数输入
