# 模型的评价指标

# presion, recall, F1-score, AUC, ROC, PR

# TP 后面的是预测值， 真阳性， 假阳性
# 预测病人： 希望， 检测出TP， 找到真正患病的人: 真阳性的概率和假隐性的概率


#  *************准确率*************
# 1，accuracy_score

# 准确率
import numpy as np
from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3, 9, 9, 8, 5, 8, 1]

# 多类分类（一个任务假设只能分给一个类别）， 多类标签（一个任务可以被分类给多个类别）， 多类输出（每个样本分配一组目标值，可以认为是预测样本的多个属性，比如具体地点的
# 风速和大小 区别 todo ？？

accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

y_true = [0, 1, 2, 3, 2, 6, 3, 5, 9, 1]

accuracy_score(y_true, y_pred)
# Out[127]: 0.33333333333333331

accuracy_score(y_true, y_pred, normalize=False)  # 类似海明距离，每个类别求准确后，再求微平均
# Out[128]: 3

# 2, metrics
from sklearn import metrics

# 精度如何针对多分类 todo ??
metrics.precision_score(y_true, y_pred, average='micro')  # 微平均，精确率
# Out[130]: 0.33333333333333331

metrics.precision_score(y_true, y_pred, average='macro')  # 宏平均，精确率
# Out[131]: 0.375

metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')  # 指定特定分类标签的精确率
# Out[133]: 0.5

#  *************召回率*************
metrics.recall_score(y_true, y_pred, average='micro')
# Out[134]: 0.33333333333333331

metrics.recall_score(y_true, y_pred, average='macro')
# Out[135]: 0.3125

#  *************F1*************
metrics.f1_score(y_true, y_pred, average='weighted')
# Out[136]: 0.37037037037037035

#  *************F2*************
# 根据公式计算
from sklearn.metrics import precision_score, recall_score


def calc_f2(label, predict):
    p = precision_score(label, predict)
    r = recall_score(label, predict)
    f2_score = 5 * p * r / (4 * p + r)
    return f2_score


#  *************混淆矩阵*************
from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, y_pred)

# Out[137]:
# array([[1, 0, 0, ..., 0, 0, 0],
#        [0, 0, 1, ..., 0, 0, 0],
#        [0, 1, 0, ..., 0, 0, 1],
#        ...,
#        [0, 0, 0, ..., 0, 0, 1],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 1, 0]])

#  *************ROC*************
# 1，计算ROC值
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)

# 2，ROC曲线
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

#  *************海明距离*************
from sklearn.metrics import hamming_loss

y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
hamming_loss(y_true, y_pred)
0.25

#  *************Jaccard距离*************
import numpy as np
from sklearn.metrics import jaccard_similarity_score

y_pred = [0, 2, 1, 3, 4]
y_true = [0, 1, 2, 3, 4]
jaccard_similarity_score(y_true, y_pred)
0.5
jaccard_similarity_score(y_true, y_pred, normalize=False)
2

#  *************可释方差值（Explained variance score）************
from sklearn.metrics import explained_variance_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
explained_variance_score(y_true, y_pred)

#  *************平均绝对误差（Mean absolute error）*************
from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)

#  *************均方误差（Mean squared error）*************
from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)

#  *************中值绝对误差（Median absolute error）*************
from sklearn.metrics import median_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
median_absolute_error(y_true, y_pred)

#  *************R方值，确定系数*************
from sklearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_pred)

# -------------------- k折交叉验证 10次10折交叉验证的均值RepeatedKFold
# 每次形成数据集为每一次repreat 就会得到(k-1, 1)的一个一个k次集合及数组的长度为 x = repeat * k*(k-1)
from sklearn.model_selection import RepeatedKFold
import sklearn.datasets as datasets
import time
from sklearn.model_selection import KFold

iris = datasets.load_iris()

iris
time_int = int(time.time())
kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=time_int)
iris_data = iris["data"]
datas = kfold.split(iris["data"][:100], iris["target"][:100])
datas = list(datas)
datas

kfold_1 = KFold(n_splits=10)
for x, y in kfold_1.split(iris["data"][:100], iris["target"][:100]):
    print("===========")  # 1折留作训练集 -> (9, 1)

# 采样
from sklearn.utils import resample

# 随机数
import numpy as np

rng = np.random.RandomState(0)  # 0=seed
a = rng.uniform(low=0, high=1)  # 均匀分布
b = rng.binomial(10, 0.5)  # 二项式分布: 6, 表示有6次得到1
c = rng.multinomial(10, pvals=[0.1, 0.2, 0.7])  # 多项式分布, 做10次试验, 分布出现1,2,3类别的情况 c = [1,2,7] 出现1:1次, 出现2:2次, 出现3:7次
d = rng.beta(1, 2)
# e = rng.dirichlet()

rng.uniform(size=10)
rng.normal(size=10)
rng.binomial(n=20, p=0.1, size=1)

sum(np.random.binomial(9, 0.1, 20000) > 0)/20000
# np.random.binomial(9, 0.1, 20000)

rng.multinomial(10, [0.1, 0.2, 0.7])

rng.beta(1, 5, 10)
rng.dirichlet((10, 5, 3), 20)
rng.poisson()


# rank评估指标
