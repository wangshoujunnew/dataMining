# 集成算法
from datasetself import *
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

rng = np.random.RandomState(0)

from sklearn.impute import SimpleImputer  # 缺失值的填充

# 基础分类器从1到500的score趋势
scores = []
tr_scores = []
max_clf = None
for i in range(1, 50):
    rfc = RandomForestClassifier(
        n_estimators=i
        , random_state=0
        , bootstrap=True  # 采用自助法抽样
    )
    rfc.fit(Xtrain, Ytrain)
    score = rfc.score(Xtest, Ytest)
    tr_scores.append(rfc.score(Xtrain, Ytrain))
    scores.append(score)
    if score >= max(scores):
        max_clf = rfc

plt.plot(range(1, 50), scores, color='blue', label="test")
plt.plot(range(1, 50), tr_scores, color='red', label="train")
plt.legend()
plt.xticks(range(1, 50))
plt.show()

print(max_clf.score(Xtest, Ytest))
print(cross_val_score(max_clf, df, type_df).mean())

# from scipy.special import comb # 排列组合C comb(25,1)
# 集成算法 需要保证每个基分类器的准确率 > 50%, 否则, 集成算法将比基分类器的效果更差


# 波士顿数据集的一个缺失情况
data = load_boston()

rng.choice(10, 4, replace=False)  # 在0到10中随机选择4个数据,不重复的数据, 赋值为空 = np.nan
simpleImputer = SimpleImputer(
    missing_values=np.nan  # 缺失值长啥样 0也可以表示缺失值
    , strategy="mean"  # 填补策略, 均值填充, 中位数, 众数, constant常熟填充 , fill_value=0
)

# 使用随机森林填充缺失值(回归填补缺失值思想: 属性可以预测标签值, 标签纸值也可以预测属性值)
# 特征T有缺失值, !T + label 作为特征值, 预测T的值(T有值的部分就是训练集, T没有的值就是测试集: 从缺失值最少的开始填充, 填补特征T的时候, 其他特征缺失值先以0代替)
# np.argsort() 返回的是从小到大对应的索引而不是数值, np.sort返回之后没有索引信息

# 决策树参数的重要程度排序
# 树的个数, 最大深度, 叶子节点的最小样本数, 节点分割的最小样本数, 最大特征数, 不纯度划分标准

# 参数调整: 使用学习曲线对树的个数进行每10步骤一次的调整, 然后在某个范围内每1步的调整, 确定好树的个数的这个参数, 然后其他参数使用网格搜索
