# 决策树
# 图的安装 graphviz
# 书籍: 数据挖掘导论
# 数据集: 1. 动物分类数据集(来自和琼社区) url: https://www.kesci.com/home/dataset/5e748a6498d4a8002d2b18a9/document
# 特征: 头发,羽毛,鸡蛋,牛奶,机载,水生,捕食者,齿,骨干,呼吸,有毒的,鳍,腿,尾巴,国内,catsize,类型
# 讨论问题: 1, 如何选择特征 2, 如何停止生长, 防止过拟合
# 参数文档 菜菜课堂文档
# %%
import sys

sys.path.append("C:/Users/shoujunw/PycharmProjects/dataMining/utils")
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from utilself import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# %%
# ----------------- 数据集的获取
df = zip_read(zipf="C:\\Users\\shoujunw\\PycharmProjects\\dataMining\\algorithm\\zoo5998.zip", csvfs=["Zoo.csv"])


def data_summary(df: DataFrame):
    """
    数据情况查看
    :return:
    """
    print("========目标")
    print("==========shape", df.shape)
    print("==========各个feature name 对应的信息")
    print(df.info())
    print(df.describe(include='all'))


df = next(df)
# print(df.head(10))

data_summary(df)

type_df = df.pop("type")

# 把数据集的True和False编程0-1

# 如何保证split的数据独立同分布 todo ??
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df, type_df, test_size=0.3, shuffle=False, random_state=0)
Xtrain, Xtest, Ytrain, Ytest

# print(Ytest.head(10))
feature_zh_names = ["头发", "羽毛", "鸡蛋", "牛奶", "机载", "水生", "捕食者", "齿", "骨干", "呼吸", "有毒的", "鳍", "腿", "尾巴", "国内", "catsize"]

# 数据集很巨大的时候,一定要剪枝, 否则,树会一直生长,内存都撑爆
decision_tree = DecisionTreeClassifier(
    criterion="gini"  # 标准(不纯度), 每个节点都有一个不纯度 $p(i|t) 标签分类i在节点t上的占比
    # 当决策树欠拟合的时候使用信息熵, 维度大噪音大的时候用gini
    , random_state=0
    , splitter="best"  # random, 随机选择特征进行分裂(更能减小过拟合), best在重要的特征上进行随机的选择

    # 不加任何的树的剪枝结果为
    # 总数为: 31, 预测正确数量: 25, 预测准确率: 0.8064516129032258
    # 训练集总数为: 70, 预测正确数量: 70, 预测准确率: 1.0
    # ============ 加了这几个参数之后, 虽然测试集上的准确率降低了, 但是训练集上的准确率,没有改变, 提高了树的泛化能力
    , max_depth=4
    , min_samples_leaf=5  # 分裂之后, 叶子节点最少要有5个才能分裂
    , min_samples_split=10  # 当前节点最少要有10个节点才能分裂
    # 加完之后
    # 总数为: 31, 预测正确数量: 23, 预测准确率: 0.7419354838709677
    # 训练集总数为: 70, 预测正确数量: 66, 预测准确率: 0.9428571428571428

    , max_features=7  # 决策树使用的最大特征, 这个一般需要借助pca, 不要轻易尝试, float, sqrt, log2
    , min_impurity_decrease=0.001  # 信息增益必须>0.01

    # 目标权重参数, 设置正样本权重高一些, 完成样本标签平衡的参数, 公式是啥:todo ??
    , class_weight=dict(zip(['bird', 'fish', 'insect', 'mammal', 'mollusc.et.al', 'reptile'], [1, 1, 1, 1, 1, 1, 1]))  # balanced 会自动计算样本平衡
    # , min_weight_fraction_leaf 叶子节点最小样本权重,如果小于这个, 则会连同兄弟节点一起被剪枝

)

decision_tree.fit(Xtrain, Ytrain)
Ypredict = decision_tree.predict(Xtest)  # True为1, False为0
Ypredict_train = decision_tree.predict(Xtrain)
num = accuracy_score(Ytest, Ypredict, normalize=False)
train_num = accuracy_score(Ytrain, Ypredict_train, normalize=False)

score = accuracy_score(Ytest, Ypredict)
train_score = accuracy_score(Ytrain, Ypredict_train)

# 数据集split的时候采用了shuffle, 所以每次训练模型,模型学习到的不完全一样, 在测试集上的准去率也不一样
# 奇怪,将split改成了shuffle=False, 也指定了随机数种子, 为啥还是每次预测模型不一样, 难道模型的每次学习不是一样的?
# 解释: 除了分割数据随机性要确定外, 决策树内部还有一部分随机性: 当分叉效果一样的时候会随机选择特征, 所以还得设置决策树的随机种子
print("总数为: {}, 预测正确数量: {}, 预测准确率: {}".format(len(Ytest), num, score))
print("训练集总数为: {}, 预测正确数量: {}, 预测准确率: {}".format(len(Ytrain), train_num, train_score))

print("======查看数据图")
print("========特征的重要性")
feature_importance = list(zip(feature_zh_names, decision_tree.feature_importances_))
feature_importance.sort(key=lambda x: x[1] * -1)
print(feature_importance)
# ('牛奶', 0.4843500312651983), ('羽毛', 0.23972880335348196), ('鳍', 0.16645244215938307), ('呼吸', 0.10946872322193654)
# 可以看到有很多特征的重要性为0, 说明决策树,不是选取了所有的特征, 而是随机选取了特征, 树的深度等超赞数也会限制特征的选择, 有些特征还没有来得及选择
dot_data = export_graphviz(
    decision_tree=decision_tree
    # 特征名字, 英文可以变中文
    , feature_names=feature_zh_names
    # 标签名称
    , class_names=["两栖动物", "鸟", "鱼", "昆虫", "哺乳动物", "软体动物", "爬行动物"]
    , filled=True  # 画图的时候不同类别使用不同的颜色填充, 颜色也深,代表纯度越高
    , rounded=True  # 画图的时候呈现圆角
)
graph = graphviz.Source(dot_data, encoding="utf-8")
# graph.format = "png"
# graph.view("tree") 直接在notebook里面输出有事可以中文的, 奇怪
