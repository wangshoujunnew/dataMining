# 决策树
# 图的安装 graphviz
# 书籍: 数据挖掘导论
# 数据集: 1. 动物分类数据集(来自和琼社区) url: https://www.kesci.com/home/dataset/5e748a6498d4a8002d2b18a9/document
# 特征: 头发,羽毛,鸡蛋,牛奶,机载,水生,捕食者,齿,骨干,呼吸,有毒的,鳍,腿,尾巴,国内,catsize,类型
# 讨论问题: 1, 如何选择特征 2, 如何停止生长, 防止过拟合
# 参数文档 菜菜课堂文档
#%%
import sys
sys.path.append("C:/Users/shoujunw/PycharmProjects/dataMining/utils")
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from utilself import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

#%%
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

Xtrain, Xtest, Ytrain, Ytest = train_test_split(df, type_df, test_size=0.3, shuffle=False, random_state=0)
Xtrain, Xtest, Ytrain, Ytest

# print(Ytest.head(10))

decision_tree = DecisionTreeClassifier(
    criterion="gini",  # 标准(不纯度), 每个节点都有一个不纯度 $p(i|t) 标签分类i在节点t上的占比
    # 当决策树欠拟合的时候使用信息熵, 维度大噪音大的时候用gini
    random_state=0
)

decision_tree.fit(Xtrain, Ytrain)
Ypredict = decision_tree.predict(Xtest) # True为1, False为0
num = accuracy_score(Ytest, Ypredict, normalize=False)
score = accuracy_score(Ytest, Ypredict)

# 数据集split的时候采用了shuffle, 所以每次训练模型,模型学习到的不完全一样, 在测试集上的准去率也不一样
# 奇怪,将split改成了shuffle=False, 也指定了随机数种子, 为啥还是每次预测模型不一样, 难道模型的每次学习不是一样的?
# 解释: 除了分割数据随机性要确定外, 决策树内部还有一部分随机性: 当分叉效果一样的时候会随机选择特征, 所以还得设置决策树的随机种子
print("总数为: {}, 预测正确数量: {}, 预测准确率: {}".format(len(Ytest), num, score))
print("======查看数据图")
dot_data = export_graphviz(
    decision_tree=decision_tree,
    # 特征名字, 英文可以变中文
    feature_names=["头发", "羽毛", "鸡蛋", "牛奶", "机载", "水生", "捕食者", "齿", "骨干", "呼吸", "有毒的", "鳍", "腿", "尾巴", "国内", "catsize"],
    # 标签名称
    class_names=["两栖动物", "鸟", "鱼", "昆虫", "哺乳动物", "软体动物", "爬行动物"]


)
graph = graphviz.Source(dot_data, encoding="utf-8")
graph.format = "png"
# graph.view("tree") 直接在notebook里面输出有事可以中文的, 奇怪
