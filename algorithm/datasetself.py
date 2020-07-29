# 数据集的获取, 动物数据集
import sys

sys.path.append("C:/Users/shoujunw/PycharmProjects/dataMining/utils")
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_score
from utilself import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import numpy as np
import matplotlib.pyplot as plt

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
