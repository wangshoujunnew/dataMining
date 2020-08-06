# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 不同产品的销售情况

import pandas as pd 
from beakerx import *
import pandasql
pysqldf = lambda sql: pandasql.sqldf(sql, globals())

# +
df_csv = pd.read_csv(r"C:\Users\shoujunw\Downloads\order1.csv") # df["mi"] = df["mi"].dt.strftime('%Y-%m-%d')

df:DataFrame = pysqldf("""
select julianday(Date(ma)) - julianday(Date(mi)) as dua
    , * 
    from df_csv
    -- where ma -mi > 7 -- 选取持续时间>1周的产品

""")
# 给sum_amount, avg_amount_night 添加rank
# df["g"] = "g"
df["sum_amount_rk"] = df["sum_amount"].rank(ascending=False).astype(int)
df["avg_amount_night_rk"] = df["avg_amount_night"].rank(ascending=False).astype(int)
df.to_csv("tmp.csv")
# TableDisplay(df)
# -

tmp_df = pd.read_csv("tmp.csv", index_col=0)
TableDisplay(tmp_df)

# # 如何权衡 量多 间夜gmv小   和 量少间夜gmv多的 比较
# 1. 产品: 别栋别墅: 每间夜8542价格最高, 所贡献的总体gmv: 排名275名: 17084
#         新房特惠: 每间夜价格小308, 订单量多贡献总体gmv: 5亿, 

newdf = pd.DataFrame([[0.2], [0.1], [0.3]], columns=["a"])
newdf["g"] = "g"
newdf

newdf.groupby("g")["a"].rank().astype(int)

# # 活动持续时间

# +
# 0-6天
