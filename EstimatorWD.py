# -*- coding: utf-8 -*-
# 时间: 2020/5/18 作者: 王寿军
# 用途: wide&deep estimator

import tensorflow as tf

tf.estimator.LinearClassifier
tf.estimator.DNNLinearCombinedClassifier


def _model_fn(features, labels, mode, params):
    """
    features: dict: feature_name -> feature_value
    特征的描述被放入到param中, param中的数据来自yml配置文件
    param: 中还需要区分哪些是wide部分输入的特征, 哪些是deep部分输入的特征
    """
    feature_col = []
    wide_features = []

    # 连续特征
    for key in params["numeric_column"]:
        feature_col.append(tf.feature_column.numeric_column(features[key]))

    # 分桶
    bucket_feature_dict = {}  # 为了使用交叉特征
    # feature_name:
    #   is_wide: True
    #   is_deep: True
    #   boundaries: 20000
    for key, value_dict in params["buckets"].items():
        # key 为特征名称, value为桶的个数
        tmp = tf.feature_column.bucketized_column(
            features[key],
            boundaries=value_dict["boundaries"]
        )
        bucket_feature_dict[key] = tmp
        if value_dict["wide"]:
            pass
        # if type(boundaries) == list:  # 表示需要直接用于特征学习
        #     feature_col.append(tmp)
        # else:
        #     pass  # 表示值用于特征交叉等操作
    # 交叉
    for f, size in params["cross"].items():
        fs = f.split("&")
        # 所有分桶特征的交叉总数为 all_size

        all_size = sum([len(params["buckets"][fl]) + 1 for fl in fs])
        tmp = tf.feature_column.crossed_column(
            [bucket_feature_dict[fl] for fl in fs],
            size=size if type(size) == int else all_size
        )
        tmp = tf.feature_column.indicator_column(
            tmp
        )
        feature_col.append(tmp)

    # embed
    for key, dim in params["embed"].items():
        tmp = tf.feature_column.embedding_column(
            features[key],
            dim=dim
        )
        feature_col.append(tmp)
