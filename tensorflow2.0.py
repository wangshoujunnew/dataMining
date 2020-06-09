# -*- coding: utf-8 -*-
# 时间: 2020/5/26 作者: 王寿军
# 用途: tensorflow 2.0 学习

print("hello")

import tensorflow as tf
import numpy as np

feature_column = tf.feature_column

print(tf.__version__)

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

features = {
    "embedding": []
}

feature_column.embedding_column()