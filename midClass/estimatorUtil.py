"""
测试estimator
"""
import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    """
    最简单的模型函数
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    # 一系列操作之后得到logit
    logits = tf.layers.dense(tf.reshape(features, [-1, 1]), 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}

        output = {'predict': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions
                                          # 为了模型的导出, export_outputs, 使得模型可以使用placeholder
                                          , export_outputs=output
                                          , loss=None
                                          , train_op=None)

    # 获取损失函数
    labels = tf.reshape(labels, [-1, 1])
    loss = tf.losses.mean_squared_error(labels, logits)
    # 构建优化器与梯度更新操作
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def input_fn():
    # 一个单独的输入
    x = np.arange(100, dtype=np.float32) / 100.0
    np.random.seed(0)
    # 一个单独的输出
    y = x * 0.8 - 0.2 + np.random.normal(0, 0.1, size=[100])
    # 获取 tf.data.Dataset 对象, 返回一个二元组, 元祖0为特征, 元祖1为label
    d = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)

    return d.make_one_shot_iterator().get_next()


def model_train(model_save_dir):
    """
    训练模型
    :param model_save_dir: 模型保存目录
    :return:
    """
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_save_dir)
    model.train(input_fn=input_fn)


def model_train_server(model_save_dir):
    """
    训练模型, 导出成savemodel的方式, 并提供placeholder接口
    :param model_save_dir:
    :return:
    """

    # 模型的导出, 提供placeholder接口
    def serving_input_receiver_fn():
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input_tensor')

        # features 传入的数据字典被变化之后的结果的是tfrecord张亮,需要解析出来, receiver_tensors 指代我们要传入的数据字典
        return tf.estimator.export.ServingInputReceiver(features=x_placeholder, receiver_tensors=x_placeholder)

    model = tf.estimator.Estimator(model_fn=model_fn)
    model.train(input_fn=input_fn)
    print("执行模型的导出, 导出格式savemodel")
    model.export_savedmodel(model_save_dir, serving_input_receiver_fn
                            , strip_default_attrs=True)
