import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.logging.set_verbosity(tf.logging.INFO)


def my_init_fn():
    # 一个单独的输入
    x = np.arange(100, dtype=np.float32) / 100.0
    np.random.seed(0)

    # 一个单独的输出
    y = x * 0.8 - 0.2 + np.random.normal(0, 0.1, size=[100])

    # 获取 tf.data.Dataset 对象
    d = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)

    return d.make_one_shot_iterator().get_next()


def my_init_fn2():
    x = np.arange(100, dtype=np.float32)
    d = tf.data.Dataset.from_tensor_slices(x).batch(4)
    return d.make_one_shot_iterator().get_next()


def my_model_fn(features, labels, mode, params):
    # 直接使用 features & labels

    # 构建模型
    logits = tf.layers.dense(tf.reshape(features, [-1, 1]), 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 获取损失函数
    if labels is not None:
        labels = tf.to_float(tf.reshape(labels, [-1, 1]))
    loss = tf.losses.mean_squared_error(labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        # 定义性能指标
        mean_absolute_error = tf.metrics.mean_absolute_error(labels, logits)
        metrics = {'mean_absolute_error': mean_absolute_error}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # 构建优化器与梯度更新操作
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # 创建Estimator对象
    model = tf.estimator.Estimator(model_fn=my_model_fn, model_dir='./estimatorModel')


    def train():
        # 训练
        # NUM_EPOCHS = 20
        # for i in range(NUM_EPOCHS):
        #     model.train(my_init_fn)

        # 预测
        # 每行预测结果类似 {'logits': array([0.07357793], dtype=float32)}
        # 这里使用 my_init_fn2 也有一样的结果
        predictions = model.predict(input_fn=my_init_fn)
        for pred in predictions:
            print(pred)

        # 评估
        # 评估结果类似 {'loss': 0.03677843, 'mean_absolute_error': 0.15645184, 'global_step': 500}
        print(model.evaluate(my_init_fn))


    def look_input_fn():
        with tf.Session() as sess:
            print(sess.run(my_init_fn()))
            print(sess.run(my_init_fn()))


    look_input_fn()


    def online():
        """
        模型上线, x = tf.placeholder(tf.float32, shape=[None,2])
                dataset = tf.data.Dataset.from_tensor_slices(x)
                这种方式无法使用, make_one_shot_iterator的时候不认placeholder
        :return:
        """

        class InputFn:
            """
            制作数据迭代器
            """
            value = [1.0, 3.0]  # 设置一个批次, 可以预测多个

            @staticmethod
            def getValue():
                while True:  # 永远没有止境的迭代器, yield更具input_fn适配格式
                    for v in InputFn.value:
                        # yield (np.array(InputFn.value), np.array(InputFn.value))
                        yield (v, v)  # 依次返回一个, 如果要预测多个需要再dataset中设置batch(n),

        dataset = tf.data.Dataset.from_generator(
            output_types=(tf.float32, tf.float32)  # x的类型和y的类型
            , generator=InputFn.getValue
        )
        next = dataset.batch(2).make_one_shot_iterator().get_next()  # 如果要预测多个在这里设置batch(n)

        # input_fn的内容也是个张亮
        # (<tf.Tensor 'IteratorGetNext_2:0' shape=(?,) dtype=float32>, <tf.Tensor 'IteratorGetNext_2:1' shape=(?,) dtype=float64>)
        input_fn_generate = lambda: next

        def input_fn_generate_new():

            next = tf.data.Dataset.from_tensor_slices([3.0, 4.0, 5.0, 6.0]).batch(1).make_one_shot_iterator().get_next()
            return (next, next)


        # # 得到的已经是个迭代器了, 不是一个tensor了,不用sesion.run了
        # InputFn.value = [3.0, 4.0, 5.0, 6.0]
        predictions = model.predict(input_fn=input_fn_generate)  # 不需要session了

        # x = input_fn_generate()
        # x
        with tf.Session() as sess:
            # print("===========\n", sess.run(input_fn_generate()))
            # print(sess.run(input_fn_generate_new()))
            print("1")

            print(predictions.__next__())
            print(predictions.__next__())
            print(predictions.__next__())
            print(predictions.__next__())


        # input_fn始终是一个batch_size的输入

        # print(next(predictions))
        # print(predictions)
        # # print(next(predictions))
        # for i in predictions:
        #     print(i)

        # # with tf.Session() as sess:
        #
        # # print("estimator预测: ", sess.run(predictions))
        # InputFn.value = [3.0, 4.0, 5.0, 6.0]
        # print(next(predictions))


    online()
    # dataset = my_init_fn()
    # with tf.Session() as sess:
    #     print(sess.run(dataset))
