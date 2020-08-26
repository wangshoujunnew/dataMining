from unittest import TestCase
import tensorflow as tf
import subprocess
from estimatorUtil import *


class TensorflowTest(TestCase):
    def setUp(self):
        self.sess = tf.Session()
        print("init ...")

    def tearDown(self):
        self.sess.close()
        print("clern down ... ")

    def testEstimatorModelServer(self):
        print("测试estimator导出服务模型, 制作placeholder")
        save_model_dir = "d:/saveModel/estimator"
        # subprocess.run("del /F /S /Q {}".format(save_model_dir), shell=False)
        model_train(save_model_dir)
        """ 直接训练模型保存的文件格式
        -rw-r--r-- 1 shoujunw 1049089 169913  8月  6 22:19 events.out.tfevents.1596723540.DPSHOUJUNW
        -rw-r--r-- 1 shoujunw 1049089 130854  8月  6 22:19 graph.pbtxt
        -rw-r--r-- 1 shoujunw 1049089     16  8月  6 22:19 model.ckpt-25.data-00000-of-00001
        -rw-r--r-- 1 shoujunw 1049089    187  8月  6 22:19 model.ckpt-25.index
        -rw-r--r-- 1 shoujunw 1049089  55449  8月  6 22:19 model.ckpt-25.meta
        """

    def testEstimatorModelExport(self):
        print("查看模型的结果, 是否可以看到定义的placeholder")
        model_train_server("d:/saveModel/estimator_export")

    def testLoss(self):
        print("均方误差")
        predictions = tf.constant([1, 1], dtype=tf.int64)
        labels = [0, 1]
        loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
        print(self.sess.run(loss))
        print("二分类交叉熵")
        predictions = tf.constant([[0.1, 0.9], [0.1, 0.9]])
        labels = [1, 0]
        loss = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=labels)
        print(self.sess.run(loss))
        print("pair对的损失")
        tf.losses.mean_pairwise_squared_error

    # def variable_with_weight_loss(shape, stddev, wl):
    #     """
    #     创建一个带l2正则化的shape形状的正太分布权重参数
    #     :param stddev:
    #     :param wl:
    #     :return:
    #     """
    #     var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    #
    #     if wl is not None:
    #         weights_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weights_loss")
    #         tf.add_to_collection("losses", weights_loss)
    #     return var

    def testConv(self):
        print("卷积神经网络测试, 图像数据需要reshape到4维度, batch_size, 长, 高, 通道")
        x = [
            [1.0, 2.0, 2.0, 3.0]
        ]
        x = tf.Variable(x, dtype=tf.float32)
        x = tf.reshape(x, [-1, 2, 2, 1])
        filter_w = tf.Variable(tf.truncated_normal([2, 2, 1, 1], stddev=0.1))

        value = tf.nn.conv2d(
            input=x  # batch_size, 长, 高, 通道
            , filter=filter_w  # 卷积核的4为数据, [height,width,in_channels,out_channels]
            , padding='SAME'  # SAME 增加0列或者0行,使得所有的信息都可以利用到, VALID:舍弃
            , strides=[1, 1, 1, 1]
        )
        self.sess.run(tf.global_variables_initializer())
        print(self.sess.run(value))
        """ 由于padding, 所以输出的size和input的时候是一样的
        [[[[0.17726901]
           [0.13305116]]
        
          [[0.01567679]
           [0.16338262]]]]
        """
