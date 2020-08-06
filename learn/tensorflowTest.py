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


