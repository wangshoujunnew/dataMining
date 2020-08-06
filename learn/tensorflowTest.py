from unittest import TestCase
import tensorflow as tf

class TensorflowTest(TestCase):
    def setUp(self):
        self.sess = tf.Session()
        print("init ...")

    def tearDown(self):
        self.sess.close()
        print("clern down ... ")

    def testStart(self):
        self.sess.run(tf.global_variables_initializer())
        print("test start... ")
