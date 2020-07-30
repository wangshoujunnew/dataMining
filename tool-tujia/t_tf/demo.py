import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes

print(tf.__version__)


# c1 = tf.constant([0], name="con1")  # name: con1:0
# c2 = tf.constant([0], name="con1")  # name: con1_1:1, 快速找到变量对应在图中的位置
# c3 = tf.constant([0])  # name: Const:0
# c4 = tf.Variable([0], dtype=dtypes.float32)  # name: Variable:0
# c5 = tf.Variable([0], dtype=dtypes.float32)  # name: Variable_1:0
# c6 = tf.Variable([0], dtype=dtypes.float32)  # name: Variable_2:0
#
# g1 = tf.Graph()
# with g1.as_default():
#     # v1这个张量的名字为v1:0
#     v = tf.get_variable("v1", initializer=tf.ones_initializer(dtype=dtypes.float32), shape=(10, 10))
#     with tf.variable_scope("test"):
#         c7 = tf.constant([0], name="test1")  # name: test/Const:0
#         print("快速获取当前变量的tensor_name =========> ", c7.name)
#         c8 = tf.constant([0])  # name: test/Const_1:0, 快速找到变量对应在图中的位置
#         c9 = tf.get_variable("test2", initializer=tf.zeros_initializer(dtype=dtypes.float32), shape=(10, 10))
#
# print(g1)


# def save():
#     with tf.Session(graph=g1) as sess:
#         saver = tf.train.Saver()
#         sess.run(tf.global_variables_initializer())
#
#         with tf.variable_scope("", reuse=True):
#             # v = g1.get_tensor_by_name("v") # 值的形式 <op_name>:<output_index>
#             # v_assign = g1.get_tensor_by_name("v/Assign")
#             print(sess.run(tf.get_variable("v1")))  # 变量名
#
#         with tf.variable_scope("test", reuse=True):
#             print(sess.run(tf.get_variable("test2")))  # 变量名, 只能获取tf.get_variable 方法创建的变量
#
#         print(sess.run(g1.get_tensor_by_name("v1:0")))
#         print(sess.run(g1.get_tensor_by_name("test/Const:0")))  # tensorname: scop/<>:<>
#
#         saver.save(sess, "model/model.ckpt")
#
#         writer = tf.summary.FileWriter("tensorboard", g1)
#         writer.close()


def load():
    # 单纯的查看bert需要又数据配置等的支持,本地无法实现, 在对应的服务器上实现
    """Error
    Failed to get matching files on /data/sjw/chinese_L-12_H-768_A-12/bert_model.ckpt: Not found: FindFirstFile failed for:
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("bert_model/model.ckpt-1286.meta")
        sess.run(tf.global_variables_initializer())
        #
        saver.save(sess, "model_testmodel/bert.ckpt")
        # # graph = sess.graph
        # # print(sess.run(graph.get_tensor_by_name("v1:0")))
        #
        # saver.save(sess, "model/test.ckpt")
        writer = tf.summary.FileWriter("tensorboard_bert", sess.graph)
        writer.close()




load()
