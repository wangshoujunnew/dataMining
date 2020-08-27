# -*- coding: utf-8 -*-
# # 包的导入

# + # estimator 保存的模型是啥样子
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
print(tf.__version__)
modelsave_dir = "/Users/tjuser/Desktop/modelSave/demo1"
logsave_dir = "/Users/tjuser/Desktop/modelSave/log/demo1"
modeldate_dir = "/Users/tjuser/Desktop/modelSave/data/minist"
# -

# # Session

session = tf.Session()


def train():
    # DataSets
    minist = input_data.read_data_sets(modeldate_dir, one_hot=False)
    print(minist)

    feature_columns = [
        tf.feature_column.numeric_column("image", shape=(784,))
    ]

    clf = tf.estimator.DNNClassifier(
        hidden_units=[10]
        # , model_dir=modelsave_dir # 训练模型的时候就会保存, tensorboard 都产生了, 可能是 DNNClassfier 中定义的写入的 log 位置为 model_dir
        , n_classes=10
        , feature_columns=feature_columns
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": minist.train.images} # 3minist.train.images: ndarray
        , y=minist.train.labels.astype(np.int32)
        , num_epochs=None
        , batch_size=128
        , shuffle=True
    )
    clf.train(train_input_fn, steps=10)
    # clf.sa
    # 模型的导出
    def serving_input_receiver_fn():
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_tensor')
        features = {"image": x_placeholder}
        receiver_tensors = {'predictor_inputs': x_placeholder}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    clf.export_savedmodel(
        modelsave_dir
        , serving_input_receiver_fn
        , strip_default_attrs=True)
# train()


def load():
    saver = tf.train.import_meta_graph("{}/model.ckpt-10.meta".format(modelsave_dir))
    with tf.Session() as sess:
        saver.restore(sess, "{}/model.ckpt-10".format(modelsave_dir))
        graph = sess.graph

        test_tensor = graph.get_tensor_by_name("save/SaveV2_1/tensor_names:0")
        print(test_tensor)

        print("=打印出网络结构=")
        # 那么 estimate 的输入时啥, 可以送 feed 吗
        print("打印出可训练的变量") # keys 表示的含义 https://blog.csdn.net/hustqb/article/details/80398934
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(v)

        print("=========\n 打印所有变量")
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v)

        print("=====\n 打印出模型中所有用于正向传播的变量")
        for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
            print(v)
        print("=====\n 记录到 tensorboard 中的变量")
        for v in tf.get_collection(tf.GraphKeys.SUMMARIES):
            print(v)
            
#         print("====\n 获取所有的集合")
#         for v in sess.graph.get_collection():
#             print(v)

        # for vv in tf.get_collection(tf.GraphKeys.INIT_OP) + tf.get_collection(tf.GraphKeys.TRAIN_OP) + tf.get_collection(tf.GraphKeys.READY_OP) + tf.get_collection(tf.GraphKeys.SUMMARY_OP ) + tf.get_collection(tf.GraphKeys.UPDATE_OPS) + tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP) + tf.get_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP):
        #     print(vv)
        # print(tf.GraphKeys.INIT_OP, tf.GraphKeys.TRAIN_OP, tf.GraphKeys.READY_OP, tf.GraphKeys.SUMMARY_OP , tf.GraphKeys.UPDATE_OPS, tf.GraphKeys.LOCAL_INIT_OP, tf.GraphKeys.READY_FOR_LOCAL_INIT_OP)

        # for v in tf.get_collection(tf.GraphKeys.TRAIN_OP):

# load() # hello

# + 查看tf.GraphKeys [markdown]
# # 查看所有的操作节点
# -

# a = tf.Variable(0)
# print(a)
# b = tf.Variable(1)
# print(b.name)
# c = a * b + b
# print(c) # 初始化的是 Variable, 通过乘出来的是 Tensor
#
# session.run(tf.global_variables_initializer())
# # 因为没有输入, 所以一直等着
# graph = session.graph
# print(session.run(session.graph.get_tensor_by_name("add:0"))
#       , session.run(graph.get_tensor_by_name("loss:0"))
#      )
#
#
# dir(tf.GraphKeys)
#
# # # Bert 的网络参数结构
#
# dir(tf.saved_model)
#
# # # 模型的保存 SaveModel 方式
#
# # +
# # saveModel 方式手写数字识别
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
# from tensorflow.saved_model.signature_def_utils import predict_signature_def
# from tensorflow.saved_model import tag_constants
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# sess = tf.InteractiveSession()
# x = tf.placeholder(tf.float32, [None, 784], name="Input") # 为输入op添加命名"Input"
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
# tf.identity(y, name="Output") # 为输出op命名为"Output"
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# tf.global_variables_initializer().run()
#
# for i in range(10):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     train_step.run({x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
#
# # 将模型保存到文件
# # 简单方法：
# tf.saved_model.simple_save(sess,
#                            "./model_simple",
#                            inputs={"Input": x},
#                            outputs={"Output": y})
# # 复杂方法
# builder = tf.saved_model.builder.SavedModelBuilder("./model_complex")
# signature = predict_signature_def(inputs={'Input': x},
#                                   outputs={'Output': y})
# builder.add_meta_graph_and_variables(sess=sess,
#                                      tags=[tag_constants.SERVING],
#                                      signature_def_map={'predict': signature})
# builder.save()
# # -
#
# # !ls -l model_simple/
#
# # !ls -l model_complex/
#
# # # saveModel 的加载
#
# # +
# import numpy as np
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "/Users/tjuser/Desktop/modelSave/demo1/1596714153")
    graph = tf.get_default_graph()
    for i, v in enumerate(graph.get_operations()):
        print(i, v.values())

    # print("得到可以操作的张亮==========") # 这两个是权重和 Bias
    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(v)
    #
    # input = np.expand_dims(mnist.test.images[0], 0)
    # x = sess.graph.get_tensor_by_name('Input:0')
    # y = sess.graph.get_tensor_by_name('Output:0')
    # batch_xs, batch_ys = mnist.test.next_batch(1)
    # scores = sess.run(y,
    #                   feed_dict={x: batch_xs})
    # print("predict: %d, actual: %d" % (np.argmax(scores, 1), np.argmax(batch_ys, 1)))
