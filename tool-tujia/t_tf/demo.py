import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

print(tf.__version__)
def opDefSelf():
    """
    自己定义的一些op操作
    :return:
    """
    c1 = tf.constant([0], name="con1")  # name: con1:0
    c2 = tf.constant([0], name="con1")  # name: con1_1:1, 快速找到变量对应在图中的位置
    c3 = tf.constant([0])  # name: Const:0
    c4 = tf.Variable([0], dtype=dtypes.float32)  # name: Variable:0
    c5 = tf.Variable([0], dtype=dtypes.float32)  # name: Variable_1:0
    c6 = tf.Variable([0], dtype=dtypes.float32)  # name: Variable_2:0

    myInput = tf.placeholder(dtype=dtypes.float32, shape=[None, 1], name="myInput")
    y = myInput + 1
    myOutput = tf.identity(y, name="myOutput")  # 给tensor重命名

    g1 = tf.Graph()
    with g1.as_default():
        # v1这个张量的名字为v1:0
        v = tf.get_variable("v1", initializer=tf.ones_initializer(dtype=dtypes.float32), shape=(10, 10))
        with tf.variable_scope("test"):
            c7 = tf.constant([0], name="test1")  # name: test/Const:0
            print("快速获取当前变量的tensor_name =========> ", c7.name)
            c8 = tf.constant([0])  # name: test/Const_1:0, 快速找到变量对应在图中的位置
            c9 = tf.get_variable("test2", initializer=tf.zeros_initializer(dtype=dtypes.float32), shape=(10, 10))

    print(g1)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())


def save():
    with tf.Session(graph=g1) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        with tf.variable_scope("", reuse=True):
            # v = g1.get_tensor_by_name("v") # 值的形式 <op_name>:<output_index>
            # v_assign = g1.get_tensor_by_name("v/Assign")
            print(sess.run(tf.get_variable("v1")))  # 变量名

        with tf.variable_scope("test", reuse=True):
            print(sess.run(tf.get_variable("test2")))  # 变量名, 只能获取tf.get_variable 方法创建的变量

        print(sess.run(g1.get_tensor_by_name("v1:0")))
        print(sess.run(g1.get_tensor_by_name("test/Const:0")))  # tensorname: scop/<>:<>

        saver.save(sess, "model/model.ckpt")

        writer = tf.summary.FileWriter("tensorboard", g1)
        writer.close()


def saveModelDemo():
    """
    使用saveModel的格式保存模型
    :return:
    """
    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())
        builder = tf.saved_model.builder.SavedModelBuilder("./saveModel")
        signature = predict_signature_def(inputs={'myInput': myInput},
                                          outputs={'myOutput': myOutput})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()


# saveModelDemo()
def loadSaveModel(sess: tf.Session):
    tf.saved_model.loader.load(sess, export_dir=r"C:\Users\shoujunw\PycharmProjects\dataMining\tool-tujia\t_tf\saveModel", tags=[tag_constants.SERVING])
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("myInput:0")
    y = graph.get_tensor_by_name("myOutput:0")
    yout = sess.run(y, feed_dict={
        x: [[1]]
    })
    print(yout)


# loadSaveModel(sess)


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


def load_bert():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("c:/Users/shoujunw/PycharmProjects/dataMining/modelSave/bert/model.ckpt-1286.meta")
        saver.restore(sess, "c:/Users/shoujunw/PycharmProjects/dataMining/modelSave/bert/model.ckpt-1286")

        # writer = tf.summary.FileWriter("bertGraph", sess.graph)
        # writer.close()
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v)

        print("======================")
        for v in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
            print(v)

        # tf.get_variable_to_shape_map


def getOPname(sess: tf.Session):
    # bert模型
    checkpoint_path = 'c:/Users/shoujunw/PycharmProjects/dataMining/modelSave/bert/model.ckpt-1286'
    saver = tf.train.import_meta_graph("c:/Users/shoujunw/PycharmProjects/dataMining/modelSave/bert/model.ckpt-1286.meta")
    saver.restore(sess, checkpoint_path)

    # 打印bert中所有的张亮操作, 顺序是, 前项传播, loss, 优化, 梯度, 反向传播
    for i, v in list(enumerate(sess.graph.get_operations()))[:100]:
        print("op{}, value{}".format(i, v.values()))


def loadEstimator():
    saver = tf.train.import_meta_graph(r"C:\Users\shoujunw\PycharmProjects\dataMining\tool-tujia\t_tf\estimatorModel\model.ckpt-400.meta")
    with tf.Session() as sess:
        saver.restore(sess, r"C:\Users\shoujunw\PycharmProjects\dataMining\tool-tujia\t_tf\estimatorModel\model.ckpt-400")
        for i, v in enumerate(sess.graph.get_operations()):
            print(i, v.values())

loadEstimator()


def testEstimator():

    es = tf.estimator.BaselineRegressor(
        model_dir="estimatorModel"
    )
    es.train()




# getOPname(sess)
# # load_bert()
#
# sess.close()
