from unittest import TestCase
import pandas as pd
import re
import tensorflow as tf
import jieba
import jieba.posseg as pseg  # 词性标注
import subprocess
from functional import seq
from pandas.core.generic import NDFrame
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from sklearn.linear_model import LogisticRegression
from estimatorUtil import *
from sklearn2pmml import PMMLPipeline, sklearn2pmml


class TensorflowTest(TestCase):
    def setUp(self):
        self.sess = tf.Session()
        print("init ...")

    def init(self):
        print("session 变量等信息的初始化")
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

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
        # model_train_server("d:/saveModel/estimator_export")
        model_train_server("/Users/tjuser/Desktop/modelSave/estimator_export")

        """
        部署服务 http://lionheartwang.github.io/blog/2017/12/10/tensorflowmo-xing-bao-cun-yu-jia-zai-fang-fa/
        bazel build //tensorflow_serving/model_servers:tensorflow_model_server
        bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base
        """

    def testSaveModel(self):
        print("非 estimator 模型, 保存层 SaveModel ")
        x = tf.placeholder(dtype=tf.float32, shape=(1,), name="input")
        y = x + 1
        tf.identity(y, name="output")  # 最好重新命名
        export_dir = "/Users/tjuser/Desktop/modelSave/savemodel_export"
        """
        assets/ 添加可能需要的外部文件
        assets.extra/
        variables/
            variables.data-*****-of-*****
            variables.index
        saved_model.pb
        """
        subprocess.run("rm -rf {}".format(export_dir), shell=True)
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        signature = predict_signature_def(inputs={'input': x},
                                          outputs={'output': y})
        builder.add_meta_graph_and_variables(self.sess
                                             , tags=[tag_constants.TRAINING]
                                             , signature_def_map={'predict': signature})
        builder.save()

    def testLookModelStructSession(self):
        print("查看模型的结构, saveModel 的模型")
        model_save_dir = "/Users/tjuser/Desktop/modelSave/savemodel_export"
        tag = tag_constants.TRAINING
        tf.saved_model.loader.load(self.sess, [tag], model_save_dir)
        for i, v in enumerate(self.sess.graph.get_operations()):
            print(i, v.values())

        """
        0 (<tf.Tensor 'input:0' shape=(1,) dtype=float32>,)
        1 (<tf.Tensor 'add/y:0' shape=() dtype=float32>,)
        2 (<tf.Tensor 'add:0' shape=(1,) dtype=float32>,)
        3 (<tf.Tensor 'output:0' shape=(1,) dtype=float32>,)
        """

    def testLookModelStructEstimator(self):
        print("查看模型的结构, saveModel 的模型")
        model_save_dir = "/Users/tjuser/Desktop/modelSave/estimator_export/1596792407"
        tag = tag_constants.TRAINING
        tf.saved_model.loader.load(self.sess, tags=[tag_constants.SERVING], export_dir=model_save_dir)
        for i, v in enumerate(self.sess.graph.get_operations()):
            print(i, v.values())

        grap = self.sess.graph
        x = grap.get_tensor_by_name("input_tensor:0")
        y = grap.get_tensor_by_name("output_logits:0")
        print(self.sess.run(y, feed_dict={
            x: [[0.1]]
        }))

    def testPairWise(self):
        print("tensorflow pairwise test ... ")

    def testSeqmap(self):
        print("测试类似 scala 的函数式编程 package: PyFunctional, 实现单词统计")
        words = ["a", "b", "c", "a"]
        datas = seq(words).map(lambda x: (x, 1)).group_by_key().map(lambda x: (x[0], len(x[1])))
        print(datas)
        print("map 过程中打印")

        def map_self(x):
            print("this value = : {}".format(x))
            return x

        print(seq(words).map(map_self).to_list())
        print("类似 sql 操作")
        schemal = [("name", str), ("age", int)]
        datas = [
            ["shoujunw", 10],
            ["why", 22],
            ["wyh", "10"],
            ["wyh1", "10a"]
        ]

        def build_df(lists, schemal):
            """
            每行形成一个 json, 在程序中会做好类型校验
            :param lists:
            :param schemal:
            :return:
            """

            def map_self(x):
                result = {}
                for index, name_type in enumerate(schemal):
                    try:
                        if name_type[1] == int:
                            x[index] = int(x[index])
                        elif name_type[1] == float:
                            x[index] = float(x[index])
                        elif name_type[1] == str:
                            x[index] = str(x[index])
                        else:
                            pass
                        result[name_type[0]] = x[index]
                    except:
                        print("validate: data '{}' type is error, need {}, givend {}".format(x[index], name_type,
                                                                                             type(x[index])))

                return result

            return seq(lists).map(map_self)

        df = build_df(datas, schemal)
        print(df)
        response1 = df.filter(lambda x: x.get("age", -1) > 11 and "wh" in x["name"]).map(lambda x: x["name"])
        print(response1)

    def testPandas(self):
        print("测试 pandas 中的一些 sql 操作")
        data = [
            [10, 3],
            [10, 3],
            [9, 3],
            [9, 1],
        ]
        df = pd.DataFrame(data, columns=["col1", "col2"])
        print(df.head())
        print("sql where 操作")
        print(df.query("col1 > 9 and col2 < 2 and col1 in [2, 9, 10]"))
        # df.query('A < @Cmean and B < @Cmean') 过滤均值等 ...
        print("sql select 操作")
        print(df.eval("col1, col2, col1 + 10"))  # eval只能得到一个列名, 如果用,的话, 则得到的是一个数组类型的列, 得到的是 ndarray
        df.eval("col3 = col1 + 1", inplace=True)
        print(df)  # 得到的是 Series
        print("给字段使用自定义函数")
        f1 = lambda x: x + 1
        df["col1"] = df["col1"].map(f1)
        print(df)
        print("group by 的聚合函数")
        a1 = lambda x: sum(x)
        df_group: NDFrame = df.groupby(by=["col1"], as_index=True)
        print(df_group.agg({"col1": "count", "col2": a1, "col3": a1}))
        print("row number 应用")
        # 先排序, 后分组
        df["row_number"] = df.sort_values(["col2"], ascending=False).groupby(["col1"]).cumcount()
        print(df)
        print("rank 函数的用法, 当数值一样的时候, dense 方法索引相同, first 继续递增")
        df["rank"] = df.groupby(["col1"])["col2"].rank(method="dense", ascending=False).astype(int)
        print(df)
        print("窗口函数 leg, lead")
        print(df.groupby(["col1"])["col2"].shift(1))  # 移动的幅度为 -1, 向前移动一个
        print("滚动函数 rolling")
        # min_periods 窗口内观测数量如果小于它, 返回 Nan, win_type: 居中, 居左, 局又, on: 要计算的列, closed: 区间的开闭
        print(df.sort_values(["col2"], ascending=True).groupby(["col1"])["col2"].rolling(window=2).sum())
        """
        col1  数据的索引, sum, 以索引为准,可以拼接到原 df 中
        10    2    NaN
              3    4.0
        11    0    NaN
              1    6.0
        """

    def testPandasHight(self):
        print("测试 pandas 的高级功能")
        df = pd.DataFrame(
            [
                [1, 2],
                [2, 3]
            ], columns=["c1", "c2"]
        )
        print("分类下数据的统计量, all的时候还回打出众数和unique, include: numpy.number 只看数字, numpy.object: 看分类变量")
        print(df.groupby(["c1"]).describe(include='all', percentiles=[0.1, 0.2, 0.3, 1]))

        print("透视表index: groupby 的字段, values: 做聚合的列, aggfunc=[np.sum, np.mean], fill_value=0, margins=True")
        # margins=True 表示添加汇总 ALL, columns=[""], index+columns 共同构成了 group 元素, 只不过 columns 中的元素放在了列上
        print(pd.pivot_table(df, index=["c1"], values=["c2"]))  # 默认聚合用均值, 可以一下看多种聚合, groupby 每个字段只能看一种聚合
        print(df.info())

        print("""
        pandas 的所有绘图
        会得出与上述相同的结果
        • ‘bar’ or ‘barh’ for bar plots #条状图
        • ‘hist’ for histogram #频率柱状图（计算某些值出现的频率）
        • ‘box’ for boxplot #箱线图（）
        • ‘kde’ or ‘density’ for density plots #密度图（需要scipy这个包）
        • ‘area’ for area plots #区域图（不同域的面积占比）
        • ‘scatter’ for scatter plots #散点图 >>> plt.scatter(df['part A'], df['part B'])
        • ‘hexbin’ for hexagonal bin plots # >>> plt.hexbin(df['part A'], df['part B'], df['part C'])
        • ‘pie’ for pie plots #饼图，比较适合与Series对象，看不同的占比
        """)
        print("数据的相对分布图 dataframe.plot.scatter(x='A', y='B').show() ")

    def testTfRecord(self):
        print("测试 TfRecord 的制作和读取, 和使用 dataset 读取 tfrecord")
        print("定义 tfrecord 的格式")
        # 写入对象
        save_path = "/Users/tjuser/Desktop/modelSave/tfrecord/test"
        # subprocess.run("rm -rf {}/*".format(save_path), shell=True)
        writer = tf.python_io.TFRecordWriter(save_path)

        # 不同的数据类型
        def type_feature_andwrite(schemal, data_line, writer):
            """
            :param data_line: 一行数据, 对应了很多的特征 [特征 1, 特征 2, 特征 3]
            :param schemal: ("name": int), 特征 1 对应的名称和数据类型
            :return:
            """
            record_dict = {}

            for index, info in enumerate(schemal):
                feature_name, feature_type = info
                value = data_line[index]
                tmp_value = None
                if feature_type == int:
                    tmp_value = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
                elif feature_type == float:
                    tmp_value = tf.train.Feature(float_list=tf.train.FloatList(value=[value])),
                elif feature_type == bytes:
                    tmp_value = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])),
                else:
                    pass

                assert tmp_value != None, "data type is Error: {}".format(feature_type)
                record_dict[feature_name] = tmp_value
            example = tf.train.Example(features=tf.train.Features(feature=record_dict))
            writer.write(example.SerializeToString())

        record_exmple = [
            [1, 1.0]
        ]
        schemal = [("age", int), ("score", float)]
        for i in record_exmple:
            type_feature_andwrite(schemal, i, writer)

        writer.close()

        print("读取 tfrecord")
        print("测试 tensorflow featureColumns 的使用")
        file_path = "/Users/tjuser/Desktop/modelSave/tfrecord/test"
        dataset = tf.data.TFRecordDataset([file_path])

        def parser(record):
            features = tf.parse_single_example(
                record,
                features={
                    'age': tf.FixedLenFeature([], tf.int64),
                    'score': tf.FixedLenFeature([], tf.float32),
                }
            )
            return features["age"], features['score']

        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        next = iterator.get_next()
        print(self.sess.run(next))
        """
        (1, 1.0)
        """

    def getWeightNorm(self, shape, name):
        print("以正太分布初始化一个 shape 的权重举证")
        # 用Variable 的方式没有创建成功,改用 get_variable 的方式
        weight = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer)
        return weight

    def testEmbeding(self):
        print("test tensorflow 的 embeeding")
        embedding = self.getWeightNorm(shape=[10, 10], name="embeeding")
        self.init()
        print(self.sess.run(tf.nn.embedding_lookup(embedding, ids=[[1], [2]])))

    def testTfFeatureColumn(self):
        print("tensorflow feautre columns 的使用")
        # 特征字典
        features = {
            "class": [0, 1, 2, 3],  # 类别特征 -> one-hot 0,1,2,3, 特征的值必须是一个 list, list 的长度就是 batch_size
            "class2": ['a', 'b', 'a', 'a']
        }

        class_f = tf.feature_column.categorical_column_with_identity(key="class",
                                                                     num_buckets=4)  # 如果特征超出了 4 个桶, 则报错: ,如果线上出现了未知的桶如何处理 todo??
        # 使用指标列包裹
        class_f = tf.feature_column.indicator_column(class_f)
        # input_list = [class_f]
        # tensors = tf.feature_column.input_layer(features, input_list)
        # print(self.sess.run(tensors))

        class2 = tf.feature_column.categorical_column_with_vocabulary_list(key="class2", vocabulary_list=['a', 'b'])
        class2 = tf.feature_column.indicator_column(class2)
        input_list = [class_f, class2]
        tensors = tf.feature_column.input_layer(features, input_list)
        print("===========")
        print(self.sess.run(tensors))

    def testPyMc(self):
        print("测试 pymc3")
        import pymc3 as pymc

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
        print("pair对的损失, pair 对的个数, Cn2")
        predictions = [0.1, 0.9, 0.5]
        labels = [1, 0, 0]
        print(self.sess.run(tf.losses.mean_pairwise_squared_error(labels=labels, predictions=predictions)))

    def testRe(self):
        print("测试正则")
        print("替换")
        """
        . : 出了换行符的任意字符
        \w: [a-zA-Z0-9_],  \s: 空白字符
        {m,n}?: 贪婪模式, 尽可能匹配少的, 匹配 m 个, 如果不加?, 则竟可能匹配 n 个
        []: 集合
        """
        result = re.sub("[12]", "", "123")  # 把 123中 pattern 部分替换成""
        print(result)
        print("切割")
        print(re.split("[12]", "a1b2c"))

    def testJieBa(self):
        print("测试 jieba 分词")
        print("加载词典")
        # jieba.load_userdict()
        # 加载停用词 load_stop_words(), 或者词性过滤
        # 标点符号
        remove_chars = '[·’!"#$%&\'()＃！（）*+,-./:;<=>?@，：?★、…．＞【】［］《》？“”‘’[\\]^_`{|}~]+'

        print("全模式False(精确模式) 用于自然语言处理")
        result = jieba.cut("我来到北京清华大学", cut_all=False)
        print([*result])
        print("全模式, 所有可能组成的短语会形成词")
        result = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", cut_all=True)
        print([*result])  # 科学, 学院, 科学院, 中国科学院
        result = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
        print([*result])

        print("查看句子中的词性情况")
        words_pair = pseg.cut("我来到北京清华大学")
        print(" ".join(["{}/{}".format(word, flag) for word, flag in words_pair]))

    def testSkitLearnLR(self):
        print("测试 LR 模型的训练和加载")
        model = PMMLPipeline([("lr", LogisticRegression(multi_class='ovr'))])
        X = [[1, 2, 3], [1, 2, 4]]
        y = [0, 1]
        model.fit(X, y)
        sklearn2pmml(model, 'skitlearn_model/LR.pmml')


    def testNLTK(self):
        print("""
        NLP 工具包 NLTK 的使用
        搜索文本
        单词搜索：
        相似词搜索；
        相似关键词识别；
        词汇分布图；
        生成文本；
        
        =====
        获取和处理语料库/corpus 语料库和词典的标准化接口
        字符串处理/tokenize,stem 分词
        搭配发现/collocations t 检验, 卡方
        词性标识符/tag
        分类/classify, cluster 决策树,最大熵,贝叶斯,EM
        分块/chunk 命名实体
        解析/parse
        语义解释/sem, inference 
        指标评估/mertrics 精度,召回率,协议系数
        概率和估计
        应用/app, chat 图形化的关键词排序, 分析器, WordNet查看器, 聊天机器人
        """)
