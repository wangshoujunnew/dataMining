from unittest import TestCase
from fileUtil import *
from houseEmbed.UserSession import *
from functional.streams import *
from CommentTag import *
from Model import *
from HTTP import *
from SQL import *

class Test(TestCase):
    def __init__(self):
        self.data_dir = "d:/data/houseData"
        self.valid_data = f"{self.data_dir}/tujia_20191212.land.valid.data"

    def setUp(self):
        print("start ... 加载数据")
        self.comment_data = """
        {"createTime":"2018-11-29 07:37:12","updateTime":"2018-11-29 07:37:12","tags":[{"sort":"环境","topicId":122,"topic":"夜景","emotionType":0,"clause":"夜景很美","channelType":6,"enumDataStatus":0}]}
        """

        self.label = [1] * 10 + [0] * 10
        random.shuffle(self.label)
        self.predict = random.choices([1, 0], k=20)
        self.data_dir = "d:/data/houseData"
        self.valid_data = f"{self.data_dir}/tujia_20191212.land.valid.data"

    def test_load(self):
        SVMModel.load_data(self.valid_data)

    def test_svm_load(self):
        f, t = load_svmlight_file("d:/1.txt")
        print(f.todense())
        print(t)

    def test_feature(self):
        print(SVMModel.feature)

    def test_pearson(self):
        self.predict_new = self.predict[:15] + [None] * 5
        print(self.predict_new, "\n", self.predict)
        df = pd.DataFrame(np.concatenate([np.array(self.predict_new).reshape([-1,1]),
                                          np.array(self.predict).reshape([-1,1])], axis=1))
        SVMModel.feature_pearson_image(df)

    def test_sql(self):
        SQL.user_save()

    def test_grad(self):
        from autograd import grad
        def f(x, y): return 4*x + 5*y

        f_grad = grad(f, 0)
        print(f_grad(1., 0.))

    def test_pair(self):
        print(self.label)
        print(SVMModel.make_pair(self.label))

    def test_ap(self):
        SVMModel.rank_map(self.label, self.predict, k=10)
        SVMModel.rank_ndcg(self.label, self.predict, k=10)

    def test_http(self):
        HTTP.split_bucket()

    def test_model(self):
        # SVMModel.acc(self.label, self.predict)
        x = np.linspace(1, 10, 10).reshape((-1, 1))
        y = np.random.rand(10, 1)
        data = np.concatenate([x, y], axis=1)
        for i in range(100):
            # 产生噪点, 必须先转为1维度才可以赋值
            data[:, 1] = (np.random.rand(10, 1) + data[:, 1].reshape([10, 1])).reshape(-1)
            SVMModel.plt_dy_call(SVMModel.show_image_dy, data)

    def test_comment(self):
        data = CommentTag.parse_comment_tag(self.comment_data)
        data

    def test_on(self):
        x = open("d:/user_action_on_unit/log.dat", "r", encoding="utf-8").readlines()[-1]
        print(x)
        # a = UserSession.parse_line(x)
        # a

    def test_useraction(self):
        df = load_2_pandas("d:/user_action_on_unit/log.dat", len(user_action_on_unit_fields))
        df.columns = user_action_on_unit_fields
        df["sid"] = "a5fb980b-4418-4e48-8f54-84a26cecb646"

        # df = df[df["order"] == "true"]

        # 数据压缩成json
        data = seq(df.to_numpy()).map(lambda x:
                                      dict(zip(user_action_on_unit_fields, x)))

        # data = data.to_list()
        # data

        data = data.map(UserSession.user_group) \
            .group_by_key().map(lambda x: UserSession.parse_session(x[1])) \
            .map(lambda x: x.train_data())

        data = data.to_list()[0]
        print(data)