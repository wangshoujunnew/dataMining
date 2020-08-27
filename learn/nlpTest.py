from unittest import TestCase
from nltk.corpus import stopwords, words  # 停用词汇, 和词汇
from nltk.corpus import wordnet as wn # 面向语义的英语词典, 单词和同义词


class NLPTest(TestCase):
    def setUp(self):
        print("test start .. ")

    def testNLTK(self):
        print("test NLTK")
        print(stopwords.words("english"))  # 加载英文词停用预料
        words.words()  # 词汇表
        print(wn.synsets("motorcar")) # 同义词集
        print(wn.synset("car.n.01").lemma_names())
        
    def testPE(self):
        print("测试位置编码, ，这样模型就具备了学习词序信息的能力。https://zhuanlan.zhihu.com/p/106644634")

