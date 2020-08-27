# 强化学习 UCB 相关, https://goldengrape.github.io/posts/python/shi-yong-pythonjin-xing-bei-xie-si-tong-ji-fen-xi/ 使用 pymc 进行贝叶斯统计
# 随机数
from unittest import TestCase
import numpy as np
import scipy.stats as stats


class UCBTest(TestCase):
    def init(self):
        print("初始化 Beta 分布")
        # 10个臂, 初始化, 每个都是成功 1 次, 失败一次
        prior_a = 1.  # aka successes
        prior_b = 1.  # aka failures
        self.estimated_beta_params = np.zeros((10, 2))
        self.estimated_beta_params[:, 0] = prior_a  # allocating the initial conditions
        self.estimated_beta_params[:, 1] = prior_b

    def getEveScoreByBeta(self, estimated_beta_params):
        """
        根据 目前 n 个 iterm 的 succ 和 fail 情况, 计算每个 item 的得分
        :param estimated_beta_params:
        :return:
        """
        t = float(estimated_beta_params.sum())  # 已经摇臂的总次数
        # 每个臂的试验次数
        totals = estimated_beta_params.sum(1)
        # 每个臂成功的次数
        successes = estimated_beta_params[:, 0]
        estimated_means = successes / totals  # 收益均值

        # 收益均值加置信区间
        score = estimated_means + np.sqrt(2 * np.log(t) / totals)
        return score

    def sampleAndUpdateBeta(self, estimated_beta_params, index):
        """
        根据指定的 beta 超参数产生随机数采样, 然后更新 beta 分布的超参数
        :param estimated_beta_params:
        :return: None
        """
        # 从 beta 分布中随机选择 0 和 1, 然后更新 beta 参数
        a, b = estimated_beta_params[index]
        rng = stats.beta(a, b)
        sample = rng.rvs(1)
        if sample > 0.5:
            estimated_beta_params[index][0] += 1
        else:
            estimated_beta_params[index][1] += 1
        # 更新 beta 参数

    def testUcb(self):
        self.init()
        score = None
        for i in range(10):
            score = self.getEveScoreByBeta(self.estimated_beta_params)
            max_index = np.argmax(score)
            self.sampleAndUpdateBeta(self.estimated_beta_params, max_index)
        print("最终每个臂的收益")
        # 最后查看每个臂的收益情况, 当实验次数很小的时候, 每个臂的波动性能打, 有明显区别, 由于每个臂初始化的时候成功次数都是平等的,  但是随着次数的增多, 每个臂的收益几乎一样
        print(score)
