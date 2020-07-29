#encoding=utf-8
# https://zhuanlan.zhihu.com/p/31345125 em算法python实现
# http://www.hankcs.com/ml/em-algorithm-and-its-generalization.html 算法代码详细说明

import numpy as np
from scipy import stats

def em_single(priors, observations):
    """
    EM算法单次迭代: 适用于概率模型的参数估计
    假设现在有两个硬币A和B，我们想要知道两枚硬币各自为正面的概率啊即模型的参数。我们先随机从A,B中选一枚硬币，然后扔10次并记录下相应的结果，H代表正面T代表反面。对以上的步骤重复进行5次。如果在记录的过程中我们记录下来每次是哪一枚硬币（即知道每次选的是A还是B），那可以直接根据结果进行估计（见下图a）。
    隐藏变量: 不知道当前是哪个硬币

    ===== 三硬币的规则和二硬币的规则有点不一样, 所以实现过程看的有点懵逼
        假设有3枚硬币，分别记做A，B，C。这些硬币正面出现的概率分别是π,p和q。进行如下掷硬币实验：先掷硬币A，根据其结果选出硬币B或C，正面选B，反面选硬币C；然后投掷选重中的硬币，出现正面记作1，反面记作0；独立地重复n次（n=10)，结果为1111110000
    我们只能观察投掷硬币的结果，而不知其过程，估计这三个参数π,p和q。

    Arguments
    ---------
    priors : [theta_A, theta_B，theta_C]
    observations : [m X n matrix]

    Returns
    --------
    new_priors: [new_theta_A, new_theta_B,new_theta_C]
    :param priors: 先验, 这是一个二硬币模型, 长度为3是干嘛使得: 表示实验中每行独立实验室假设使用的是筛子A,B,C的概率, note:这代码有问题吧:, 第三个参数没啥用吧!!!, 最后还是二硬币模型吧, 但是为啥题目给的概率是[0.5, 0.8, 0.6], 和相加不等于1,
    所以这三个参数代表的含义是: ==> 每个硬币正面的先验概率
    :param observations:
    :return:
    """
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}} # 给的是个三硬币模型, 最后实现还是用的2硬币模型 todo ??
    theta_A = priors[0]
    theta_B = priors[1]
    theta_c = priors[2]
    # E step
    weight_As=[]
    for observation in observations:
        len_observation = len(observation)
        num_heads = observation.sum()
        num_tails = len_observation - num_heads
        contribution_A = theta_c*stats.binom.pmf(num_heads, len_observation, theta_A) # ========= 二项分布为啥前面要乘以theta_c ??三硬币模型中为啥要用到 第三个硬币的正面的概率?? 三硬币中为啥又只假设了两个隐变量的情况?? todo
        contribution_B = (1-theta_c)*stats.binom.pmf(num_heads, len_observation, theta_B)  # 两个二项分布: 表示的含义是: 假设此行数据时有A硬币抛出,那么A硬币正面的概率为contribution_A,假设此行数据由硬币B抛出,则此硬币正面的概率为contribution_B
        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B) # ========= 将两个概率正规化，得到数据来自硬币A或者b的概率：
        # 更新在当前参数下A、B硬币产生的正反面次数
        weight_As.append(weight_A) # ====== 为什么单独把weight_A给放到列表中, 为什么这个这个值类似于三硬币模型中的μ todo ??
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails
    # M step, 通过频率设置新的priors参数, ========== 这里虽然没有写最大似然,但是这里 通过频率计算新的概率, 就已经能让没有写的似然函数最大化了
    new_theta_c = 1.0*sum(weight_As)/len(weight_As) # ==== 一共进行了len(weight_As)次独立实验, 如果是而硬币的话,是没有这个new_theta_c的计算的,  搞懂这个三硬币中为啥是这个 todo ??
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A, new_theta_B,new_theta_c]

def em(observations, prior, tol=1e-6, iterations=10000):
    """
    EM算法
    :param observations: 观测数据
    :param prior: 模型初值, 每一行数据选择A的概率, 选择B的概率, 选择C的概率
    :param tol: 迭代结束阈值
    :param iterations: 最大迭代次数
    :return: 局部最优的模型参数
    """
    import math
    iteration = 0
    while iteration < iterations:
        new_prior = em_single(prior, observations)
        delta_change = np.abs(prior[0] - new_prior[0])
        if delta_change < tol: # ============ 循环截止条件, 参数变化小于阈值(还有些业务上使用 对数似然损失降低)
            break
        else:
            prior = new_prior
            iteration += 1
    return [new_prior, iteration]

# 硬币投掷结果观测序列：1表示正面，0表示反面。
observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

# ========== 为什么我的先验不是0.5,0.5,0.5, todo ??, 当我改成0.5的时候发现, 迭代一次就收敛了为啥 ??
print em(observations, [0.5, 0.8, 0.6])