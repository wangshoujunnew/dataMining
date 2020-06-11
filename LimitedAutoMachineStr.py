"""
有限自动机配合字符串匹配
使用方法
1. build()
2. search()
"""


class State(object):
    def __init__(self, index):
        self.index = index
        self.next = {}


class LimitedAutoMachine(object):
    states = None  # pattern生成的所有状态
    pattern = None
    C = None  # 字符集
    m = None  # pattern的长度

    @staticmethod
    def max_k(p, t):
        len_t = len(t)
        max_k = 0
        for i in range(len_t):
            i = i + 1
            if p[:i] == t[len_t - i:len_t]:
                max_k = max(max_k, i)
        return max_k

    @staticmethod
    def build(pattern):
        LimitedAutoMachine.pattern = pattern
        LimitedAutoMachine.C = set([x for x in pattern])
        LimitedAutoMachine.m = len(pattern)
        m = LimitedAutoMachine.m
        LimitedAutoMachine.states = [State(x) for x in range(m)]
        LimitedAutoMachine.states.append(State(m))  # 所有的状态

        m, C, max_k, states = LimitedAutoMachine.m, LimitedAutoMachine.C, LimitedAutoMachine.max_k, LimitedAutoMachine.states

        for s_i in range(m):
            for c in C:
                text = pattern[:s_i] + c

                k = max_k(pattern, text)
                states[s_i].next[c] = states[k]

    @staticmethod
    def search_str(text):
        # search
        states, pattern = LimitedAutoMachine.states, LimitedAutoMachine.pattern

        curState = states[0]
        for i, x in enumerate(text):
            if curState.index == 7:
                print("匹配成功: 位置{}, {}".format(i - len(pattern), i))
                break

            if x in curState.next.keys():
                curState = curState.next[x]
            else:
                curState = states[0]

        if curState.index == 7:
            print("匹配成功: 位置{}, {}".format(i - len(pattern), i))
        else:
            print("没有匹配到任何的东西")

pattern = "ababaca"
LimitedAutoMachine.build(pattern)
text = "helloababacafsdfad"
LimitedAutoMachine.search_str(text)

# pattern = "ababaca"
# C = set([x for x in pattern])
# m = len(pattern)
#
# states = [State(x) for x in range(m)]
# states.append(State(m))  # 所有的状态
#
#
# def max_k(p, t):
#     len_t = len(t)
#     max_k = 0
#     for i in range(len_t):
#         i = i + 1
#         if p[:i] == t[len_t - i:len_t]:
#             max_k = max(max_k, i)
#     return max_k
#
#
# for s_i in range(m):
#     for c in C:
#         text = pattern[:s_i] + c
#
#         k = max_k(pattern, text)
#         states[s_i].next[c] = states[k]
#
# text = "helloababacafsdfad"
#
# # search
# curState = states[0]
# for i, x in enumerate(text):
#     if curState.index == 7:
#         print("匹配成功: 位置{}, {}".format(i - len(pattern), i))
#         break
#
#     if x in curState.next.keys():
#         curState = curState.next[x]
#     else:
#         curState = states[0]
#
# if curState.index == 7:
#     print("匹配成功: 位置{}, {}".format(i - len(pattern), i))
