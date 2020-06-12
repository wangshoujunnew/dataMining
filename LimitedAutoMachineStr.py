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

def testLAM():
    pattern = "ababaca"
    LimitedAutoMachine.build(pattern)
    text = "helloababacafsdfad"
    LimitedAutoMachine.search_str(text)

# ============ AC 自动机
patterns = ["abd", "abdk", "abchijn", "chnit", "ijabdf", "ijaij"]
target = "sdmfhsgnshejfgnihaofhsrnihao"

class Node(object):
    charset = set()
    def __init__(self, v=None):
        self.fail = {} # 也是一个dict, 不同的字符指向不同的失败串
        self.children = {}
        self.last = None
        self.v = v
        self.end = False


class ACTree(object):
    PrefixDictionaryTree = Node()

    @staticmethod
    def _build_prefix_DT(patterns):
        """
        通过pattern构建前缀字典树
        :param patterns:
        :return:
        """
        for pattern in patterns:
            currentNode = ACTree.PrefixDictionaryTree
            endNode = None
            for pChar in pattern:
                Node.charset.add(pChar)  # 添加字符集

                last = currentNode
                if pChar not in currentNode.children.keys():
                    currentNode.children[pChar] = Node(pChar)
                endNode = currentNode.children[pChar]

                currentNode = currentNode.children[pChar]
                currentNode.last = last

            endNode.end = True  # 最后一个节点设置为结束节点


    @staticmethod
    def _get_curnode_str(node: Node):
        """
        查看从根节点到当前节点的字符串
        :param node:
        :return:
        """
        init_char = ""
        curNode = node
        while curNode:
            if curNode.v:  # 头结点是没有v的
                init_char += curNode.v
            curNode = curNode.last
        return init_char[::-1]

    @staticmethod
    def _tree_prefix_search(target):
        """
        树的前缀搜索, 是为了找到路径A的后缀和其他路径的前缀的最大吻合长度
        :param target:
        :return:
        """
        curNode = ACTree.PrefixDictionaryTree
        for pChar in target:
            if pChar in curNode.children.keys():
                curNode = curNode.children[pChar]
            else:
                return None
        return curNode

    @staticmethod
    def _backFix_match_preFix_node(pattern_backfix):
        """
        使用当前pattern的后缀取匹配其他pattern的前缀, 返回能够匹配到的最大长度的其他pattern的节点
        :param root:
        :param pattern_backfix:
        :return:
        """
        root = ACTree.PrefixDictionaryTree

        max_length = len(pattern_backfix)

        # 搜索前缀
        result = root
        for i in list(range(max_length))[::-1]:  # 从最大开始搜索
            i += 1  # i表示要搜索的字符串长度, 而不是表示的下标, 而且搜索的是后缀
            curNode = ACTree._tree_prefix_search(pattern_backfix[max_length - i:max_length])
            if curNode != None:
                return curNode
            else:
                pass  # 继续搜索看看有没有长度稍微小一点的前缀
        return result  # 如果没有匹配的, 则只能返回根节点了

    @staticmethod
    def _build_fail_ref():
        """
        通过广度优先遍历给每个节点简历fail指针
        :return:
        """
        root = ACTree.PrefixDictionaryTree
        iter_queue = []
        for c in root.children:
            n = root.children[c]
            iter_queue.append((c, n))
        while iter_queue:
            c, n = iter_queue.pop(0)
            for chilren_c in n.children:
                chilren_n = n.children[chilren_c]
                iter_queue.append((chilren_c, chilren_n))

            # 给每个节点的fail添加引用
            curNodeStr = ACTree._get_curnode_str(n)
            for x in Node.charset:
                if x not in n.children.keys():
                    # 当前节点的字符串+不再此pattern的字符组成新的pattern, 查看这个pattern的后缀和其他pattern的前缀的匹配情况, 如果没有, 则返回的节点是root节点
                    toFail = ACTree._backFix_match_preFix_node(curNodeStr + x)
                    n.fail[x] = toFail

    @staticmethod
    def ac_search(text: str):
        curNode = ACTree.PrefixDictionaryTree
        find = []
        for i, s in enumerate(text):
            i += 1
            if s in curNode.children.keys():
                curNode = curNode.children[s]
            else:
                curNode = curNode.fail.get(s, ACTree.PrefixDictionaryTree)  # 如果没有失败节点指向, 则默认指向根节点

            if curNode.end:
                pattern = ACTree._get_curnode_str(curNode)
                # print("找到了: pattern: {}, 位置: {}, {}".format(pattern, i - len(pattern), i))
                find.append((pattern, i - len(pattern), i))  # 找到了什么pattern, 在text中的起始位置在哪里

        if find:
            print("{}找到了pattern".format(text))
            for i in find:
                print(i)
        else:
            print("{}啥也没有找到".format(text))


    @staticmethod
    def build_stream(patterns):
        """
        构建流程
        :return:
        """
        ACTree._build_prefix_DT(patterns)
        ACTree._build_fail_ref()

def test_ac():


    ACTree.build_stream(patterns)
    ACTree.ac_search(target)
    ACTree.ac_search("".join(patterns))

test_ac()