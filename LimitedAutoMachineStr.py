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

# 构建前缀字典树先
FontFixTree = Node()

for pattern in patterns:
    cur = FontFixTree
    for p in pattern:
        Node.charset.add(p) # 添加字符集

        last = cur
        if p not in cur.children.keys():
            cur.children[p] = Node(p)

        cur = cur.children[p]
        cur.last = last
FontFixTree

def search_font_fix(str_p, root:Node):
    """树的前缀搜索"""
    cur = root
    for s in str_p:
        if s in cur.children.keys():

        else:
            return None

def road_length(n:Node): # 当前节点到根节点的最长路径
    k = 0
    cur = n
    while True:
        if cur.last:
            k += 1
            cur = cur.last
        else:
            break
    return k

def check_max_back_fix(root:Node, pattern_back_fix):
    """
    检测其他分支和当前pattern的后缀匹配的前缀的最大长度, 然后最大程度的那个节点
    """
    m = len(pattern_back_fix)
    k = 0
    max_k_node = None
    cur = root
    for i in range(m):
        i = i + 1
        is_ok = True
        for p in pattern_back_fix[:i]:
            if p in cur.children.keys():
                cur = cur.children.get(p)
            else:
                is_ok = False
                break
        if is_ok:
            k = max(k, i)
            max_k_node = cur
    if k == 0:
        return k, root
    else:
        return k, max_k_node

# 当前节点的字符串
def cur_node_str(node:Node):
    s = ""
    cur = node
    while cur:
        if cur.v: # 头结点是没有v的
            s += cur.v
        cur = cur.last
    return s[::-1]

# 广度优先遍历树
def iter_tree(root:Node):
    queue = []
    for c in root.children:
        n = root.children[c]
        queue.append((c, n))
    while queue:
        c, n = queue.pop(0)
        for chilren_c in n.children:
            chilren_n = n.children[chilren_c]
            queue.append((chilren_c, chilren_n))

        max_m = road_length(n) # 当前分支的后缀和其他分支的前缀最大长度只能为max_m


        # 给每个节点的fail添加引用
        curNodeStr = cur_node_str(n)
        for x in Node.charset:
            if x not in n.children.keys():
                k, max_lenth_node = check_max_back_fix(root, curNodeStr+x)
                print("最大节点", k, cur_node_str(max_lenth_node))
                toFail = max_lenth_node
                n.fail[x] = toFail

        print(c, n)

iter_tree(FontFixTree)

FontFixTree