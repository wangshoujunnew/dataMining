# 冒泡排序
from copy import deepcopy


def sortSelf(example):
    if example is None or len(example) <= 1:
        return example

    for i in range(1, len(example) + 1):  # 表示第 1 次到第 len(example) 次的冒泡
        for j in range(len(example) - i):
            if example[j] > example[j + 1]:
                example[j], example[j + 1] = example[j + 1], example[j]
    return example


# print(sortSelf([1, 3, 4, 2, 5, 34, 6, 7, 8, 3, 4, 5, 6]))
# 最长回文字符串
example = "babad"
example = "cbbd"
maxText = ""


def maxLenHuiText(example, start, end):
    global maxText
    if end - start <= 0:
        return True
    if maxLenHuiText(example, start + 1, end - 1) and example[start] == example[end]:
        if len(maxText) < (end - start + 1):
            maxText = example[start:end + 1]
        return True
    else:
        return False


# maxLenHuiText(example, 0, len(example) - 1)
# print("原数据: {}, 最长回文子串: {}".format(example, maxText))

# 大数相乘
def bigDataAdd(a: list, b: list):
    if a == None or len(a) == 0:
        return b
    else:
        aMax = len(a) - 1
        bMax = len(b) - 1
        result = []
        jingwei = 0
        while aMax >= 0 or bMax >= 0:
            if aMax >= 0 and bMax >= 0:
                tmp = a[aMax] + b[bMax] + jingwei
                jingwei = tmp // 10
                weishu = tmp % 10
                result.append(weishu)
                aMax -= 1
                bMax -= 1
            elif aMax < 0:
                tmp = b[bMax] + jingwei
                jingwei = tmp // 10
                weishu = tmp % 10
                result.append(weishu)
                bMax -= 1
            else:
                tmp = b[bMax] + jingwei
                jingwei = tmp // 10
                weishu = tmp % 10
                result.append(weishu)
                aMax -= 1
        return result[::-1]


def bigDataChen(a: list, b: list):
    if a is None or len(a) == 0 or b is None or len(b) == 0:
        raise Exception("数据错误")
    bigdataAdds = []

    # b * a
    for i in range(len(a))[::-1]:
        tmp = []
        addNum = 0  # 进位数
        for j in range(len(b))[::-1]:
            # print("a*b= {} * {} + {}".format(a[i], a[j], addNum))
            chenResult = a[i] * b[j] + addNum
            addNum = chenResult // 10
            weiNum = chenResult % 10  # 尾数
            tmp.append(weiNum)
            # 需要在前面加多少个 0
        tmp = [0] * (len(a) - i - 1) + tmp
        # print(tmp[::-1])
        # exit(0)
        bigdataAdds.append(tmp[::-1])

    # 大数相加
    result_list = []
    for i in bigdataAdds:
        result_list = bigDataAdd(result_list, i)
    return result_list


# print(bigDataChen([1,2,3,4], [1,2,3,4]))
# 整数转为罗马数字

def zhen2luoma(num):
    # 组成的数除了等距还有特殊数字
    dicts = [(1000, 'M'), (900, "CM"), (500, 'D'), (400, "CD"), (100, 'C'), (90, "XC"), (50, 'L'), (40, "XL"),
             (10, 'X'), (9, "IX"), (5, 'V'), (4, "IV"), (1, 'I')]
    # teshuMap = {4: "IV", 9: "IX", 40: "XL", 90: "XC", 400: "CD", 900: "CM"}
    result = []
    i = 0
    while i < len(dicts):
        d = dicts[i]
        fix = num // d[0]
        num %= d[0]
        i += 1
        if fix <= 0:
            continue
        result += [d[1]] * fix
        num %= d[0]
    return result


# print(zhen2luoma(1994), zhen2luoma(4))
# 动态规划生成 n 对括号
def mkKuoHaoN(pois, left, right, n):
    if n == 1:
        print(pois[0:left] + ["(", ")"] + pois[right + 1:])  # todo

    else:
        for i in range(left, right + 1):
            tmpPois = deepcopy(pois)
            tmpPois[i] = "("
            mkKuoHaoN()


# 水壶问题
x = 1
y = 3
z = 2

# 定义操作节点
ops = ["clearX", "clearY", "fillX", "fillY", "X2Y", "Y2X"]
initState = (0, 0)
endState = {(1, 2), (0, 2)}

curState = None


def shuihu(x, y, z, hasOps=None):
    global curState
    if curState is None:
        curState = initState
    if curState in endState:
        print("找到了可行解")
        print(ops)
        return
    for op in ops:
        if op in hasOps:
            return

        if op == "clearX":
            x = 0
        elif op == "clearY":
            y = 0
        elif op == "fillX":
            x = 1
        elif op == "fillY":
            y = 3
        elif op == "X2Y":
            diff = 3 - y
            if x >= diff:
                y = 3
                x = x - diff
            else:
                tmp = x
                x = 0
                y += tmp

        elif op == "Y2X":
            diff = 3 - x
            if y >= diff:
                x = 1
                y = y - diff
            else:
                tmp = y
                y = 0
                x += tmp


        tmpOps = deepcopy(hasOps)
        tmpOps.add(op)
        shuihu(x, y, z, tmpOps)

# 二叉树的构建
preorder = [3,9,20,15,7] # 前序遍历
midorder = [9,3,15,20,7] # 中序遍历
# 3是树根, 3 有左子树 9, 右子树 15,20,7
class Node:
    def __int__(self):
        self.left = None
        self.right = None
        self.value = None

def f1(value, pres, mids):
    # pre 前序, mids 中序
    leftListPre, leftListMid, rightListPre, rightListMid = [], [], [], []

    split = -1
    for i in range(len(mids)):
        if mids[i] == value: # 左边的是左子树, 右边的是右子树
            split = i
            break;
    if split != -1:
        leftListMid = mids[0:split]
        rightListMid = mids[split+1:len(mids)]
    else:
        rightListMid = mids


    for i in range(1, len(pres)):
        if pres[i] in leftListMid:
            leftListPre.append(pres[i])
        else:
            rightListPre.append(pres[i])

    return leftListPre, leftListMid, rightListPre, rightListMid

def erchashu(preorder, midorder):
    root = Node()
    root.left = None
    root.right = None
    root.value = preorder[0]

    # 得到左子树的先序遍历和中序遍历
    # 在 midorder 中找到 root.value 的位置
    leftListPre, leftListMid, rightListPre, rightListMid = f1(root.value, preorder, midorder)
    if leftListPre != None and leftListMid != None and len(leftListPre) > 0 and len(leftListMid) > 0:
        root.left = erchashu(leftListPre, leftListMid)
    if rightListMid != None and rightListPre != None and len(rightListPre) > 0 and len(rightListMid) > 0:
        root.right = erchashu(rightListPre, rightListMid)
    return root

root = erchashu(preorder, midorder)

def bianliQian(root):
    print(root.value)
    if root.left != None:
        bianliQian(root.left)
    if root.right != None:
        bianliQian(root.right)

def bianliZhong(root):
    if root.left != None:
        bianliZhong(root.left)
    print(root.value)
    if root.right != None:
        bianliZhong(root.right)

bianliZhong(root)

