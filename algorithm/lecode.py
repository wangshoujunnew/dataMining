# 快速排序
from copy import deepcopy

exmaple = [2, 1, 3, 3, 0, 4, 5]
exmaple = [0]


def qucickSort(nums, left, right):
    if left >= right:
        return

    jizhun = nums[left]
    leftPoint, rightPoint = left + 1, right
    leftIsFind, rightIsFind = False, False
    while leftPoint < rightPoint:
        if nums[leftPoint] <= jizhun:
            leftPoint += 1
        else:
            leftIsFind = True

        if nums[rightPoint] >= jizhun:
            rightPoint -= 1
        else:
            rightIsFind = True

        if leftIsFind and rightIsFind:
            # swap
            nums[leftPoint], nums[rightPoint] = nums[rightPoint], nums[leftPoint]

            leftIsFind, rightIsFind = False, False

    # 碰头了
    mid = None
    if nums[leftPoint] < jizhun:
        # 交换基准和nums[leftPoint]
        nums[left], nums[leftPoint] = nums[leftPoint], nums[left]
        mid = leftPoint
    else:
        # 交换基准和nums[leftPoint - 1]
        nums[left], nums[leftPoint - 1] = nums[leftPoint - 1], nums[left]
        mid = leftPoint - 1

    qucickSort(nums, left=left, right=mid - 1)
    qucickSort(nums, left=mid + 1, right=right)


# qucickSort(exmaple, 0, len(exmaple) - 1)
# print(exmaple)


# Z字走法
def getNext(i, j, nums):
    # 得到矩阵需要填充的下一个坐标
    if i <0 and j<0:
        return 0, 0
    else:
        if j % 3 == 0:
            if i != nums-1:
                return i+1, j
            else:
                return i-1,j+1
        else:
            return i - 1, j + 1

def zRoute(ss:str, nums:int):
    datas = {} # 存储对应坐标点的元素
    i, j = -1, -1

    for s in ss:
        nextI, nextJ = getNext(i, j, nums)
        i=nextI
        j=nextJ
        datas[(nextI,nextJ)] = s

    print(datas)
    maxCol = max(map(lambda x: x[1],list(datas.keys())))

    for i in range(nums):
        for j in range(maxCol):
            value = datas.get((i,j))
            if value == None:
                print("#", end="") # 坐标点没有的元素采用#填充
            else:
                print(value, end="")
        print("")


# zRoute("LEETCODEISHIRING", 4)
# 回溯法处理0-1背包
beibao = [2,1,3,2]
jiazhi = [12, 10, 20, 15]
mapDict = dict(zip(beibao, jiazhi))
C = 5 # 背包容量
maxValue = -1 # 最大价值
def bag01(C, wuping_jiazhi, muqianBag):
    """
    :param C: 背包的容量
    :param wuping_jiazhi: 物品及其对应的价值
    :param muqianBag: 目前背包里面转入的东西
    :return:
    """
    global maxValue
    if C < 0:
        return -1
    if len(wuping_jiazhi) == 0:
        value = sum(map(lambda x: mapDict.get(x), muqianBag))
        if value > maxValue:
            maxValue = value
            print("over === bag is: {}, maxValue: {}".format(muqianBag, maxValue))
            # return muqianBag
        return

    for w, value in wuping_jiazhi.items():
        # 对每个物品选择是和否
        tmpDict = deepcopy(wuping_jiazhi)
        del tmpDict[w]

        bag01(C, tmpDict, muqianBag) # 不选择w
        # 选择w
        if C-w >= 0:
            tmpBag = deepcopy(muqianBag)
            tmpBag.append(w)
            bag01(C-w, tmpDict, tmpBag) # 装入w之后剩余的容量

bag01(C, dict(zip(beibao, jiazhi)), [])

