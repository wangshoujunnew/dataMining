# 快速排序
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


zRoute("LEETCODEISHIRING", 4)