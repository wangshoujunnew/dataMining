# 快速排序
exmaple = [2, 1, 3, 3, 0, 4, 5]
ex


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


qucickSort(exmaple, 0, len(exmaple) - 1)
print(exmaple)