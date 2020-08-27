from unittest import TestCase


class Util:
    pass


class TopicTest(TestCase):
    def testQuickSort(self):
        print("测试快速排序")

        # 选择一个基准 pivot

        arr = [1, 2, 3, 4, 5]

        # 得到基准 3
        left, right = 0, len(arr)
        mid = (left + right) / 2
        mid_value = arr[mid]
        print("mid_value: {}={}".format(mid, mid_value))
