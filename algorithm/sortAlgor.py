# 冒泡排序
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


maxLenHuiText(example, 0, len(example) - 1)
print("原数据: {}, 最长回文子串: {}".format(example, maxText))
