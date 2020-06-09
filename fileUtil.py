import pandas as pd
import numpy as np


def list_2_file(l: list, f: str, mode="w+"):
    fw = open(f, mode, encoding="utf-8")
    l = list(map(lambda x: str(x), l))
    fw.writelines("\n".join(l))
    fw.close()


def open_(f):
    return open(f, "r", encoding="utf-8")


def load_2_pandas(f, col_nums, seq="\t"):
    """加载不规则的数据成为pandas"""

    def line_parse(line):
        arr = line.strip().split(seq)
        arr = arr[0:col_nums]
        if len(arr) <= col_nums:
            return arr + [None] * (col_nums - len(arr))
        else:
            arr

    l = list(map(lambda x: line_parse(x), open_(f).readlines()))
    df = pd.DataFrame(np.array(l))
    df.columns = [x for x in range(col_nums)]
    return df
