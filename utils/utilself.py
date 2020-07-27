import pandas as pd
import json
import time
import zipfile


class Counter:
    """
    记录处理文件的行数
    1. 查看处理过程中错误信息
    2. 总共错误了多少行
    """
    time_start = 0
    counter = 0
    error_lines = 0
    act_error_lines = 0  # 文件中有多少行真实出错
    error_msg = []  # 错误信息
    line_frequ = 50000  # 每加载多少行显示一下进度

    @staticmethod
    def data_counter(init=False):
        # 数据处理加载器
        if init:
            Counter.time_start = int(time.time())

            Counter.counter = 0
            Counter.error_lines = 0
            Counter.act_error_lines = 0
        else:
            Counter.counter += 1
            if Counter.counter % Counter.line_frequ == 0:
                print("[process line {}]".format(Counter.counter))

    @staticmethod
    def show_error():
        if Counter.error_lines > 0:
            print("\n".join(Counter.error_msg))

        time_end = int(time.time())
        print("time cost: {} second".format(time_end - Counter.time_start))

    @staticmethod
    def add_error(info):
        Counter.act_error_lines += 1
        if Counter.error_lines < 100:
            Counter.error_lines += 1
            Counter.error_msg.append(info)


# 采用utf-8打开文件
def open_new(fi):
    return open(fi, "r", encoding="utf-8")


# map对象输出到文件中, key\tmapInfo
def map_w_file(dict_obj, f):
    fo = open(f, "w", encoding="utf-8")
    for k, v in dict_obj.items():
        fo.writelines("{}\t{}\n".format(k, json.dumps(v)))

    fo.close()


# 将list中的元素编程字符串
def map2str(l):
    return list(map(lambda x: str(x), l))


# 将文件load,然后加载成map对象
def load2obj(f, sample, header=None, seq="\t"):
    print("======load")
    Counter.error_lines = 0
    """f: 文件位置
       sample: 每一行的类型[(name, int:不要用0代替int,直接用int)]
       header: 只取文件的前header行
    """
    tmp = {}
    Counter.data_counter(True)
    for f in open_new(f):
        try:
            arr = f.strip().split(seq)
            Counter.data_counter()
            if header != None and Counter.counter > header:
                return tmp
            if sample == dict:
                tmp[arr[0]] = json.loads(arr[1])
            elif type(sample) == list:
                line_json = dict(zip(list(map(lambda x: x[0], sample)), arr[1:]))
                for k, v in sample:
                    if v == int:
                        line_json[k] = int(line_json[k])
                    elif v == float:
                        line_json[k] = float(line_json[k])
                    else:
                        pass  # 字符串不做处理,保持原样
                tmp[arr[0]] = line_json
        except Exception as e:
            Counter.add_error("{}, {}".format(f, e))

    Counter.show_error()
    print("总行数: {}, [查看文件5条样例]================".format(len(tmp)))
    print(list(tmp.items())[:5])
    return tmp


def zip_read(**kwargs):
    """
    zip压缩文件的读取
    :return:
    """
    zipf = kwargs["zipf"]
    csvfs = kwargs["csvfs"]
    seq = kwargs["seq"] if "seq" in kwargs.keys() else ","
    with zipfile.ZipFile(zipf, "r") as zf:
        for csvf in csvfs:
            data = zf.open(csvf)
            df = pd.read_csv(data)
            yield df


if __name__ == '__main__':
    # 时间测试
    df = zip_read(zipf="C:\\Users\\shoujunw\\PycharmProjects\\dataMining\\algorithm\\zoo5998.zip", csvfs=["Zoo.csv"])
    df = next(df)
    print(df.head(10))
