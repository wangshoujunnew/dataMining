def list_2_file(l: list, f: str):
    fw = open(f, "w+", encoding="utf-8")
    l = list(map(lambda x: str(x), l))
    fw.writelines("\n".join(l))
    fw.close()

if __name__ == '__main__':
    list_2_file([1,2], "d:/tmp.dat")