"蓝皮"


def grad_by_day(order_uv_list):
    v = []
    for i, x in enumerate(order_uv_list):
        v_i = 0
        if i - 1 < 0:
            pass
        else:
            v_i = order_uv_list[i] - order_uv_list[i - 1]
        v.append(v_i)
    return v


def show_info(info, channel, bucket, v, v_add):
    info = "info:{}, {}, {}\n".format(info, channel, bucket)

    info += "\n".join(["{}({})".format(x[0], x[1]) for x in zip(v, v_add)])
    print(info)

info = "orderuv"
channel = ["tujia", "ctrip"]
bucket = ["B", "E"]
import pymysql
for c in channel:
    for b in bucket:

        sql = """select {}, date from appu2ostatic where channel = "{}" and bucket="{}" order by date desc limit 10""".format(info, c, b)
        db = pymysql.connect("172.31.84.108","innovationman","pasZMHLYRDQ","innovation_data")
        cursor = db.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        d = list(map(lambda x: x[0], data))
        v = grad_by_day(d)
        v_add = grad_by_day(v)
        show_info(info, channel, bucket, v, v_add)
db.close()