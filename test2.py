import datetime, time

from functional import seq

# hadoop fs ls -ls

myfiles = ["/user/tujiadev/data/rankdata_v2/house_base_info/", "/user/tujiadev/data/rankdata_v2/house_cluster_data/"]


def getLen(start_date, end_date):
    date_str=start_date
    add_c=1
    newEndDate = date_str
    for i in range(100):
        date_str=newEndDate
        if newEndDate == end_date:
            continue
        format_str = '%Y-%m-%d' if '-' in date_str else '%Y%m%d'
        now = datetime.datetime.strptime(date_str, format_str)
        delta = datetime.timedelta(days=1)
        n_days = now + delta
        newEndDate=str(n_days.strftime('%Y-%m-%d'))
        add_c += 1
    return add_c


print(getLen("2020-01-01", "2020-01-03"))