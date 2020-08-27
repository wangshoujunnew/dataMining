# 本地merge

from utilself import *

roombaseinfo = load2obj("roombaseinfoTmp", dict)
finalPrice = load2obj("finalPrice", [("showPrice", float)])
cityPrice = load2obj("cityPrice", [("cityPerMeterIndex", float), ("cityPerIndex", float)])
regionPrice = load2obj("regionPrice", [("regionPrice", float), ("regionPricePerson", float)])

print(list(roombaseinfo.items())[:10])
print(list(finalPrice.items())[:10])
print(list(regionPrice.items())[:10])

fw = open("roombaseinfo", "w", encoding="utf-8")

for house_id, info in roombaseinfo.items():
    if house_id in finalPrice.keys():
        info["showPrice"] = finalPrice[house_id]["showPrice"]
    if house_id in cityPrice.keys():
        info["priceLevel"] = {}
        info["priceLevel"]["cityPerIndex"] = cityPrice[house_id]["cityPerIndex"]
        info["priceLevel"]["cityPerMeterIndex"] = cityPrice[house_id]["cityPerMeterIndex"]
    if house_id in regionPrice.keys():
        info["regionPrice"] = regionPrice[house_id]["regionPrice"]
        info["regionPricePerson"] = regionPrice[house_id]["regionPricePerson"]

    fw.writelines("{}\t{}\n".format(house_id, info))

fw.close()
