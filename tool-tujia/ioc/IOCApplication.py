"""
类似java spring ioc 容器
"""

"""基础Bean, 不要让容器找Bean, *而是Bean主动向容器注册*, 在类的__init__方法添加注入函数"""


def registe(*args, **kwargs):
    pass


class Application(object):
    bean_map = {}

    @staticmethod
    def get_bean(name):
        if Application.bean_map.get(name) is None:
            raise Exception(f"bean {name} 不存在")
        else:
            pass


class Person(object):
    def __init__(self):
        pass

# import yaml
#
# app_yml_path = ""
#
#
# class Application:
#     obj_map = {}
#     yml_dict = yaml.load(open(app_yml_path, "r", encoding="utf-8")) if len(app_yml_path) > 0 else {}  # 这个需要静态加载, 而不是初始化加载
#
#
# # 自动注入注解, 解决的问题: 如何给属性上添加注解, 使用属性注解,将一个方法变成属性调用
# # 注入的思想,其实是一种聚合的思想
# # 默认全都用单例对象, 单利不单利不是使用它的人决定的, 而是该对象本身是否应该单利或者多里, 应该在yml中指明是否单利, 默认多利
# def AutoWrite(fn):
#     # 装饰类里面的方法
#
#     def wrapper(instance, **kwargs):
#
#         name = fn.__name__
#
#         # 如果此对象已经有该属性值, 则直接返回
#         if eval(f"instance._{name}") is not None:
#             print("has value ... ", eval(f"instance._{name}"))
#             return eval(f"instance._{name}")
#
#         # if name in Application.obj_map.keys():
#         #     return Application.obj_map[name]
#         new_name = str(name[0]).upper() + name[1:]
#
#         def look_up_from_yml(name, yml_dict):
#             """从IOC的yml配置文件查看此对象是否有配置"""
#             if name in yml_dict.keys():
#                 param = yml_dict[name]
#             else:
#                 param = ""
#             return param
#
#         param = look_up_from_yml(name, Application.yml_dict)
#         is_single = False
#         if "single" in param.keys():
#             is_single = True
#         param.pop("single")
#
#         code = f"{new_name}({param})"
#         print(code)
#         new_obj = eval(code)  # 对象的初始化全靠字典参数
#
#         if is_single:
#             Application.obj_map[name] = new_obj
#
#         return Application.obj_map[name]
#
#     return wrapper
#
#
# class Name(object):
#     pass
#
#
# class Person(object):
#
#     @AutoWrite
#     def name(self):
#         if hasattr(self, "_name"):
#             pass
#         else:
#             self._name = None
#
#
# if __name__ == '__main__':
#     app_yml_path = "ffadfa"
#     p1 = Person()
#     p2 = Person()
#
#     p1.name()
#     p2.name()
#
#     # print(id() == id(p2.name_more()))
#
#     a1 = eval("Name()")
#     a2 = eval("Name()")
#     print(p1 == p2)
