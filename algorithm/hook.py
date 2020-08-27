# 注册函数,
# 钩子函数, 把我们自己实现的hook函数在某一时刻挂接到目标挂载点上, hook的概念在windows的消息响应机制里面体现的尤为明显
# 回调函数的使用

import time


class LazyPerson(object):
    def __init__(self, name):
        self.name = name
        self.watch_tv_func = None
        self.have_dinner_func = None  # 存储钩子函数的引用

    def get_up(self):
        print("%s get up at:%s" % (self.name, time.time()))

    def go_to_sleep(self):
        print("%s go to sleep at:%s" % (self.name, time.time()))

    def register_tv_hook(self, watch_tv_func):
        self.watch_tv_func = watch_tv_func

    def register_dinner_hook(self, have_dinner_func):
        self.have_dinner_func = have_dinner_func

    # 调用钩子函数
    def enjoy_a_lazy_day(self):

        # get up
        self.get_up()
        time.sleep(3)
        # watch tv
        # check the watch_tv_func(hooked or unhooked)
        # hooked
        if self.watch_tv_func is not None:
            self.watch_tv_func(self.name)
        # unhooked
        else:
            print("no tv to watch")
        time.sleep(3)
        # have dinner
        # check the have_dinner_func(hooked or unhooked)
        # hooked
        if self.have_dinner_func is not None:
            self.have_dinner_func(self.name)
        # unhooked
        else:
            print("nothing to eat at dinner")
        time.sleep(3)
        self.go_to_sleep()


# ============ 四个钩子函数的具体实现
def watch_daydayup(name):
    """
    :param name: 钩子函数的参数
    :return:
    """
    print("%s : The program ---day day up--- is funny!!!" % name)


def watch_happyfamily(name):
    print("%s : The program ---happy family--- is boring!!!" % name)


def eat_meat(name):
    print("%s : The meat is nice!!!" % name)


def eat_hamburger(name):
    print("%s : The hamburger is not so bad!!!" % name)


# =================== end


if __name__ == "__main__":
    lazy_tom = LazyPerson("Tom")
    lazy_jerry = LazyPerson("Jerry")
    # register hook
    lazy_tom.register_tv_hook(watch_daydayup)
    lazy_tom.register_dinner_hook(eat_meat)
    lazy_jerry.register_tv_hook(watch_happyfamily)
    lazy_jerry.register_dinner_hook(eat_hamburger)
    # enjoy a day
    lazy_tom.enjoy_a_lazy_day()
    lazy_jerry.enjoy_a_lazy_day()
