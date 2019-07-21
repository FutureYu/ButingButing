# encoding: utf-8

# 分类
CATEGORY_COUNT = 20

# 图像大小
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

BUTING_PATH = r"E://ButingButing"


import time

def now():
    return time.strftime("[%H:%M:%S]", time.localtime())


def Log(*args):
    print(time.strftime("[%H:%M:%S]", time.localtime()), end=" ")
    for arg in args:
        print(arg, end=" ")
    print()
