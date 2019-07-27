# encoding: utf-8

import tensorflow as tf
from rpi_define import *
from data_set import DataSet
from train import MODEL_DIR, model, INPUT, DROPOUT_RATE
from glob import glob
import numpy as np
import os
import cv2
import csv
import requests
import sys
from skimage import io, transform
import json
import matplotlib.pyplot as plt
import random
import shutil

class_path = BUTING_PATH + r"\data\classes.txt"


class Predictor:
    def __init__(self):
        self.output = model()
        self.sess = tf.InteractiveSession()

    def __del__(self):
        self.sess.close()

    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))

    def predict_img(self, image_path):
        """
        :param image_path: 图片地址
        :return: json
        """
        # 修改图片
        ori_path = image_path
        image_path = resize(image_path)

        # 判断
        res = []
        predict = tf.reshape(self.output, [-1, CATEGORY_COUNT])
        pred = self.sess.run(predict,
                             feed_dict={INPUT: [DataSet.read_image(image_path)], DROPOUT_RATE: 0.})

        with open(class_path, "r") as f:
            contents = f.readlines()
            for i, content in enumerate(contents):
                res.append({"name": content.split()[0], "prob": int(pred[0][i] * 10000)})

        res = sorted(res, key=lambda res: float(res['prob']), reverse=True)

        # 选取其他人画的
        nums = random.sample(range(0, 10), 2)
        otherpics = []
        for num in nums:
            otherpics.append(rf"..\static\dist\img\sp\{res[0]['name']}-{num}.png")
        file_name = image_path.split(r"\received")[1]
        ori_img = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)
        ori_img = cv2.resize(ori_img, (200, 200))
        if not os.path.exists(rf"{BUTING_PATH}\code\static\dist\img\sh"):
            os.makedirs(rf"{BUTING_PATH}\code\static\dist\img\sh")
        plt.imsave(rf"{BUTING_PATH}\code\static\dist\img\sh" + file_name, ori_img, cmap='gray')
        oripics = [r"..\static\dist\img\sh" + file_name]
        return {"size": len(pred[0]), "res": res, "otherpic": otherpics, "oripic": oripics}


def resize(image_path):
    src_image_path = image_path
    dst_image_path = image_path.split(".png")[0] + "_resized" + ".png"
    img = cv2.imread(src_image_path, cv2.IMREAD_GRAYSCALE)

    # 寻找中心
    height, width = img.shape

    left, top, right, bottom = width, height, -1, -1
    for y in range(height):
        for x in range(width):
            if img[y, x] < 128:
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    # 反转颜色
    for y in range(height):
        for x in range(width):
            img[y, x] = 255 - img[y, x]

    # 计算几何中心坐标
    centerX = (left + right) // 2
    centerY = (top + bottom) // 2
    w = right - left
    h = bottom - top

    # 将几何中心与画布中心重合
    new_img = np.zeros(img.shape)
    new_img[(height - h) // 2: (height + h) // 2, (width - w) //
            2: (width + w) // 2] = img[top: bottom, left: right]

    delta = max(h, w) // 2

    img = new_img[height // 2 - delta: height // 2 + delta,
                  width // 2 - delta: width // 2 + delta]
    # 重设大小
    output = transform.resize(img, (IMAGE_WIDTH, IMAGE_WIDTH))
    plt.imsave(dst_image_path, output, cmap='gray')
    return dst_image_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        Log("请输入参数")
        exit()
    image_path = sys.argv[1]

    p = Predictor()
    p.load_model()
    res = p.predict_img(image_path=image_path)
    Log("Predict result: ", res["res"][0]["name"])
