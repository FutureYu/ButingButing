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
        image_path = resize(image_path)

        # 判断
        res = []
        predict = tf.reshape(self.output, [-1, CATEGORY_COUNT])
        pred = self.sess.run(predict,
                             feed_dict={INPUT: [DataSet.read_image(image_path)], DROPOUT_RATE: 0.})

        with open(class_path, "r") as f:
            contents = f.readlines()
            for i, content in enumerate(contents):
                if i > 9:
                    break
                res.append({"name": content.split()[
                           0], "prob": int(pred[0][i] * 10000)})

        res = sorted(res, key=lambda res: float(res['prob']), reverse=True)
        simpics = get_similar_pics(image_path, [res[0]["name"], res[1]["name"], res[2]["name"]])
        nums = random.sample(range(0, 100), 10)
        otherpics = []
        for num in nums:
            otherpics.append(rf"{BUTING_PATH}\{data_sp}\{res[0]['name']}\{num}.png")
        return {"size": len(pred[0]), "res": res, "simpic": simpics, "otherpic": otherpics}


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

def count_feature(image_path):
    '''
    计算特征值
    '''
    ori_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    feature_img = transform.resize(ori_img, (IMAGE_WIDTH, IMAGE_WIDTH))
    mean_value = np.mean(np.mean(feature_img))
    feature = feature_img >= mean_value
    feature = np.matrix(feature, np.int8)
    return feature.reshape([-1])

def get_similar_pics(image_path, names):
    # 计算特征
    feature = count_feature(image_path)
    # 读取三个文件夹的npy，并依次比较距离，选取最大的
    res = []
    for name in names:
        con_arr = np.load(rf"{BUTING_PATH}\{data_sp}\{name}\feature.npy")  # 读取npy文件
        index = 0    
        for i in range(0, 100):  # 循环数组
            arr = con_arr[i, :]  # 获得第i张的单一数组
            min = 28*28
            count = 0
            for j in range(28*28):
                if arr[j] != feature[j]:
                    count += 1
            if count < min:
                count = min
                index = i
        res.append(rf"{BUTING_PATH}\data_sp\{name}\{index}.png")
    return res


def get_features(name):
    arr = []
    for i in range(100):
        path = rf"{BUTING_PATH}\data_sp\{name}\{i}.png"
        arr.append(count_feature(path))
    np.save(arr)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        Log("请输入参数")
        exit()
    image_path = sys.argv[1]

    p = Predictor()
    p.load_model()
    res = p.predict_img(image_path=image_path)
    Log("Predict result: ", res["res"][0]["name"])
