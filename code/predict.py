# encoding: utf-8

import tensorflow as tf
from rpi_define import *
from data_set import DataSet
from train import MODEL_DIR, model, INPUT, DROPOUT_RATE
from glob import glob
import os
import cv2
import csv
import requests
import sys
import json

class_path = BUTING_PATH + r"\data\classes.txt"


class Predictor:
    output = model()
    predict = tf.reshape(output, [-1, CATEGORY_COUNT])
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

    def __del__(self):
        self.sess.close()

    def predict_img(self, image_path):
        """
        :param image_path: 图片地址
        :return: json
        """
        # 修改图片
        new_path = resize(image_path)
        # 判断
        res = []
        pred = self.sess.run(self.predict,
                        feed_dict={INPUT: [DataSet.read_image(new_path)], DROPOUT_RATE: 0.})

        with open(class_path, "r") as f:
            contents = f.readlines()
            for i, content in enumerate(contents):
                res.append({"name": content[: -1], "prob": int(pred[0][i] * 10000)})
        
        res = sorted(res, key=lambda res : float(res['prob']), reverse = True)
        
        os.remove(new_path)
        return json.dumps({"size": len(pred[0]), "res": res})


def resize(image_path):
    src_image_path = image_path
    dst_image_path = "tmp.png"
    img = cv2.imread(src_image_path)
    # 转换为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 寻找中心
    height, width = img.shape
    left, top, right, bottom = width, height, -1, -1

    for y in range(height):
        for x in range(width):
            if img[y, x] < 128:
                if x < left: left   = x
                if y < top: top     = y
                if x > right: right = x
                if y > bottom: bottom = y  
    
    shiftX = (left + right) // 2
    shiftY = (top + bottom) // 2
    delta = max(max(shiftX - left, shiftX - right), max(shiftY - bottom, top - shiftY))
    if not(shiftY - delta - 10 < 0 or shiftY + delta + 10 > height 
        or shiftX - delta - 10 < 0 or shiftX + delta + 10 > width):
        delta += 10

    img = img[shiftY - delta: shiftY + delta, shiftX - delta: shiftX + delta]

    # 重设大小
    output = cv2.resize(img, (IMAGE_WIDTH, IMAGE_WIDTH))

    cv2.imwrite(dst_image_path, output)
    return dst_image_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        Log("请输入参数")
        exit()
    image_path = sys.argv[1]

    p = Predictor()
    res = json.loads(p.predict_img(image_path=image_path))
    Log(res["res"][0]["name"])
