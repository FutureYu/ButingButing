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

class_path = BUTING_PATH + r"\data\classes.txt"

def start_predict(image_path):
    """
    生成测试集的预测输出文件
    :param image_path: 图片地址
    :return: Class
    """

    # 加载训练好的模型进行预测
    output = model()
    predict = tf.reshape(output, [-1, CATEGORY_COUNT])
    max_idx_p = tf.argmax(predict, 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        pred = sess.run(max_idx_p,
                        feed_dict={INPUT: [DataSet.read_image(image_path)], DROPOUT_RATE: 0.})
    with open(class_path, "r") as f:
        content = f.readlines()
        return content[int(pred[0])]
    
def resize(image_path):
    src_image_path = image_path
    dst_image_path = "tmp.png"
    img = cv2.imread(src_image_path)
    # 转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 重设大小
    output = cv2.resize(img_gray, (IMAGE_WIDTH, IMAGE_WIDTH))

    cv2.imwrite(dst_image_path, output)
    return dst_image_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        Log("请输入参数")
        exit()
    image_path = sys.argv[1] 

    # 修改图片
    new_path = resize(image_path)
    # 判断
    Log(start_predict(new_path))
    os.remove(new_path)
