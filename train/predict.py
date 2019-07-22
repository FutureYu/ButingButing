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

CLASS_PATH = BUTING_PATH + r"\data\classes.txt"

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
    with open(CLASS_PATH, "r") as f:
        content = f.readlines()
        return content[int(pred[0])]
    


if __name__ == '__main__':
    # 测试集图片文件夹地址
    if len(sys.argv) < 2:
        Log("请输入参数")
        exit()
    IMAGE_PATH = sys.argv[1] 

    # 生成结果文件
    Log(start_predict(IMAGE_PATH))
