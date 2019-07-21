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


def resize_test_images(src_image_dir, dst_image_dir):
    """
    将所有图片转换成相同的大小并存储到指定文件夹下
    :param src_image_dir: 原始图片所在的文件夹
    :param dst_image_dir: 存储到指定文件夹
    :return: None
    """
    src_images = glob(os.path.join(src_image_dir, "*.jpg"))
    for src_image_path in src_images:
        image_name = os.path.basename(src_image_path)
        dst_image_path = os.path.join(dst_image_dir, image_name)
        image_data = cv2.imread(src_image_path)
        output_image = cv2.resize(image_data, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imwrite(dst_image_path, output_image)


def make_output_csv(src_image_dir, output_csv_path):
    """
    生成测试集的预测输出文件
    :param src_image_dir: 图片目录地址
    :param output_csv_path: 输出csv的文件地址
    :return: None
    """
    # 打开/创建输出的 csv 文件
    file_test = open(output_csv_path, 'w', newline='')
    csv_writer = csv.writer(file_test)
    csv_writer.writerow(["ImageName", "CategoryId"])

    # 枚举测试集图片
    image_paths = glob(os.path.join(src_image_dir, "*.jpg"))

    # 加载训练好的模型进行预测
    output = model()
    predict = tf.reshape(output, [-1, CATEGORY_COUNT])
    max_idx_p = tf.argmax(predict, 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        for image_path in image_paths:
            pred = sess.run(max_idx_p,
                            feed_dict={INPUT: [DataSet.read_image(image_path)], DROPOUT_RATE: 0.})
            # 写入 csv 文件
            csv_writer.writerow([os.path.basename(image_path), pred[0] + 1])


if __name__ == '__main__':
    # 测试集图片文件夹地址
    SRC_IMAGE_DIR = r"E:\Weilan\easy\test"
    # 处理成统一大小的测试集图片的文件夹地址
    NORM_IMAGE_DIR = r"E:\Weilan\easy\norm_test"
    # 输出的结果文件 .csv 文件
    OUTPUT_CSV_PATH = r"E:\Weilan\easy\output.csv"

    # 如果测试集图片没有预处理成固定大小，则先进行预处理
    print("开始图片预处理")
    if not os.path.exists(NORM_IMAGE_DIR):
        os.makedirs(NORM_IMAGE_DIR)
        resize_test_images(SRC_IMAGE_DIR, NORM_IMAGE_DIR)
    print("图片预处理结束")

    # 生成结果文件
    print("开始生成结果")
    make_output_csv(NORM_IMAGE_DIR, OUTPUT_CSV_PATH)


