# encoding: utf-8

import tensorflow as tf
import os
import csv
from rpi_define import *
from data_set import DataSet
from train import MODEL_DIR, model, INPUT, LABEL, DROPOUT_RATE


# 在验证集上测试准确率
if __name__ == '__main__':
    # 验证集图片文件夹地址
    DATA_DIR = BUTING_PATH + r"\data"
    # 验证集标注数据
    VAL_CSV_PATH = BUTING_PATH + r"\data\val.csv"
    LOG_PATH = BUTING_PATH + rf"\log"
    LOG_CSV_PATH = LOG_PATH + rf"\val_{today()}.csv"
    # 创建验证集对象
    val_set = DataSet(DATA_DIR, VAL_CSV_PATH)

    # 加载训练好的模型进行预测
    output = model()
    predict = tf.reshape(output, [-1, CATEGORY_COUNT])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(tf.reshape(LABEL, [-1, CATEGORY_COUNT]), 1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(max_idx_p, max_idx_l), tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

        all_acc = 0.0
        result = dict()
        for img, label, image_path, category_id in val_set:
            acc, pred = sess.run([accuracy, max_idx_p],
                                 feed_dict={INPUT: [img], LABEL: [label], DROPOUT_RATE: 0.})
            all_acc += acc
            # 保存预测的结果
            result[image_path] = {"label": category_id, "predict": pred + 1}

        # 打印准确率
        Log('accuracy={}'.format(all_acc / val_set.get_size()))

        # 打印预测错的结果
        i = 0

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        Errorcsv = open(LOG_CSV_PATH, "w", newline="")
        writer = csv.writer(Errorcsv)
        writer.writerow(['number', 'path', 'label', 'predict'])

        for image_path in result:
            if result[image_path]["label"] != result[image_path]["predict"]:
                writer.writerow(
                    [i, image_path, result[image_path]["label"], result[image_path]["predict"]])
                i += 1

        Errorcsv.close()
