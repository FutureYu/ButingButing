# encoding: utf-8
import datetime
from rpi_define import *
from data_set import DataSet
import numpy as np
import tensorflow as tf
import os
import time
import tensorflow.contrib.rnn as rnn


MODEL_PATH = BUTING_PATH + r'\model\recognize.model'
MODEL_DIR = BUTING_PATH + r'\model'
DATA_PATH = BUTING_PATH + r"\data"


def init_w(shape):
    # shape: [filter_size_height, filter_size_width, color_channels, k_output]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))


def init_b(shape):
    return tf.Variable(tf.constant(0.01, shape=shape, dtype=tf.float32))


def conv(input_data, w, b, stride=1, use_bn=False):
    y = tf.nn.bias_add(tf.nn.conv2d(input_data, w, strides=[
                       1, stride, stride, 1], padding='SAME'), b)
    if use_bn:
        y = tf.layers.batch_normalization(y)
    return tf.nn.relu(y)


def pool(input_data):
    return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(input_data, rate=0.0):
    return tf.nn.dropout(input_data, rate=rate)


def batch_normalize(input_data):
    return tf.layers.batch_normalization(input_data)


def dense(input_data, output_units, activation=None):
    input_shape = input_data.get_shape().as_list()
    if len(input_shape) <= 2:
        x = input_data
    else:
        size = 1
        for i in range(1, len(input_shape)):
            size *= input_shape[i]
        x = tf.reshape(input_data, [-1, size])

    return tf.keras.layers.Dense(units=output_units, activation=activation)(x)


# batch_size, height, weight, channel
INPUT = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
LABEL = tf.placeholder(tf.float32, [None, CATEGORY_COUNT])
DROPOUT_RATE = tf.placeholder(tf.float32)


def model():
    x = INPUT

    # 卷积层 * 3
    w_conv_1 = init_w([3, 3, 1, 32])
    b_conv_1 = init_b([32])
    conv_1 = conv(x, w_conv_1, b_conv_1, stride=2)

    w_conv_2 = init_w([3, 3, 32, 64])
    b_conv_2 = init_b([64])
    conv_2 = conv(conv_1, w_conv_2, b_conv_2, stride=2)

    w_conv_3 = init_w([3, 3, 64, 128])
    b_conv_3 = init_b([128])
    conv_3 = conv(conv_2, w_conv_3, b_conv_3, stride=2)

    # 全连接层
    fully_conn_1 = dense(
        input_data=conv_3, output_units=256, activation=tf.nn.relu)
    dropout_fully_conn_1 = dropout(fully_conn_1, DROPOUT_RATE)

    # 输出层
    fully_conn_2 = dense(input_data=dropout_fully_conn_1,
                         output_units=CATEGORY_COUNT, activation=None)

    return fully_conn_2


def train_model(learning_rate=0.001, batch_size=32, dropout_rate=0.2, val_steps=100, max_step=1000000):
    """
    训练模型
    :param learning_rate: 学习率
    :param batch_size: 批次大小
    :param dropout_rate: dropout 比例
    :param val_steps: 每训练 val_steps 步就使用验证集进行一次验证
    :param max_step: 最大训练步数
    :return: None
    """
    output = model()

    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=LABEL))
    tf.summary.scalar('loss', loss)

    # 定义优化器
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # 定义准确率
    predict = tf.reshape(output, [-1, CATEGORY_COUNT])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(tf.reshape(LABEL, [-1, CATEGORY_COUNT]), 1)

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(max_idx_p, max_idx_l), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # 日志
    merge_summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            os.path.join('summary', datetime.datetime.now().strftime(
                '%Y-%m-%d_%H-%M-%S')),
            sess.graph)
        step = 0

        # 初始化参数
        sess.run(tf.global_variables_initializer())

        # 加载上次训练的结果继续训练
        last_check_point = tf.train.latest_checkpoint(MODEL_DIR)
        if last_check_point is not None:
            saver.restore(sess, last_check_point)
            Log('restore from {}'.format(last_check_point))
            step = int(str(last_check_point).split('-')[-1])

        Log('Start training ...')

        # 开始训练
        last_acc = 0.0
        last_loss = 1.0
        while step <= max_step:

            # 每隔 val_steps 训练次数使用验证集部分做验证
            if step % val_steps == 0:
                # 因为验证集较大，全部验证一次耗时较长，这里只取部分验证
                batch_x_val, batch_y_val = val_set.get_batch(50)
                acc, val_loss, pred = sess.run([accuracy, loss, max_idx_p],
                                               feed_dict={INPUT: batch_x_val, LABEL: batch_y_val, DROPOUT_RATE: 0.0})
                Log('step {} acc {}'.format(step, acc))

                # 如果效果有优化就保存当前的模型参数
                if acc > last_acc or val_loss < last_loss:
                    Log('save model at step {}'.format(step))
                    saver.save(sess, MODEL_PATH, global_step=step)
                    last_acc = acc
                    last_loss = val_loss

            # 获取一个批次的训练数据
            batch_x, batch_y = train_set.get_batch(batch_size)

            # 训练
            train_summary, _, train_loss, train_acc = sess.run(
                [merge_summary, train_op, loss, accuracy],
                feed_dict={INPUT: batch_x, LABEL: batch_y, DROPOUT_RATE: dropout_rate})
            Log('step [%d]  loss=%.8f  accuracy=%.8f' %
                (step, train_loss, train_acc))
            # 写入日志
            train_writer.add_summary(train_summary, step)

            step += 1

        train_writer.close()
        Log(f"Finish train at step {step}")
        # TODO save model


if __name__ == '__main__':
    # 训练集
    train_set = DataSet(DATA_PATH, DATA_PATH + r"\train.csv")
    # 验证集
    val_set = DataSet(DATA_PATH, DATA_PATH + r"\val.csv")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 开始训练模型
    train_model(learning_rate=0.001, batch_size=32,
                val_steps=100, max_step=1000000)
