import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
import csv
import pandas as pd
import random

npy_dir = "data_npy/"  # npy文件夹路径
dest_dir = "train_data/"  # 训练文件存储的路径
test_dir = "test_data/"  # 测试文件存储的路径


def split_train_and_val_data_set(src_csv_path, train_csv_path, val_csv_path):
    """
    从原始数据集中分离出训练集和验证集
    :param src_csv_path: 原始数据集标注.csv文件地址
    :param train_csv_path: 生成的训练集数据标注.csv文件地址
    :param val_csv_path: 生成的验证集数据标注.csv文件地址
    :return: None
    """
    file_train = open(train_csv_path, 'w', newline='')
    train_writer = csv.writer(file_train)
    train_writer.writerow(["ImageName", "CategoryId"])

    file_val = open(val_csv_path, 'w', newline='')
    val_writer = csv.writer(file_val)
    val_writer.writerow(["ImageName", "CategoryId"])

    csv_data = pd.read_csv(src_csv_path)

    categories = set()
    for i in range(csv_data.shape[0]):
        category = csv_data["CategoryId"][i]
        if (category not in categories) or (random.randint(0, 9) > 0):
            train_writer.writerow([csv_data["ImageName"][i], category])
            categories.add(category)
        else:
            val_writer.writerow([csv_data["ImageName"][i], category])



def npy2png(npy_dir, dest_dir, test_dir, name):
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    with open(f"{dest_dir}data.csv", 'a', newline='') as file_train:
        train_writer = csv.writer(file_train)
        con_arr = np.load(f"{npy_dir}{name}.npy")  # 读取npy文件
        for i in range(0, 5000):  # 循环数组 最大值为图片张数  三维数组分别是：图片张数
            arr = con_arr[i, :]  # 获得第i张的单一数组
            arr = arr.reshape([28, 28])
            plt.imsave(f"{dest_dir}{name}-{i}.png", arr, cmap='gray')  # 定义命名规则，保存图片为彩色模式
            train_writer.writerow([f"{dest_dir}{name}-{i}.png", f"{name}"])
        print(f"{name}-train_data, finish")
    with open(f"{test_dir}test.csv", 'a', newline='') as file_test:
        test_writer = csv.writer(file_test)
        for i in range(5000, 6000):  # 循环数组 最大值为图片张数  三维数组分别是：图片张数
            arr = con_arr[i, :]  # 获得第i张的单一数组
            arr = arr.reshape([28, 28])
            plt.imsave(f"{test_dir}{name}-{i}.png", arr, cmap='gray')  # 定义命名规则，保存图片为彩色模式
            test_writer.writerow([f"{test_dir}{name}-{i}.png", f"{name}"])
        print(f"{name}-test_data, finish")
    


if __name__ == "__main__":
    # with open(npy_dir + "/classes.txt") as f:
    #     names = f.read().split()
    #     file_data = open(f"{dest_dir}data.csv", 'w', newline='')
    #     file_writer = csv.writer(file_data)
    #     file_writer.writerow(["ImageName", "CategoryId"])
    #     for name in names:
    #         npy2png(npy_dir, dest_dir, test_dir, name)
    split_train_and_val_data_set(f"{dest_dir}data.csv", "train.csv", "val.csv")