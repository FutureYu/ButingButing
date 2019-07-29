import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
import csv
import pandas as pd
import random
from skimage import io
from rpi_define import *
import shutil

npy_dir = BUTING_PATH + r"\data_npy"  # npy文件夹路径
dest_dir = BUTING_PATH + r"\data"  # 训练文件存储的路径


class DataSet(object):
    def __init__(self, images_dir, csv_file_path, shuffle=True):
        """
        初始化数据集
        :param images_dir: 图片目录地址
        :param csv_file_path: 标注数据的.csv文件地址
        :param shuffle: 是否打乱顺序
        """
        self.shuffle = shuffle
        self.csv_data = pd.read_csv(csv_file_path)
        self.csv_data["ImagePath"] = [os.path.join(
            images_dir, file_name) for file_name in self.csv_data['ImageName']]
        self.pool = list()
        self.num_epoch = -1

    def get_size(self):
        """
        获取数据集大小
        :return: 数据集大小
        """
        return self.csv_data.shape[0]

    def reset_pool(self):
        """
        初始化采样池
        :return: None
        """
        self.pool = [_ for _ in range(self.get_size())]
        if self.shuffle:
            random.shuffle(self.pool)

    @staticmethod
    def read_image(path):
        """
        读取图片数据
        :param path: 图片路径
        :return: 图像数据矩阵
        """
        return (io.imread(path, as_gray=True) / 255.0).reshape([28, 28, 1])

    @staticmethod
    def id_to_onehot(category_id):
        """
        将分类ID转换为独热向量
        :param category_id: 分类ID
        :return: 独热向量
        """
        onehot = np.zeros([CATEGORY_COUNT])
        onehot[category_id - 1] = 1
        return onehot

    @staticmethod
    def onehot_to_id(onehot):
        """
        将独热向量转换为分类ID
        :param onehot: 独热向量
        :return: 分类ID
        """
        return np.argmax(onehot) + 1

    def _get_batch_indices(self, size):
        ndx_list = list()
        if size >= len(self.pool):
            remain = size - len(self.pool)
            ndx_list.extend(self.pool)
            self.reset_pool()
            self.num_epoch += 1
            ndx_list.extend(self._get_batch_indices(remain))
        elif size > 0:
            ndx_list.extend(self.pool[:size])
            self.pool = self.pool[size:]
        return ndx_list

    def get_batch(self, size):
        """
        获取一个批次的采样数据
        :param size: 批次大小
        :return: 图像数据和对应的标签
        """
        ndx_list = self._get_batch_indices(size)
        data = [DataSet.read_image(self.csv_data["ImagePath"][ndx])
                for ndx in ndx_list]
        label = [DataSet.id_to_onehot(self.csv_data["CategoryId"][ndx])
                 for ndx in ndx_list]
        return data, label

    def __iter__(self):
        """
        遍历所有采样数据和对应的标签
        :return: 单个采样数据和对应的标签
        """
        for p, c in zip(self.csv_data["ImagePath"], self.csv_data["CategoryId"]):
            yield DataSet.read_image(p), DataSet.id_to_onehot(c), p, c


def npy2png(npy_dir, dest_dir, name, catergory_id, train_num):
    if not os.path.exists(npy_dir):
        Log("npy not exists")  # 数据集不存在
        os.makedirs(npy_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    con_arr = np.load(f"{npy_dir}\{name}.npy")  # 读取npy文件
        
    with open(fr"{dest_dir}\train.csv", 'a', newline='') as file_train:
        train_writer = csv.writer(file_train)
        for i in range(0, train_num):  # 循环数组 最大值为图片张数  三维数组分别是：图片张数
            arr = con_arr[i, :]  # 获得第i张的单一数组
            arr = arr.reshape([28, 28])
            plt.imsave(rf"{dest_dir}\{name}-{i}.png", arr,
                       cmap='gray')  # 定义命名规则，保存图片为灰色模式
            train_writer.writerow([f"{name}-{i}.png", f"{catergory_id}"])
        Log(f"{name}-train_data, finish")

    with open(rf"{dest_dir}\val.csv", 'a', newline='') as file_test:
        test_writer = csv.writer(file_test)
        for i in range(train_num, train_num + 1):  # 循环数组 最大值为图片张数  三维数组分别是：图片张数
            arr = con_arr[i, :]  # 获得第i张的单一数组
            arr = arr.reshape([28, 28])
            plt.imsave(rf"{dest_dir}\{name}-{i}.png", arr,
                       cmap='gray')  # 定义命名规则，保存图片为灰色模式
            test_writer.writerow([f"{name}-{i}.png", f"{catergory_id}"])
        Log(f"{name}-test_data, finish")


def CopyFile(src_file, dst_file):
    if not os.path.isfile(src_file):
        Log(src_file, "not exist")
    else:
        fpath, fname = os.path.split(dst_file)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(src_file, dst_file)  # 复制文件
        Log("copy finished")


if __name__ == "__main__":
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    with open(rf"{dest_dir}\val.csv", 'w', newline='') as file_test:
        test_writer = csv.writer(file_test)
        test_writer.writerow(["ImageName", "CategoryId"])
    with open(rf"{dest_dir}\train.csv", 'w', newline='') as file_train:
        train_writer = csv.writer(file_train)
        train_writer.writerow(["ImageName", "CategoryId"])
    with open(npy_dir + r"\classes.txt") as f:
        names = f.read().split()
        for i in range(len(names)):
            npy2png(npy_dir, dest_dir, names[i], i + 1, 3)
    CopyFile(npy_dir + r"\classes.txt", dest_dir + r"\classes.txt")
    pass
