import json
from scipy import interpolate
from rpi_define import*
import pylab as pl
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from skimage import io, transform

DATA_JSON = BUTING_PATH+r"\data_ndjson"  # 路径
f_data = open(DATA_JSON+r"\classes.txt", 'r')

if not os.path.exists(BUTING_PATH + r"\code\static\dist\img\sp"):
    os.makedirs(BUTING_PATH + r"\code\static\dist\img\sp")


for eachline in f_data:
    # classes_txt文件中的每一行
    eachline = eachline.strip('\n')
    f = open(DATA_JSON+rf"\{eachline}.ndjson", 'r')
    for j in range(0, 10):
        line = f.readline()
        setting = json.loads(line)
        for i in range(0, len(setting['drawing'])):
            x = setting['drawing'][i][0]
            y = setting['drawing'][i][1]
            pl.plot(x, y, 'k')
        ax = pl.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        pl.axis('off')
        pl.savefig(rf"{BUTING_PATH}\code\static\dist\img\sp\{eachline}-{j}.png")
        pl.close()
        oldimg = cv2.imread(
            fr"{BUTING_PATH}\code\static\dist\img\sp\{eachline}-{j}.png", cv2.IMREAD_GRAYSCALE)
        newimg = cv2.resize(oldimg, (200, 200), interpolation=cv2.INTER_CUBIC)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dil = cv2.erode(newimg, kernel)
        plt.imsave(
            rf"{BUTING_PATH}\code\static\dist\img\sp\{eachline}-{j}.png", dil, cmap='gray')
    f.close()
    Log(f"{eachline} finished!")
f_data.close()
