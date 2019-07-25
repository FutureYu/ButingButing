# 不听不听·涂鸦识别
**使用前先运行命令`pip install -r requestments.txt`**
**使用前先运行命令`python code/init.py`**

## 1. 项目说明
- 数据集来自 Google QuickDraw 项并开源目，感谢他们团队对人工智能发展做出的贡献
- 项目要求训练出可以识别手绘涂鸦的模型，并可以在网页等多渠道与用户交互


## 2. 代码说明
项目结构如下
ButingButing
├── code                         
|    ├── train.py                  
|    ├── val.py
|    ├── predict.py               
|    ├── server.py               
|    ├── rpi_define.py               
|    └── web_things         
├── model
├── summary
├── data
|    ├── imgs.png
|    ├── data.csv
|    ├── train.csv
|    └── val.csv
├── data_npy
|    └── imgs.npy              
             

### `rpi_define.py`
手绘涂鸦识别的一些基本常量定义。
其中可以使用`Log()`函数输出是添加当前时间，例如：`Log("Creating eval data set...")`

### `data_set.py`
数据集预处理，将`npy`格式图片转为单个的`png`格式图片，并拆分出训练集和验证集；
实现了一个数据集类，以便于训练和验证过程中使用；

### `train.py`
模型训练。

### `val.py`
模型准确率验证。

### `predict.py`
图像识别接口


## 3. 数据集标注格式说明
- 文件格式

`.csv`

- 键值说明

|名称|类型|描述|
| --- | --- | --- |
|ImageName|string|图片文件名|
|ClassId|string|商品分类 ID|


## 4. 项目规范
* Modules: lower_with_under
* Packages: lower_with_under	 
* Classes: CapWords
* Exceptions: CapWords	 
* Functions: lower_with_under()
* Global/Class Constants: CAPS_WITH_UNDER
* Global/Class Variables: lower_with_under
* Instance Variables: lower_with_under
* Method Names: lower_with_under()
* Function/Method Parameters: lower_with_under	 
* Local Variables: lower_with_under	 


* 输出时应使用`Log()`输出含有时间的信息