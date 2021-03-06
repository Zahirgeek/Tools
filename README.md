[split_files.py](./split_files.py)
- 将一个文件夹中的文件分割到多个文件夹中，每个文件夹中包含n个文件

[copy_files.py](./copy_files.py)

- 将一个文件夹下的二级目录所有文件复制/剪切到另一个文件夹中

[random_copy-move.py](./random_copy-move.py)

- 随机从一个文件夹中选取n个文件复制/剪切到另一个文件夹中

[recursive_files.py](./recursive_files.py)

- 将一个文件夹中所有子目录的文件复制/剪切到另一个文件夹

[match.py](./match.py)

- 用第一个文件夹内容匹配第二个文件夹的内容，将匹配到的第一个文件夹中的文件输出

### 车牌数据集处理

- 车牌数据的文件名命名方式为：<车牌名称>-<数字>.<后缀名>	(不包含<>)

[recursive_files.py](./车牌数据集处理/recursive_files.py)

- 将一个文件夹中所有子目录的文件复制/剪切到根目录，文件重名不覆盖，使用“文件名-数字”重命名文件

- *需要额外安装numpy

  #### ccpd2019数据集处理

  - *需要额外安装numpy和opencv-python

  [ccpd2019_rename.py](./车牌数据集处理/ccpd2019数据集处理/ccpd2019_rename.py)

  - 将CCPD2019图片的文件名重命名为车牌号

  [ccpd2019_matting.py](./车牌数据集处理/ccpd2019数据集处理/ccpd2019_matting.py)
  
  - 把ccpd2019中的车牌图片选择并抠出，保存到其他目录中
  
  [ccpd2019_image_correction.py](./车牌数据集处理/ccpd2019数据集处理/ccpd2019_image_correction.py)
  
  - 把ccpd2019中的车牌图片选择并抠出，重命名车牌并进行矫正

### PaddlePaddle数据集处理

[datasets_to_txt.py](PaddlePaddle/datasets_to_txt.py)

- 将数据集生成含有image label的txt文件

[trans_RGB.py](PaddlePaddle/trans_RGB.py)

- 把三通道RGB图片转换为单通道图片

### YOLO数据集处理

[voc2txt.py](YOLO/voc2txt.py)

- VOC标签信息存入txt文件