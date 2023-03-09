# 分割数据集
import os
import argparse
import random
import math
from shutil import copy

def get_args():
    parser = argparse.ArgumentParser(description="分割Yolov5obb数据集")
    parser.add_argument("--train", type=float,
                        help="训练集比例")
    parser.add_argument("--val", type=float,
                        help="验证集比例")
    parser.add_argument("--image_path", '-i', type=str,
                        help="图片位置")
    parser.add_argument("--label_path", '-l', type=str,
                        help="标注文件位置")
    parser.add_argument("--output", '-o', type=str, default="ddck",
                        help="分割后的数据输出位置，不指定则默认为该脚本同级ddck目录下")

    args = parser.parse_args()
    return args


args = get_args()
# train datasets占比
train_per = args.train
# val datasets占比
val_per = args.val
# test datasets占比
test_per = 1.0-(train_per+val_per)

assert train_per > 0 and test_per >= 0 and test_per < 1 and train_per < 1 and val_per < 1, 'train test val的比例不正确，重新输入'

# 图片和标签所在的目录
# image_folder = os.path.join('yolo_datasets', 'images')
image_folder = args.image_path
label_folder = args.label_path

image_name_list = os.listdir(image_folder)

# 从数据集中随机选取元素进行划分

train_test_per = train_per + test_per
# 数据集总数
image_num = len(image_name_list)

train_num = 0
test_num = 0
val_num = 0
if train_test_per == 1.0:
    # 向上取整
    train_num = math.ceil(image_num * train_per)
    test_num = image_num - train_num
else:
    train_num = math.floor(image_num * train_per)
    test_num = math.floor(image_num * test_per)
    val_num = image_num - (train_num + test_num)

print('train datasets的数量: {}, test datasets的数量: {}, val datasets的数量: {}'.format(train_num, test_num, val_num))
train_name_list = random.sample(image_name_list, train_num)
train_name_set = set(train_name_list)
others_set = set(image_name_list) - train_name_set
test_name_list = random.sample(list(others_set), test_num)
test_name_set = set(test_name_list)
val_name_set = others_set - test_name_set

def mkdir(path):
    '''
    文件夹如果不存在，则创建
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def copy_images_labels(image_name_set, image_folder, label_folder, to_imagepath, to_labelpath):
    '''
    将图片复制到目标文件夹下
    image_name_set: 需要复制的图片名称，set类型
    image_folder: 原图所在目录
    label_folder: 原标签所在目录
    to_imagepath: 图片复制的目标文件夹
    to_labelpath: 标签复制的目标文件夹
    '''
    for i, image_name in enumerate(image_name_set):
        image_path = os.path.join(image_folder, image_name)
        copy(image_path, to_imagepath)
        image_name = image_name.split('.')[:-1]
        if isinstance(image_name, list):
            image_name = ".".join(image_name)
        txt_name = '{}.txt'.format(image_name)

        txt_path = os.path.join(label_folder, txt_name)
        try:
            copy(txt_path, to_labelpath)
        except Exception as e:
            print('复制标签失败:')
            print(e)


ddck_str = args.output
# 训练集目录
train_image_path = os.path.join(ddck_str, 'train', 'images')
train_label_path = os.path.join(ddck_str, 'train', 'labelTxt')
# 测试集目录
test_image_path = os.path.join(ddck_str, 'test', 'images')
test_label_path = os.path.join(ddck_str, 'test', 'labelTxt')
# 验证集目录
val_image_path = os.path.join(ddck_str, 'val', 'images')
val_label_path = os.path.join(ddck_str, 'val', 'labelTxt')


# 把图片复制到训练集目录
if train_name_set:
    mkdir(train_image_path)
    mkdir(train_label_path)
    copy_images_labels(train_name_set, image_folder, label_folder, train_image_path, train_label_path)

# 把图片复制到测试集目录
if test_name_set:
    mkdir(test_image_path)
    mkdir(test_label_path)
    copy_images_labels(test_name_set, image_folder, label_folder, test_image_path, test_label_path)

# 把图片复制到验证集目录
if val_name_set:
    mkdir(val_image_path)
    mkdir(val_label_path)
    copy_images_labels(val_name_set, image_folder, label_folder, val_image_path, val_label_path)