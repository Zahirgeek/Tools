# 分割数据集
import os
import argparse
import random
import math
from shutil import copy

def get_args():
    parser = argparse.ArgumentParser(description="分割Yolov5数据集")
    parser.add_argument("--train", type=float,
                        help="训练集比例")
    parser.add_argument("--val", type=float,
                        help="验证集比例")
    parser.add_argument("--image_path", '-i', type=str,
                        help="图片位置")
    parser.add_argument("--label_path", '-l', type=str,
                        help="标注文件位置")
    parser.add_argument("--output", '-o', type=str, default="newname",
                        help="分割后的数据输出位置，不指定则默认为该脚本同级newname目录下")

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

def copy_images_labels(image_name_set, image_folder, label_folder, images_output_path, labels_output_path):
    '''
    将图片和标签分别复制到目标文件夹下
    image_name_set: 需要复制的图片名称，set类型
    image_folder: 原图所在目录
    label_folder: 原标签所在目录
    images_output_path: 图片复制的目标文件夹
    labels_output_path: 标签复制的目标文件夹
    '''
    for i, image_name in enumerate(image_name_set):
        # 复制图片
        image_path = os.path.join(image_folder, image_name)
        copy(image_path, images_output_path)
        
        # 复制对应的标签文件
        image_name_without_ext = image_name.split('.')[:-1]
        if isinstance(image_name_without_ext, list):
            image_name_without_ext = ".".join(image_name_without_ext)
        txt_name = '{}.txt'.format(image_name_without_ext)

        txt_path = os.path.join(label_folder, txt_name)
        try:
            copy(txt_path, labels_output_path)
        except Exception as e:
            print('复制标签失败: {}'.format(txt_name))
            print(e)


newname_str = args.output
newname_path = os.path.split(newname_str)[0]
newname_basename = os.path.split(newname_str)[1]
newname_count = 0
while(os.path.exists(newname_str)):
    newname_count += 1
    newname_basename = newname_basename + str(newname_count)
    newname_str = os.path.join(newname_path, newname_basename)

# 创建Ultralytics标准目录结构
# 图片目录
images_path = os.path.join(newname_str, 'images')
train_images_path = os.path.join(images_path, 'train')
val_images_path = os.path.join(images_path, 'val')
test_images_path = os.path.join(images_path, 'test')

# 标签目录
labels_path = os.path.join(newname_str, 'labels')
train_labels_path = os.path.join(labels_path, 'train')
val_labels_path = os.path.join(labels_path, 'val')
test_labels_path = os.path.join(labels_path, 'test')

# 创建目录结构
mkdir(train_images_path)
mkdir(val_images_path)
mkdir(test_images_path)
mkdir(train_labels_path)
mkdir(val_labels_path)
mkdir(test_labels_path)

# 把图片和标签复制到训练集目录
if train_name_set:
    copy_images_labels(train_name_set, image_folder, label_folder, train_images_path, train_labels_path)

# 把图片和标签复制到测试集目录
if test_name_set:
    copy_images_labels(test_name_set, image_folder, label_folder, test_images_path, test_labels_path)

# 把图片和标签复制到验证集目录
if val_name_set:
    copy_images_labels(val_name_set, image_folder, label_folder, val_images_path, val_labels_path)

print(f'数据集分割完成！输出目录: {newname_str}')
print(f'目录结构:')
print(f'  {newname_str}/')
print(f'  ├── images/')
print(f'  │   ├── train/ ({len(train_name_set)} 张图片)')
print(f'  │   ├── val/ ({len(val_name_set)} 张图片)')
print(f'  │   └── test/ ({len(test_name_set)} 张图片)')
print(f'  └── labels/')
print(f'      ├── train/ ({len(train_name_set)} 个标签)')
print(f'      ├── val/ ({len(val_name_set)} 个标签)')
print(f'      └── test/ ({len(test_name_set)} 个标签)')