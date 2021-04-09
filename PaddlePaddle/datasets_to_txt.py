# 将数据集生成含有image label的txt文件 例：JPEGImages/N0120.jpg Annotations/N0120.png
import os
import argparse
import platform
import random
import re
import sys


def get_parser():
    parser = argparse.ArgumentParser(description='将数据集生成含有image label的txt文件，用于飞桨模型训练')
    parser.add_argument('--image_path', '-i', type=str,
                        help='图片位置')
    parser.add_argument('--label_path', '-l', type=str,
                        help='标签位置')
    parser.add_argument('--test', '-t', type=float, default=0.1,
                        help='测试集占全部数据集比例，默认是全部数据集的1/10，输入类型为float')
    parser.add_argument('--validation', '-v', type=float, default=0.0,
                        help='验证集占全部数据集比例，默认是0，输入类型为float')

    return parser.parse_args()


def remove_separator(path):
    '''
    给路径末尾有分隔符的字符串删除分隔符
    :param path: 路径
    :return: path
    '''
    if platform.system().lower() == 'windows':
        if path.endswith('\\'):
            path = path[: -1]
    else:
        if path.endswith('/'):
            path = path[: -1]
    return path


def get_image_label(image_path, label_path):
    '''
    根据图片和标记的路径生成含有图片 标记对应信息的list
    :param image_path: 图片路径
    :param label_path: 标记路径
    :return: ["image label"] list
    '''
    image_basename = os.path.basename(remove_separator(image_path))
    label_basename = os.path.basename(remove_separator(label_path))
    images_list = os.listdir(image_path)
    label_list = os.listdir(label_path)

    image_label_list = []
    for i, image in enumerate(images_list):
        image_name = image.split('.')[0]
        # 查看image、label列表中对应元素文件名是否一致
        if image_name == label_list[i].split('.')[0]:
            image_label = '{} {}'.format(os.path.join(image_basename, image), os.path.join(label_basename, label_list[i]))
            image_label_list.append(image_label)
        else:
            # 正则匹配label列表中是否和image文件名匹配
            image_index = list(
                filter(lambda x: re.match(image_name + '\..*', label_list[x]) != None, list(range(len(label_list)))))
            if image_index:
                image_label = '{} {}'.format(os.path.join(image_basename, image), os.path.join(label_basename, label_list[image_index[0]]))
                image_label_list.append(image_label)
            else:
                print('{}未找到对应label'.format(os.path.join(image_path, image)))
                continue

    print('共{}个数据集'.format(len(image_label_list)))
    return image_label_list


def split_train_test_val(args):
    '''
    分割train test val数据集
    :param args: args
    :return:
    '''
    image_path = args.image_path
    label_path = args.label_path
    # 生成包含全部数据集image label的list
    image_label_list = get_image_label(image_path=image_path, label_path=label_path)
    image_label_length = len(image_label_list)

    train_sample_list = []
    test_sample_list = []
    val_sample_list = []
    # 分割train test val数据集
    if args.test > 0:
        test_length = int(args.test * image_label_length)
        test_sample_list = random.sample(image_label_list, test_length)
        train_sample_list = [i for i in image_label_list if i not in test_sample_list]
        if args.validation > 0:
            val_length = int(args.validation * image_label_length)
            val_sample_list = random.sample(image_label_list, val_length)
            train_sample_list = [i for i in train_sample_list if i not in val_sample_list]
        elif args.validation < 0:
            print('validation参数不能小于0，重新输入')
            sys.exit(1)
    elif args.test < 0:
        print('test 参数不能小于0，重新输入')
        sys.exit(1)

    image_path_dirname = remove_separator(args.image_path)
    txt_path_dirname = os.path.dirname(image_path_dirname)
    train_txt_path = os.path.join(txt_path_dirname, 'train_list.txt')
    test_txt_path = os.path.join(txt_path_dirname, 'test_list.txt')
    val_txt_path = os.path.join(txt_path_dirname, 'val_list.txt')

    # 写入文件
    if train_sample_list:
        with open(train_txt_path, 'w') as f:
            train_list = [line+'\n' for line in train_sample_list]
            f.writelines(train_list)
            print('生成{}个训练集'.format(len(train_sample_list)))
    if test_sample_list:
        with open(test_txt_path, 'w') as f:
            test_list = [line+'\n' for line in test_sample_list]
            f.writelines(test_list)
            print('生成{}个测试集'.format(len(test_sample_list)))
    if val_sample_list:
        with open(val_txt_path, 'w') as f:
            val_list = [line+'\n' for line in val_sample_list]
            f.writelines(val_list)
            print('生成{}个验证集'.format(len(val_sample_list)))


if __name__ == '__main__':
    args = get_parser()
    split_train_test_val(args)