import cv2
import numpy as np

import os
import argparse


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def rename_file(file):
    '''
    利用CCPD文件名获取车牌号码
    :param file: 文件名
    :return: 车牌号码
    '''
    filename_ = file.split('-')[4]
    filename_list = filename_.split('_')
    num_character = 0
    carid_list = list()
    for filename_split in filename_list:
        num_character += 1
        num = int(filename_split)
        if num_character == 1:
            carid_list.append(provinces[num])
        elif num_character == 2:
            carid_list.append(alphabets[num])
        else:
            carid_list.append(ads[num])
    name_renamed = ''.join(carid_list)
    return name_renamed


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_location_nd(right_bottom, left_bottom, left_top, right_top):
    '''
    输入位置，例如154&383，返回位置的ndarray
    :param location:位置，str类型
    :return:位置，list类型
    '''
    right_bottom_list = right_bottom.split('&')
    left_bottom_list = left_bottom.split('&')
    left_top_list = left_top.split('&')
    right_top_list = right_top.split('&')
    return np.float32([left_bottom_list, right_bottom_list, left_top_list, right_top_list])


def get_carid_location(filename):
    '''
    获取车牌在图中的4个点坐标，返回位置的list
    :param filename: CCPD2019文件名
    :return: np.float32([left_bottom_list, right_bottom_list, left_top_list, right_top_list])
    '''
    filename_list = filename.split('-')
    locations = filename_list[3]
    # right_bottom, left_bottom, left_top, right_top = locations.split('_')
    right_top, left_top, left_bottom, right_bottom = locations.split('_')
    points1 = get_location_nd(right_bottom, left_bottom, left_top, right_top)
    return points1


def save_carid(image_path, carid_save_path, carid, filename):
    '''
    保存车牌图片
    :param image_path: 车辆图片位置
    :param carid_save_path: 保存位置
    :param carid: 车牌号
    :param filename: CCPD2019文件名
    :return:
    '''
    img = cv2.imread(image_path)
    suffix = '.jpg'
    image_name = '{}{}'.format(carid, suffix)
    save_path = os.path.join(carid_save_path, image_name)
    create_folder(carid_save_path)

    points1 = get_carid_location(filename)
    points2 = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])
    M = cv2.getPerspectiveTransform(points1, points2)
    dst = cv2.warpPerspective(img, M, (94, 24))
    cv2.imencode(suffix, dst)[1].tofile(save_path)


def read_files(args):
    ccpd_folder = args.input_files
    create_folder(args.output_files)

    for root, dirs, files in os.walk(ccpd_folder, topdown=False):
        # 遍历ccpd_folder文件夹下的所有图片，分析文件名，做车牌抠图和矫正操作
        if not files:
            continue

        for image_name in files:
            image_path = os.path.join(root, image_name)
            carid = rename_file(image_name)
            save_carid(image_path=image_path, carid_save_path=args.output_files, carid=carid, filename=image_name)


def get_args():
    parser = argparse.ArgumentParser(description="把ccpd2019中的车牌图片选择并抠出，重命名车牌并进行矫正")
    parser.add_argument("--input_files", '-i', type=str, default='./ccpd_test',
                        help="The path of the input files.")
    parser.add_argument("--output_files", '-o', type=str, default='./ccpd_out',
                        help="The path of the output files")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    read_files(args)
    print('finish')