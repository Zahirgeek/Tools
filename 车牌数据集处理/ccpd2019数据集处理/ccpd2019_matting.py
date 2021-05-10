import os
from shutil import copy, move
from PIL import Image, ImageDraw
import glob
import platform
import cv2
import argparse

parser = argparse.ArgumentParser(description="把ccpd2019中的车牌图片选择并抠出，保存到其他目录中")
parser.add_argument("input_files", type=str,
                    help="The path of the input files.")
parser.add_argument("--output_files", type=str, default="",
                    help="The path of the output files,"
                         "If you do not fill in this item, the default is the same as the input_files path.")

args = parser.parse_args()


def is_dir_exists(path):
    # 判断目录是否存在，不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)

input_dir = os.path.join(args.input_files)
output_dir = os.path.join(args.output_files)
is_dir_exists(output_dir)
# files_list = os.listdir(input_dir)
files_list = glob.glob(input_dir + '/*')


def get_filename(path):
# 输入一个路径获取文件名,包含后缀
    if platform.system().lower() == 'windows':
        filename = path.split('\\')[-1]
    else:
        filename = path.split('/')[-1]
        
    return filename


for file in files_list:
    filename = get_filename(file)
    filename_list = filename.split('-')
    img_list3 = filename_list[2].split('_')
    xmin = int(img_list3[0].split('&')[0])
    ymax = int(img_list3[1].split('&')[1])
    xmax = int(img_list3[1].split('&')[0])
    ymin = int(img_list3[0].split('&')[1])
    
    
    img = cv2.imread(file)
    crop_image = img[ymin: ymax, xmin: xmax]
#     cv2.imshow('crop', crop_image)
#     cv2.waitKey(0)
    save_path = os.path.join(output_dir, filename)
    
    # reshape[94, 24]
    crop_image = cv2.resize(crop_image,(94,24),interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, crop_image)