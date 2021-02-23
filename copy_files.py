# 将一个文件夹下的二级目录所有文件复制/剪切到另一个文件夹中
import os
from shutil import copy, move
import argparse

parser = argparse.ArgumentParser(description="将一个文件夹下的二级目录所有文件复制/剪切到另一个文件夹中")
parser.add_argument("input_files", type=str,
                    help="The path of the input files.")
parser.add_argument("--output_files", type=str,default="",
                    help="The path of the output files,"
                         "If you do not fill in this item, the default is the same as the input_files path.")
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")

args = parser.parse_args()

# 文件所在位置
img_path = os.path.join(args.input_files)
dir_list = os.listdir(img_path)

# 目标文件夹
if not args.output_files:
    to_path = img_path
else:
    to_path = os.path.join(args.output_files)
    if not os.path.exists(to_path):
        os.makedirs(to_path)

for dirs in dir_list:
    dir_path = os.path.join(img_path, dirs)
    files_list = os.listdir(dir_path)
    if args.type == 'c' or args.type == 'copy':
        for file in files_list:
            from_path = os.path.join(img_path, dirs, file)
            copy(from_path, to_path)
    elif args.type == 'm' or args.type == 'move':
        for file in files_list:
            from_path = os.path.join(img_path, dirs, file)
            move(from_path, to_path)