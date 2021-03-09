# 将CCPD2019图片的文件名重命名为车牌号
import os
from shutil import copy, move
import argparse

parser = argparse.ArgumentParser(description="将CCPD2019图片的文件名重命名为车牌号")
parser.add_argument("input_files", type=str,
                    help="The path of the input files.")
parser.add_argument("--output_files", type=str, default="",
                    help="The path of the output files,"
                         "If you do not fill in this item, the default is under 'output' path.")
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")

args = parser.parse_args()

input_fold = args.input_files
output_fold = args.output_files

def is_fold_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

if output_fold:
    is_fold_exists(output_fold)
else:
    is_fold_exists(os.path.join(input_fold, 'output'))


# test codes
# input_fold = os.path.join('ccpd_1500')
# output_fold = os.path.join('output')
# is_fold_exists(output_fold)
# test codes


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def rename_file(file):
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

# 输入文件夹下的文件名列表
files_list = os.listdir(input_fold)
# 重命名文件列表
rename_list = list()
for file in files_list:
    if os.path.isdir(file):
        # 目录
        pass
    else:
        # 文件
        shuffix = file.split('.')[-1]
        carid_name = rename_file(file)
        from_path = os.path.join(input_fold, file)
        to_path = os.path.join(output_fold, carid_name+'.'+shuffix)

        if args.type == 'c' or args.type == 'copy':
            copy(from_path, to_path)
        elif args.type == 'm' or args.type == 'move':
            move(from_path, to_path)