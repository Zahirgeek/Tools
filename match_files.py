# 用一个文件夹中的文件匹配另一个文件夹中的同名文件，复制或剪切到第三个目录。可以指定被匹配文件夹中文件的后缀名
import os
import platform
from shutil import move, copy
import argparse

parser = argparse.ArgumentParser(description="用一个文件夹中的文件匹配另一个文件夹中的同名文件，复制或剪切到第三个目录。可以指定被匹配文件夹中文件的后缀名")
parser.add_argument("first_path", type=str,
                    help="需要匹配的文件夹")
parser.add_argument("--second_path", type=str,
                    help="被匹配的文件夹")
parser.add_argument("--output_path", type=str, default="",
                    help="输出的文件夹，如不指定，则默认为该程序所在的文件夹下的cp或mv文件夹内")
parser.add_argument("--suffix", type=str,
                    help="指定被匹配文件夹中要匹配的文件名后缀")
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")

args = parser.parse_args()

from_path = args.first_path
match_path = args.second_path

def split_win_linux(path):
    # 判断运行系统为windows还是linux,将输入的路径的最后一个文件夹返回
    if platform.system().lower() == 'windows':
        split_path = path.split('\\')[-1]
    else:
        split_path = path.split('/')[-1]

    return split_path

if not args.output_path:
    if args.type == 'c' or args.type == 'copy':
        from_path_cp = os.path.join('cp', split_win_linux(from_path))
        match_path_cp = os.path.join('cp', split_win_linux(match_path))
    elif args.type == 'm' or args.type == 'move':
        from_path_cp = os.path.join('mv', split_win_linux(from_path))
        match_path_cp = os.path.join('mv', split_win_linux(match_path))
else:
    from_path_cp = os.path.join(args.output_path, split_win_linux(from_path))
    match_path_cp = os.path.join(args.output_path, split_win_linux(match_path))


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdirs(from_path_cp)
mkdirs(match_path_cp)
from_path_list = os.listdir(from_path)

for from_file in from_path_list:
    suffix = args.suffix
    match_file = from_file.split('.')[0] + suffix
    match_file = os.path.join(match_path, match_file)

    if os.path.exists(match_file):
        from_file_path = os.path.join(from_path, from_file)
        if args.type == 'c' or args.type == 'copy':
            copy(match_file, match_path_cp)
            copy(from_file_path, from_path_cp)
        elif args.type == 'm' or args.type == 'move':
            move(match_file, match_path_cp)
            move(from_file_path, from_path_cp)