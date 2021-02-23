# 将一个文件夹中的文件分割到多个文件夹中，每个文件夹中包含n个文件
from glob import glob
from shutil import move, copy
import os
import argparse

parser = argparse.ArgumentParser(description="将一个文件夹中的文件分割到多个文件夹中，每个文件夹中包含n个文件")
parser.add_argument("input_files", type=str,
                    help="The path of the input files.")
parser.add_argument("--output_files", type=str, default="",
                    help="The path of the output files,"
                         "If you do not fill in this item, the default is the same as the input_files path.")
parser.add_argument("--num", type=int, default=200,
                    help="Number of files contained in each folder."
                    )
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")

args = parser.parse_args()

# 图片所在文件夹
path = os.path.join(args.input_files)
imgs_path = glob(path+'/*')

num = 0
if args.type == 'c' or args.type == 'copy':
    for img in imgs_path:
        if num % args.num == 0:
            if not args.output_files:
                to_path = os.path.join(path, str(num+1))
            else:
                # 图片所在文件夹
                to_path = os.path.join(args.output_files, str(num))
            if not os.path.exists(to_path):
                os.makedirs(to_path)
        copy(img, to_path)
        num += 1
elif args.type == 'm' or args.type == 'move':
    for img in imgs_path:
        if num % args.num == 0:
            if not args.output_files:
                to_path = os.path.join(path, str(num+1))
            else:
                # 图片所在文件夹
                to_path = os.path.join(args.output_files, str(num))
            if not os.path.exists(to_path):
                os.makedirs(to_path)
        move(img, to_path)
        num += 1