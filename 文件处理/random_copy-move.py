# 随机从一个文件夹中选取n个文件复制/剪切到另一个文件夹中
import os
import glob
from shutil import copy, move
import argparse
import random
import sys

parser = argparse.ArgumentParser(description="随机从一个文件夹中选取n个文件复制/剪切到另一个文件夹中")
parser.add_argument("input_files", type=str,
                    help="The path of the input files.")
parser.add_argument("--output_files", type=str,
                    help="The path of the output files.")
parser.add_argument("--num", type=float, default=0,
                    help="The number of files you want to copy")
parser.add_argument("--arithmetic", default=0,
                    help="算式，支持乘法*和除法/")
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy")

args = parser.parse_args()

input_path = os.path.join(args.input_files)
output_path = os.path.join(args.output_files)
num = int(args.num)
arithmetic = args.arithmetic

if arithmetic != 0 and num == 0:
    num_division = arithmetic.split('/')
    num_multiplication = arithmetic.split('*')
    if len(num_division) == 2:
        num = float(num_division[0]) / float(num_division[1])
    elif len(num_multiplication) == 2:
        num = float(num_multiplication[0]) * float(num_multiplication[1])
    else:
        print('arithmetic参数输入错误，请重新输入')
        sys.exit(1)
    num = int(num)
    
type_ = args.type
if num == 0:
    print('未选中任何文件，请检查输入数量是否有误')
    sys.exit(1)

try:
    files_list = glob.glob(input_path+'/*')
except Exception as e:
    print('输入文件夹错误，请检查输入文件夹中是否有文件，或不存在该文件夹，错误代码: ', e)
    sys.exit(1)
else:
    try:
        selected = random.sample(files_list, num)
    except Exception as e:
        print('输入的文件数量超出了文件夹中的文件数量，请重新输入')
        sys.exit(1)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if type_ == 'c' or type_ == 'copy':
        for file in selected:
            copy(file, output_path)
    elif type_ == 'm' or type_ == 'move':
        for file in selected:
            move(file, output_path)
