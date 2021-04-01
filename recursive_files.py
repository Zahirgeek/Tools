# 将一个文件夹中所有子目录的文件复制/剪切到另一个文件夹
import os
from shutil import move, copy
import argparse
import platform

parser = argparse.ArgumentParser(description="将一个文件夹中所有子目录的文件复制/剪切到另一个文件夹")
parser.add_argument("input_path", type=str,
                    help="选择要处理的文件夹")
parser.add_argument("--output_path", type=str,default="",
                    help="选择输出的文件夹，默认为处理的文件夹")
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")

args = parser.parse_args()

# 要遍历的文件夹
input_path = args.input_path
# input_path = os.path.join('selected')
if not args.output_path:
    error_path = os.path.join(input_path, 'error')
else:
    error_path = os.path.join(args.output_path)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

for root, dirs, files in os.walk(input_path, topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        if root == input_path:
            continue
        try:
            if args.type == 'c' or args.type == 'copy':
                copy(file_path, input_path)
                #print('正在复制{}'.format(os.path.join(root, name)))
            elif args.type == 'm' or args.type == 'move':
                move(file_path, input_path)
                #print('正在移动{}'.format(os.path.join(root, name)))
        except Exception as e:
            print('移动/复制{}失败，错误代码:{}'.format(os.path.join(root, name), e))
            if platform.system().lower() == 'windows':
                error_p = os.path.join(error_path, root.split('\\')[-1])
            else:
                error_p = os.path.join(error_path, root.split('/')[-1])
            mkdirs(error_p)
            try:
                if args.type == 'c' or args.type == 'copy':
                    copy(file_path, error_p)
                elif args.type == 'm' or args.type == 'move':
                    move(file_path, error_p)
            except Exception:
                pass
            continue