# 将一个文件夹中所有子目录的文件复制/剪切到根目录，文件重名不覆盖，使用“文件名-数字”重命名文件
import os
import shutil
from shutil import move, copy
import argparse
import platform
import numpy as np

parser = argparse.ArgumentParser(description="将一个文件夹中所有子目录的文件复制/剪切到根目录，文件重名不覆盖，使用“文件名-数字”重命名文件")
parser.add_argument("input_path", type=str,
                    help="选择要处理的文件夹")
parser.add_argument("--error_path", type=str,default="",
                    help="选择错误输出的文件夹，默认为input文件夹")
parser.add_argument("--output_path", type=str,default="",
                    help="选择输出的文件夹，默认为input文件夹")
parser.add_argument("--type", type=str, default="c",
                    help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")
parser.add_argument("--remove", "-r", action="store_false", default=True,
                    help="是否删除空文件夹，默认删除")

args = parser.parse_args()

# 要遍历的文件夹
input_path = args.input_path
output_path = args.output_path
# input_path = os.path.join('selected')
if not args.error_path:
    error_path = os.path.join(input_path, 'error')
else:
    error_path = os.path.join(args.error_path)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

if output_path:
    try:
        mkdirs(output_path)
    except Exception as e:
        print('创建输出文件夹失败，请检查地址是否输入正确')

if output_path == '':
    exist_list = os.listdir(input_path)
else:
    exist_list = os.listdir(output_path)
exists_file_np = np.array(exist_list)

def single_file_copypath(from_file, to_path, exists_file_np):
    '''
    将单个文件不重复地复制到另一个文件夹中，返回复制的目标文件夹目录和文件已存在的ndarray
    from_file: 要复制的文件名路径，路径+文件名
    to_path: 复制的目标文件夹
    '''
    if platform.system().lower() == 'windows':
        file = from_file.split('\\')[-1]
    else:
        file = from_file.split('/')[-1]
    # print(file)
    if not file in exists_file_np:
        to_file = os.path.join(to_path, file)
        exists_file_np = np.append(exists_file_np, file)
        return to_file, exists_file_np
    else:
        num = 0
        while(file in exists_file_np):
            filename = file.split('.')[0]
            if len(filename.split('-')) <= 2:
                filename = filename.split('-')[0]
            shuffix = file.split('.')[-1]
            num += 1
            file = '{}-{}.{}'.format(filename, num, shuffix)
            to_file = os.path.join(to_path, file)
        # print('a', exists_file_np)
        exists_file_np = np.append(exists_file_np, file)
        return to_file, exists_file_np


root_dirs = []
for root, dirs, files in os.walk(input_path, topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        if root == input_path and output_path == '':
            continue
        if not output_path == '':
            input_path = output_path
        # print(file_path)
        try:
            to_file, exists_file_np = single_file_copypath(file_path, input_path, exists_file_np)
            if args.type == 'c' or args.type == 'copy':
                # copy(file_path, input_path)
                copy(file_path, to_file)
                #print('正在复制{}'.format(os.path.join(root, name)))
            elif args.type == 'm' or args.type == 'move':
                # move(file_path, input_path)
                move(file_path, to_file)
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
    if root == input_path:
        root_dirs = dirs

if args.remove:
    for folder in root_dirs:
        remove_folder = os.path.join(input_path, folder)
        try:
            shutil.rmtree(remove_folder, ignore_errors=True)
        except Exception as e:
            print(e)