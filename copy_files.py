# 将一个文件夹下的二级目录所有文件复制到另一个文件夹中
import os
from shutil import copy

# 文件所在位置
img_path = os.path.join('imgs')
dir_list = os.listdir(img_path)

# 目标文件夹
to_path = os.path.join('img')
if not os.path.exists(to_path):
    os.makedirs(to_path)

for dirs in dir_list:
    dir_path = os.path.join(img_path, dirs)
    files_list = os.listdir(dir_path)
    for file in files_list:
        from_path = os.path.join(img_path, dirs, file)
        copy(from_path, to_path)