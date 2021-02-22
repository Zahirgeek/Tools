# 将output文件夹中的图片分割到多个文件夹中，每个文件夹中包含200个图片
from glob import glob
from shutil import move
import os

# 图片所在文件夹
path = os.path.join('output')
imgs_path = glob(path+'/*')
imgs_path

num = 0
for img in imgs_path:
    if num % 200 == 0:
        # 图片所在文件夹
        to_path = os.path.join(path, str(num))
        if not os.path.exists(to_path):
            os.makedirs(to_path)
    move(img, to_path)
    num += 1