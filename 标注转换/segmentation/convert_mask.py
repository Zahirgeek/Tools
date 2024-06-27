# 图片转换脚本，将彩色的segment mask图片转换为二值化图片
import os
import argparse
from PIL import Image

def convert_image(input_path, output_path):
    # 判断输入路径是否为文件夹
    if os.path.isdir(input_path):
        # 读取文件夹下的所有图片文件
        image_files = [f for f in os.listdir(input_path) if f.endswith('.png') or f.endswith('.jpg')]
        for image_file in image_files:
            image_path = os.path.join(input_path, image_file)
            convert_single_image(image_path, output_path)
    else:
        # 读取单个图片文件
        convert_single_image(input_path, output_path)

def convert_single_image(input_path, output_path):
    # 打开图片文件
    image = Image.open(input_path)

    # 将图片转换为二值化图片
    image = image.convert('L')
    threshold = 1
    image = image.point(lambda p: p > threshold and 255)
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    # 获取输出文件路径
    output_file = os.path.join(output_path, os.path.basename(input_path))
    # 保存转换后的图片
    image.save(output_file)

# 创建解析器
parser = argparse.ArgumentParser(description='图片转换脚本，将彩色的segment mask图片转换为二值化图片')
# 添加参数
parser.add_argument('--input', '-i', type=str, help='图片文件夹位置或图片文件路径')
parser.add_argument('--output', '-o', type=str, help='结果输出位置')

args = parser.parse_args()

# 转换图片
convert_image(args.input, args.output)
