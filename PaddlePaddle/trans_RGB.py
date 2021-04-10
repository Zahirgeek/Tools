# RGB转换灰度图
import argparse
import os
import glob
import platform
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description='把三通道RGB图片转换为单通道图片')
    parser.add_argument('--image_path', '-i', type=str,
                        help='图像输入路径')
    parser.add_argument('--output_path', '-o', type=str,
                        help='转换后的图像输出路径')
    parser.add_argument('--delete_image', '-del', action='store_true', default=False,
                        help='是否删除原图，默认不删除')

    return parser.parse_args()


def add_separator(path):
    '''
    给路径末尾无分隔符的字符串或list添加分隔符
    :param path: 路径
    :return: path
    '''
    if platform.system().lower() == 'windows':
        if not path.endswith('\\'):
            path += '\\'
    else:
        if not path.endswith('/'):
            path += '/'
    return path


def create_folder(path):
    # 路径不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)


def convert_images(args):
    images_path = add_separator(args.image_path)
    output_path = args.output_path
    create_folder(output_path)

    images_list = glob.glob(images_path+'*')
    for i, image_path in enumerate(images_list):
        img = Image.open(image_path)
        if len(img.split()) > 1:
            img = img.convert('L')
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_path, image_name)
        img.save(save_path)
        if args.delete_image:
            os.remove(image_path)


if __name__ == '__main__':
    args = get_args()
    convert_images(args)