'''
Generate train.txt/val.txt/test.txt files under ./data/my_data/ directory. One line for one image, in the format like
image_index image_absolute_path img_width img_height box_1 box_2 ... box_n.
Box_x format: label_index x_min y_min x_max y_max.
(The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
image_index is the line index which starts from zero. label_index is in range [0, class_num - 1].

For example:

0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
...
'''
import argparse
import os
import fnmatch
import xml.dom.minidom as minidom


def read_xml(xml_path: str, coco_dict: dict):
    '''
    处理xml文件，返回label信息
    :param xml_path: 单个xml文件位置
    :param coco_dict:
    :return: xml_info: width, height, boxes_x
    '''
    # 打开xml文档
    dom = minidom.parse(xml_path)
    # 得到文档元素对象
    root = dom.documentElement
    size = root.getElementsByTagName("size")[0]
    width = size.getElementsByTagName("width")[0]
    width = width.childNodes[0].data

    height = size.getElementsByTagName("height")[0]
    height = height.childNodes[0].data

    objects = root.getElementsByTagName("object")
    box_x_list: list = list()
    for object in objects:
        name = object.getElementsByTagName("name")[0]
        name = name.childNodes[0].data
        assert coco_dict.get(name) != None, '请检查coco.names文件中有没有"{}"这个label'.format(name)
        label_index = coco_dict.get(name)

        xmin = object.getElementsByTagName("xmin")[0]
        xmin = xmin.childNodes[0].data
        ymin = object.getElementsByTagName("ymin")[0]
        ymin = ymin.childNodes[0].data
        xmax = object.getElementsByTagName("xmax")[0]
        xmax = xmax.childNodes[0].data
        ymax = object.getElementsByTagName("ymax")[0]
        ymax = ymax.childNodes[0].data
        box_x = '{} {} {} {} {}'.format(label_index, xmin, ymin, xmax, ymax)
        box_x_list.append(box_x)

    if box_x_list == []:
        return None
    else:
        boxes_x = ' '.join(box_x_list)

    xml_info = '{} {} {}'.format(width, height, boxes_x)
    return xml_info


def process_folders(args):
    '''
    处理用户输入的目录和文件
    :param args:
    :return:
    '''
    xmls_path: str = os.path.expanduser(args.xml_path)
    images_path: str = os.path.expanduser(args.image_path)
    assert os.path.exists(images_path), '输入的图片路径不存在'
    assert os.path.exists(xmls_path), '输入的xml路径不存在'
    assert os.path.exists(args.coco_names_path), '输入的coco.names文件不存在'

    txt_name = os.path.basename(args.txt_save_path)
    assert txt_name.split('.')[-1] == 'txt', '请输入正确的txt路径，包含文件名'

    images_list: list = os.listdir(args.image_path)
    assert images_list is not [], '输入的图片路径下没有图片'

    images_set: set = set(images_list)

    coco_dict: dict = dict()
    with open(args.coco_names_path, 'r') as coco_file:
        coco_list = coco_file.readlines()
        for i, coco in enumerate(coco_list):
            category = coco.split('\n')[0]
            coco_dict.update({category: i})

    # txt中的文件index
    num = 0
    # txt每行内容
    txt_lines: list = []
    for root, dirs, files in os.walk(xmls_path):
        for file in files:
            # 文件名
            file_name = file.split('.')[0]
            for image in images_set:
                # 遍历images_set，查找xml文件名是否有对应的图片，没有则跳过读取
                if fnmatch.fnmatch(image, file_name+'.*'):
                    xml_path = os.path.join(root, file)
                    xml_info = read_xml(xml_path, coco_dict)
                    if xml_info is None:
                        continue
                    image_path = os.path.join(args.image_path, image)
                    image_abs_path = os.path.abspath(image_path)
                    txt_line = '{} {} {}'.format(num, image_abs_path, xml_info)
                    num += 1
                    txt_lines.append(txt_line)

    with open(args.txt_save_path, 'w', encoding='utf-8') as file:
        for line in txt_lines:
            file.write(line+'\n')


def get_args():
    parser = argparse.ArgumentParser(description="VOC标签信息存入txt文件，格式详见脚本文件注释，注意将coco.names文件中添加相应label。")
    parser.add_argument("--xml_path", '-x', type=str,
                        help="VOC标注xml文件所在位置")
    parser.add_argument("--image_path", '-i', type=str,
                        help="图片位置")
    parser.add_argument("--txt_save_path", '-s', type=str, default='./info.txt',
                        help='txt文件保存位置，默认保存在当前文件夹')
    parser.add_argument("--coco_names_path", '-c', type=str,
                        help='coco.names文件所在位置')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    process_folders(args)
    print('finished')
