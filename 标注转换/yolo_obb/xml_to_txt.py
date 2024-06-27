from xml.dom.minidom import parse
import xml.dom.minidom
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='把CVAT for images 1.1格式的标注文件转换为yolo obb格式')

    parser.add_argument('--annotations_path', '-a', type=str,
                        help='CVAT for images 1.1标注文件路径')
    parser.add_argument('--output_path', '-o', type=str, default='labels',
                        help='yolo obb label输出目录，默认是当前文件夹中的labels子文件夹')

    args = parser.parse_args()
    return args


args = get_args()

# 标注文件的位置
annotations_path = args.annotations_path
# txt保存路径
output_path = args.output_path


def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

createFolder(output_path)

# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse(annotations_path)
collection = DOMTree.documentElement
images = collection.getElementsByTagName("image")

for i, image in enumerate(images):
    print("---name---:", image.getAttribute("name"))
    name = '{}.txt'.format(image.getAttribute("name").split('/')[-1].split('.')[0])

    # 写入文件
    # with open(os.path.join(output_path, name), 'w') as f_out:
    polygons = image.getElementsByTagName("polygon")
    lines_list = list()
    for j, polygon in enumerate(polygons):
        # 获取points属性
        points = polygon.getAttribute("points")

        if not points:
            continue

        label = polygon.getAttribute("label")
        difficult = '0'

        p_list = list()
        # 处理坐标点

        for point in points.split(';'):
            x, y = point.split(',')
            x, y = float(x), float(y)
            p_list.append(x)
            p_list.append(y)
        outline = ' '.join(list(map(str, p_list))) + ' ' + str(label )+ ' ' + difficult
        lines_list.append(outline)
    if lines_list:
        write_lines = '\n'.join(lines_list)
        # print("output path: ", output_path)
        # print("name: ", name)
        with open(os.path.join(output_path, name), 'w') as f_out:
            f_out.write(write_lines)  # 写入txt文件中