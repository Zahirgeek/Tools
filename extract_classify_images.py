# 读取CVAT for images 1.1格式导出的xml的标注文件，提取里面的tag内容，将图片按照tag分类到对应文件夹。将需要解析的xml文件和其对应的images文件夹放至同一目录
import argparse
import os
import shutil
from lxml import etree


def get_args():
    parser = argparse.ArgumentParser(description="读取CVAT for images 1.1格式导出的xml的标注文件，提取里面的tag内容，将图片按照tag分类到对应文件夹。将需要解析的xml文件和其对应的images文件夹放至同一目录")
    parser.add_argument("--xml_path", "-x", type=str, default="annotations.xml",
                        help="xml文件位置")
    parser.add_argument("--output_path", "-o", type=str, default="output",
                        help="图片分类后保存的位置")

    args = parser.parse_args()
    return args

def path_exist(path):
    if not os.path.exists(path):
        print("{}路径不存在，正在创建".format(path))
        os.makedirs(path)
        print("{}路径创建完成".format(path))

def parsexmlfile(path, output_path):
    '''
    解析xml文件
    :param
    path:xml文件位置
    output_path:输出文件位置
    :return:
    '''
    assert os.path.exists(path), "用户输入的xml路径不存在，请检查后重新输入"
    abs_path = os.path.dirname(os.path.abspath(path))

    tree = etree.parse(path)
    # 标签名称，list
    labels = list()
    label_elem = tree.xpath("//labels/label")
    for i, l in enumerate(label_elem):
        if l.xpath("./type/text()")[0] == "tag":
            labels.extend(l.xpath("./name/text()"))

    # <image>，list
    images = tree.xpath("//image")

    # 保存label名称到指定位置
    path_exist(output_path)
    with open(os.path.join(output_path, "obj.names"), "w") as f:
        for i in labels:
            f.write(i + '\n')

    # 创建保存图片的目录
    output_image_path_list = list()
    for i, label in enumerate(labels):
        output_image_path = os.path.join(output_path, str(i))
        output_image_path_list.append(output_image_path)
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)
    # 创建一个无标签目录
    output_nolabel_path = os.path.join(output_path, "nolabel")
    if not os.path.exists(output_nolabel_path):
        os.makedirs(output_nolabel_path)
    output_image_path_list.append(output_nolabel_path)

    # 复制图片到指定的分类目录
    for i, image in enumerate(images):
        image_name = image.get("name")
        # 图片位置
        image_abs_path = os.path.join(abs_path, "images", image_name)
        tag = image.xpath("./tag/@label")
        try:
            label = tag[0]
        except Exception as e:
            print("{}没有标签".format(image_abs_path))
            shutil.copy(image_abs_path, output_nolabel_path)
        else:
            label_index = labels.index(label)
            target_path = os.path.join(output_path, str(label_index))
            shutil.copy(image_abs_path, target_path)

        #进度条
        progress_bar = i / (len(images) - 1) * 100
        print("\r{:.1f}%".format(progress_bar), end="")

    #分类文件夹为空，则删除
    for i, output_image in enumerate(output_image_path_list):
        if not os.listdir(output_image):
            try:
                os.rmdir(output_image)
            except Exception as e:
                print("文件夹删除失败,错误代码:{}".format(e))


if __name__ == '__main__':
    args = get_args()
    xml_path = args.xml_path
    output_path = args.output_path
    parsexmlfile(xml_path, output_path)