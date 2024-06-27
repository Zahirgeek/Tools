# 将coco segmentation polygon格式的标注文件转换为yolo seg格式，需要安装tqdm
import os
import json
import shutil
import argparse
from datetime import datetime
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="将coco segmentation polygon格式的标注文件转换为yolo seg格式，需要安装tqdm")
    parser.add_argument("--img_dir", "-i", type=str, required=True,
                        help="图片文件位置")
    parser.add_argument("--json_path", "-j", type=str, required=True,
                        help="coco格式的json文件位置")
    parser.add_argument("--output_path", "-o", type=str, default="output",
                        help="保存转换后的文件位置")


    args = parser.parse_args()
    return args

def write_yolo_txt_file(txt_file_path, label_seg_x_y_list):
    if not os.path.exists(txt_file_path):
        with open(txt_file_path, "w") as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')
    else:
        with open(txt_file_path, "a") as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')

def coco_to_yolo(in_json_path, img_dir, target_dir):
    # 加载COCO格式的JSON文件
    with open(in_json_path, "r", encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建类别ID到名称的映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    coco_bar = tqdm(coco_data['images'])
    # 处理每个图像
    for j, image in enumerate(coco_bar):
        # coco_bar.set_description_str(f"正在处理第{j+1}张")
        img_id = image['id']
        img_name = image['file_name']
        img_width = image['width']
        img_height = image['height']
        # 找到与当前图像相关的所有标注
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        # 创建YOLO格式的标注
        yolo_annotations = []
        for ann in annotations:
            cat_id = ann['category_id']
            segmentation = ann['segmentation']  # [x, y, width, height]
            seg_x_y_list = [i / img_width if num % 2 == 0 else i / img_height for num, i in
                            enumerate(segmentation[0])]  # 归一化图像分割点信息
            label_seg_x_y_list = seg_x_y_list[:]
            label_seg_x_y_list.insert(0, cat_id-1)  # 图像类别与分割点信息[label,x1,y1,x2,y2,...,xn,yn]
            label_seg_x_y_list = list(map(str, label_seg_x_y_list))
            label_seg_x_y_str = " ".join(label_seg_x_y_list)
            yolo_annotations.append(label_seg_x_y_str)

        # 保存YOLO格式的标注文件
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(target_dir, 'annotations', txt_name)
        # 图片保存位置
        img_cp_dir = os.path.join(target_dir, 'images', img_name)

        # 如果目标地址存在标签文件，则重命名该图片和文件
        if os.path.exists(txt_path):
            # 获取当前时间
            now = datetime.now()
            # 格式化时间字符串，包含毫秒
            time_str_with_millis = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
            txt_name = time_str_with_millis + '.txt'
            txt_path = os.path.join(target_dir, 'annotations', txt_name)
            file_extension = os.path.splitext(img_name)[-1]
            img_cp_dir = os.path.join(target_dir, 'images', f'{time_str_with_millis}{file_extension}')


        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))


        img_path = os.path.join(img_dir, img_name)
        shutil.copy(img_path, img_cp_dir)

    # 保存类别映射文件
    with open(os.path.join(target_dir, 'classes.txt'), 'w') as f:
        for cat_id, cat_name in categories.items():
            f.write(f"{cat_name}\n")

    print(f"结果已保存至{os.path.abspath(target_dir)}")


if __name__ == "__main__":
    args = get_args()

    img_dir = args.img_dir
    in_json_path = args.json_path
    assert os.path.exists(img_dir), "输入的图片文件位置不存在，请重试"
    assert os.path.exists(in_json_path), "输入的json文件位置不存在，请重试"

    target_dir = args.output_path
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "annotations"), exist_ok=True)

    coco_to_yolo(in_json_path, img_dir, target_dir)


