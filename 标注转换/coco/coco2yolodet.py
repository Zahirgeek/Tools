# 将coco segmentation mask格式标注的文件转换为yolo目标识别格式,需要安装opencv-python、tqdm
import json
import os
import shutil
import argparse
import cv2
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="将coco segmentation mask格式标注的文件转换为yolo目标识别格式,需要安装opencv-python、tqdm")
    parser.add_argument("--img_dir", "-i", type=str, required=True,
                        help="图片文件位置")
    parser.add_argument("--json_path", "-j", type=str, required=True,
                        help="coco格式的json文件位置")
    parser.add_argument("--output_path", "-o", type=str, default="output",
                        help="保存转换后的文件位置")
    parser.add_argument("--show", "-s", action='store_true',
                        help="如果要保存带框的图，打开此选项")


    args = parser.parse_args()
    return args

def coco_to_yolo(coco_file, output_dir, images_dir, is_show=False):
    # 加载COCO格式的JSON文件
    with open(coco_file, 'r', encoding='UTF-8') as f:
        coco_data = json.load(f)

    # 创建类别ID到名称的映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    coco_bar = tqdm(coco_data['images'])
    # 处理每个图像
    for j, image in enumerate(coco_bar):
        img_id = image['id']
        img_name = image['file_name']
        img_width = image['width']
        img_height = image['height']

        # 加载原图
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)

        # 找到与当前图像相关的所有标注
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        # 创建YOLO格式的标注
        yolo_annotations = []
        for ann in annotations:
            cat_id = ann['category_id']
            bbox = ann['bbox']  # [x, y, width, height]

            # 转换为YOLO格式 [class, x_center, y_center, width, height]
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height

            yolo_ann = f"{cat_id-1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_ann)

            # 在原图上画框
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(img, categories[cat_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(img, f"{cat_id-1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存YOLO格式的标注文件
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, 'annotations', txt_name)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        # 保存带有边界框的图像
        if is_show and yolo_annotations:
            vis_img_name = 'vis_' + img_name
            vis_img_path = os.path.join(output_dir, 'det', vis_img_name)
            cv2.imwrite(vis_img_path, img)

        # 拷贝原图
        img_cp_dir = os.path.join(output_dir, 'images', img_name)
        shutil.copy(img_path, img_cp_dir)


    # 保存类别映射文件
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cat_id, cat_name in categories.items():
            f.write(f"{cat_name}\n")

    print(f"结果已保存至{os.path.abspath(output_dir)}")

if __name__ == "__main__":
    args = get_args()
    # 使用示例
    # coco_file = 'instances_default1.json'
    # output_dir = 'output1'
    # images_dir = 'img'

    coco_file = args.json_path
    output_dir = args.output_path
    images_dir = args.img_dir
    assert os.path.exists(images_dir), "输入的图片文件位置不存在，请重试"
    assert os.path.exists(coco_file), "输入的json文件位置不存在，请重试"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    if args.show is True:
        os.makedirs(os.path.join(output_dir, "det"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    coco_to_yolo(coco_file, output_dir, images_dir, is_show=args.show)