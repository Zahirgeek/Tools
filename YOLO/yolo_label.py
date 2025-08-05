"""
YOLO标签处理工具
使用方法：
查看帮助信息：
python yolo_label.py -h

生成train.txt文件：
python yolo_label.py 1 --folder_path /path/to/images_and_labels

生成空的标签文件：
python yolo_label.py 2 --image_path /path/to/images

可视化标签并保存：
python yolo_label.py 3 --label_path /path/to/labels --image_path /path/to/images --save_path /path/to/save --view_images --class_names person car dog cat
如果不想查看图片，只需要省略 --view_images 参数。
保存的图片会使用 "person", "car", "dog", "cat" 作为类别名称。如果没有提供类别名称，或者类别ID超出提供的名称列表，将使用 "Class X" 作为默认名称。
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# 定义一些颜色（BGR格式）
COLORS = [
    (255, 0, 0),  # 蓝色
    (0, 255, 0),  # 绿色
    (0, 0, 255),  # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 洋红色
    (0, 255, 255),  # 黄色
    (128, 0, 0),  # 深蓝色
    (0, 128, 0),  # 深绿色
    (0, 0, 128),  # 深红色
    (128, 128, 0),  # 橄榄色
]


def generate_train_txt(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "train.txt"), "w") as f:
        for image_file in tqdm(image_files, desc="生成train.txt"):
            f.write(f"data/images/{os.path.basename(folder_path)}/{image_file}\n")
    print(f"train.txt已生成在{output_folder}目录下")


def generate_empty_labels(image_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in image_extensions]

    for image_file in tqdm(image_files, desc="生成空标签文件"):
        label_file = os.path.splitext(os.path.join(image_path, image_file))[0] + ".txt"
        if not os.path.exists(label_file):
            open(label_file, "a").close()
    print(f"空标签文件已生成在{image_path}目录下")


def visualize_labels(label_path, image_path, save_path, view_images, class_names):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in image_extensions]
    os.makedirs(save_path, exist_ok=True)

    for image_file in tqdm(image_files, desc="处理图像"):
        image = cv2.imread(os.path.join(image_path, image_file))
        if image is None:
            print(f"无法读取图像: {image_file}")
            continue

        height, width = image.shape[:2]

        label_file = os.path.splitext(os.path.join(label_path, image_file))[0] + ".txt"
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    left = int((x - w / 2) * width)
                    top = int((y - h / 2) * height)
                    right = int((x + w / 2) * width)
                    bottom = int((y + h / 2) * height)

                    color = COLORS[int(class_id) % len(COLORS)]
                    cv2.rectangle(image, (left, top), (right, bottom), color, 2)

                    class_name = class_names[int(class_id)] if int(class_id) < len(
                        class_names) else f"Class {int(class_id)}"
                    label = f"{class_name}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(image, (left, top - label_height - baseline), (left + label_width, top), color, -1)
                    cv2.putText(image, label, (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        save_file = os.path.join(save_path, image_file)
        cv2.imwrite(save_file, image)

    print(f"标注后的图像已保存在{save_path}目录下")

    if view_images:
        cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
        for image_file in image_files:
            image = cv2.imread(os.path.join(save_path, image_file))
            if image is not None:
                display_image(image)
                key = cv2.waitKey(0)
                if key == 27:  # ESC键
                    break
        cv2.destroyAllWindows()


def display_image(image):
    screen = cv2.getWindowImageRect("Annotated Image")
    screen_width, screen_height = screen[2], screen[3]

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算缩放比例
    scale = min(screen_width / width, screen_height / height)

    # 计算缩放后的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建一个黑色背景
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # 计算图像在背景中的位置
    x_offset = (screen_width - new_width) // 2
    y_offset = (screen_height - new_height) // 2

    # 将缩放后的图像放置在背景中央
    background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # 显示图像
    cv2.imshow("Annotated Image", background)


def main():
    parser = argparse.ArgumentParser(description="YOLO标签处理工具")
    subparsers = parser.add_subparsers(dest="function", help="选择功能")

    # 功能1：生成train.txt
    parser_1 = subparsers.add_parser("1", help="生成train.txt文件")
    parser_1.add_argument("--folder_path", required=True, help="包含图片和标签的文件夹路径")

    # 功能2：生成空标签
    parser_2 = subparsers.add_parser("2", help="生成空的标签文件")
    parser_2.add_argument("--image_path", required=True, help="图像路径")

    # 功能3：可视化标签
    parser_3 = subparsers.add_parser("3", help="可视化标签并保存")
    parser_3.add_argument("--label_path", required=True, help="标签路径")
    parser_3.add_argument("--image_path", required=True, help="图像路径")
    parser_3.add_argument("--save_path", required=True, help="保存标注图像的路径")
    parser_3.add_argument("--view_images", action="store_true", help="是否查看标注后的图像")
    parser_3.add_argument("--class_names", nargs='+', default=[], help="类别名称列表")

    args = parser.parse_args()

    if args.function == "1":
        generate_train_txt(args.folder_path)
    elif args.function == "2":
        generate_empty_labels(args.image_path)
    elif args.function == "3":
        visualize_labels(args.label_path, args.image_path, args.save_path, args.view_images, args.class_names)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()