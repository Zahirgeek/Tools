"""
Crop objects from images using YOLO format annotations
"""
import os
import argparse
import cv2

# Define argument parser
parser = argparse.ArgumentParser(description='Crop objects from images using YOLO format annotations')
parser.add_argument('--image_dir', '-i', type=str, required=True, help='Directory containing images')
parser.add_argument('--label_dir', '-l', type=str, required=True, help='Directory containing YOLO format annotation files')
parser.add_argument('--output_dir', '-o', type=str, required=True, help='Directory to save cropped objects')
# 开关参数，是否需要padding
parser.add_argument('--padding', '-p', action='store_true', help='Whether to pad images to square')

args = parser.parse_args()

# Loop through images and corresponding labels
for img_file in os.listdir(args.image_dir):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(args.image_dir, img_file)
        label_path = os.path.join(args.label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        if not os.path.exists(label_path):
            continue

        # Load image and label file
        img = cv2.imread(img_path)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        count = 0
        # Loop through labels and crop objects
        for line in lines:
            # x, y, width, height, _ = map(float, line.strip().split())
            # x_min = int(x)
            # y_min = int(y)
            # x_max = int(x + width)
            # y_max = int(y + height)

            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = int((x_center - width / 2) * img.shape[1])
            y_min = int((y_center - height / 2) * img.shape[0])
            x_max = int((x_center + width / 2) * img.shape[1])
            y_max = int((y_center + height / 2) * img.shape[0])

            obj_img = img[y_min:y_max, x_min:x_max]

            # 如果需要padding
            if args.padding:
                # Pad image to square
                max_side = max(obj_img.shape[0], obj_img.shape[1])
                pad_top = (max_side - obj_img.shape[0]) // 2
                pad_bottom = max_side - obj_img.shape[0] - pad_top
                pad_left = (max_side - obj_img.shape[1]) // 2
                pad_right = max_side - obj_img.shape[1] - pad_left
                obj_img = cv2.copyMakeBorder(obj_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))

            #如果args.output_dir不存在，则创建
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)

            obj_file = os.path.join(args.output_dir,
                                    f'{img_file.replace(".jpg", "").replace(".png", "")}_{count}.jpg')
            cv2.imwrite(obj_file, obj_img)

            count += 1
