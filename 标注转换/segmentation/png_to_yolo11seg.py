"""
将png格式的分割标注转换为yolo seg格式
"""
import cv2
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert mask PNG (class-id image) to YOLO11 segmentation TXT"
    )

    parser.add_argument(
        "--images", type=str, required=True,
        help="Path to images directory"
    )

    parser.add_argument(
        "--masks", type=str, required=True,
        help="Path to masks directory (PNG with class IDs)"
    )

    parser.add_argument(
        "--labels", type=str, required=True,
        help="Output path for YOLO segmentation label TXT files"
    )

    return parser.parse_args()


def mask_to_yolo_seg(mask_path, img_path, save_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠ 跳过：找不到原图 {img_path}")
        return

    h, w = mask.shape[:2]

    classes = np.unique(mask)
    classes = classes[classes != 0]  # 0 为背景

    with open(save_path, "w") as f:
        for cls in classes:
            obj = (mask == cls).astype(np.uint8)

            contours, _ = cv2.findContours(
                obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if len(cnt) < 3:
                    continue

                cnt = cnt.squeeze()
                poly = []
                for x, y in cnt:
                    poly.append(x / w)
                    poly.append(y / h)

                line = f"{cls} " + " ".join(f"{p:.6f}" for p in poly)
                f.write(line + "\n")


def main():
    args = parse_args()

    os.makedirs(args.labels, exist_ok=True)

    print("=== Mask → YOLO11 Segmentation 开始转换 ===")

    for name in os.listdir(args.masks):
        if not name.lower().endswith(".png"):
            continue

        mask_path = os.path.join(args.masks, name)
        img_path = os.path.join(args.images, name.replace(".png", ".jpg"))
        save_path = os.path.join(args.labels, name.replace(".png", ".txt"))

        mask_to_yolo_seg(mask_path, img_path, save_path)

    print("🎉 转换完成！YOLO11 分割标签已生成。")


if __name__ == "__main__":
    main()
