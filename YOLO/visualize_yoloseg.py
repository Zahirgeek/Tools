import os
import cv2
import numpy as np
import argparse
import subprocess
import platform


def random_color(seed):
    np.random.seed(seed)
    return np.random.randint(0, 255, (3,)).tolist()


def parse_yolo_seg_overlay(img, label_path, colors):
    h, w = img.shape[:2]
    overlay = img.copy()

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        poly = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            poly.append([x, y])

        poly = np.array(poly, np.int32)
        color = colors[cls]

        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(overlay, [poly], True, color, 2)

    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)


def draw_legend_window(class_names, colors):
    legend_w = 260
    per_h = 40
    height = len(class_names) * per_h + 20

    legend = np.zeros((height, legend_w, 3), dtype=np.uint8)

    y = 20
    for i, name in enumerate(class_names):
        color = colors[i]

        cv2.rectangle(legend, (10, y), (40, y + 30), color, -1)
        cv2.putText(
            legend,
            f"{i}: {name}",
            (50, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += per_h

    cv2.imshow("Legend", legend)


def open_file(path):
    """用系统默认应用打开文件"""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", path])
    else:  # Linux
        subprocess.call(["xdg-open", path])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", "-i", type=str, required=True, help="Image directory")
    parser.add_argument("--labels", "-l", type=str, required=True, help="Label directory")
    parser.add_argument("--classes", "-c", type=str, required=True, help="Class name file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = [l.strip() for l in f.readlines()]

    colors = [random_color(i) for i in range(len(class_names))]

    draw_legend_window(class_names, colors)

    imgs = sorted([f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    n = len(imgs)
    idx = 0

    win = "Seg Viewer"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        img_name = imgs[idx]
        img_path = os.path.join(args.images, img_name)
        label_path = os.path.join(args.labels, img_name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path)
        vis = parse_yolo_seg_overlay(img, label_path, colors)

        cv2.imshow(win, vis)
        cv2.setWindowTitle(win, f"[{idx+1}/{n}] {img_name}  (o=打开图片, l=打开label)")

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break

        elif key == ord("f"):
            if idx < n - 1:
                idx += 1
            else:
                print("已经是最后一张！")

        elif key == ord("d"):
            if idx > 0:
                idx -= 1

        elif key == ord("o"):
            print(f"打开图片: {img_path}")
            open_file(img_path)

        elif key == ord("l"):
            if os.path.exists(label_path):
                print(f"打开标注文件: {label_path}")
                open_file(label_path)
            else:
                print("⚠ 没有对应的 label 文件！")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
