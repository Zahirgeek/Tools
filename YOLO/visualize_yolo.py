#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO标签可视化工具
支持显示图片和对应的YOLO格式标签，支持中文字体
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import random
import colorsys


def load_classes(classes_file):
    """
    加载类别文件
    
    Args:
        classes_file (str): classes.txt文件路径
    
    Returns:
        list: 类别名称列表
    """
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"找不到类别文件: {classes_file}")
    
    classes = []
    with open(classes_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            class_name = line.strip()
            if class_name:
                classes.append(class_name)
    
    print(f"加载了 {len(classes)} 个类别: {classes}")
    return classes


def generate_colors(num_classes):
    """
    为不同类别生成不同的颜色
    
    Args:
        num_classes (int): 类别数量
    
    Returns:
        list: BGR颜色列表
    """
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    将YOLO格式坐标转换为边界框坐标
    
    Args:
        yolo_coords (list): [center_x, center_y, width, height] (相对坐标)
        img_width (int): 图片宽度
        img_height (int): 图片高度
    
    Returns:
        tuple: (x1, y1, x2, y2) 绝对坐标
    """
    center_x, center_y, width, height = yolo_coords
    
    # 转换为绝对坐标
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    
    # 计算边界框坐标
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return x1, y1, x2, y2


def load_yolo_annotations(label_file, classes):
    """
    加载YOLO格式的标注文件
    
    Args:
        label_file (str): 标注文件路径
        classes (list): 类别名称列表
    
    Returns:
        list: 标注信息列表
    """
    annotations = []
    
    if not os.path.exists(label_file):
        return annotations
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    
                    if 0 <= class_id < len(classes):
                        annotations.append({
                            'class_id': class_id,
                            'class_name': classes[class_id],
                            'coords': coords
                        })
                    else:
                        print(f"警告: 无效的类别ID {class_id}")
    
    return annotations


def put_chinese_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """
    在图片上绘制中文文本
    
    Args:
        img (numpy.ndarray): 图片数组
        text (str): 要绘制的文本
        position (tuple): 文本位置 (x, y)
        font_scale (float): 字体大小
        color (tuple): 文本颜色 (B, G, R)
        thickness (int): 线条粗细
    """
    try:
        # 尝试使用PIL来处理中文
        from PIL import Image, ImageDraw, ImageFont
        import platform
        
        # 转换OpenCV图片到PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 根据操作系统选择中文字体
        font_size = int(font_scale * 24)
        font = None
        
        # Windows系统字体路径列表
        if platform.system() == "Windows":
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/simsun.ttc",    # 宋体
                "C:/Windows/Fonts/kaiu.ttf",      # 楷体
                "C:/Windows/Fonts/simkai.ttf",    # 楷体
                "C:/Windows/Fonts/simfang.ttf",   # 仿宋
            ]
        elif platform.system() == "Linux":
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            ]
        else:  # macOS
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
            ]
        
        # 尝试加载字体
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except Exception as e:
                continue
        
        # 如果所有字体都失败，尝试下载默认字体
        if font is None:
            try:
                font = ImageFont.load_default()
            except:
                # 如果连默认字体都加载失败，回退到OpenCV
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, color, thickness)
                return
        
        # 绘制文本背景（提高可读性）
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # 兼容旧版本PIL
            text_width, text_height = draw.textsize(text, font=font)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        
        # 转换回OpenCV格式
        img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    except ImportError:
        # 如果没有PIL，使用OpenCV的putText（英文替代）
        # 对中文类别名称使用英文映射或数字
        english_text = text
        if text == "left":
            english_text = "LEFT"
        elif text == "right":
            english_text = "RIGHT"
        else:
            # 如果是其他中文，使用拼音或编号
            english_text = f"Class_{hash(text) % 1000}"
        
        cv2.putText(img, english_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
    except Exception as e:
        # 最后的兜底方案：使用英文或数字
        fallback_text = text
        try:
            # 尝试编码检测
            if not text.isascii():
                fallback_text = f"Class_{ord(text[0]) % 100}" if text else "Unknown"
        except:
            fallback_text = "Label"
        
        cv2.putText(img, fallback_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)


def draw_annotations(img, annotations, colors):
    """
    在图片上绘制标注
    
    Args:
        img (numpy.ndarray): 图片数组
        annotations (list): 标注信息列表
        colors (list): 颜色列表
    
    Returns:
        numpy.ndarray: 绘制后的图片
    """
    img_height, img_width = img.shape[:2]
    annotated_img = img.copy()
    
    for ann in annotations:
        class_id = ann['class_id']
        class_name = ann['class_name']
        coords = ann['coords']
        
        # 转换坐标
        x1, y1, x2, y2 = yolo_to_bbox(coords, img_width, img_height)
        
        # 确保坐标在图片范围内
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # 获取颜色
        color = colors[class_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        label = f"{class_name}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # 绘制标签背景
        cv2.rectangle(annotated_img, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # 绘制标签文本
        put_chinese_text(annotated_img, label, (x1, y1 - label_size[1] - 5), 
                        font_scale=0.7, color=(255, 255, 255), thickness=2)
    
    return annotated_img


def find_image_label_pairs(image_dir, label_dir):
    """
    查找图片和标签文件对
    
    Args:
        image_dir (str): 图片目录
        label_dir (str): 标签目录
    
    Returns:
        list: 图片和标签文件对的列表
    """
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    
    if not image_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    
    if not label_path.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    pairs = []
    
    # 遍历所有图片文件
    for img_file in image_path.rglob("*"):
        if img_file.suffix.lower() in image_extensions:
            # 查找对应的标签文件
            relative_path = img_file.relative_to(image_path)
            label_file = label_path / relative_path.with_suffix('.txt')
            
            pairs.append({
                'image': img_file,
                'label': label_file,
                'has_label': label_file.exists()
            })
    
    return pairs


def visualize_dataset(image_dir, label_dir, classes_file, show_unlabeled=False):
    """
    可视化数据集
    
    Args:
        image_dir (str): 图片目录
        label_dir (str): 标签目录
        classes_file (str): 类别文件
        show_unlabeled (bool): 是否显示没有标签的图片
    """
    # 加载类别
    classes = load_classes(classes_file)
    colors = generate_colors(len(classes))
    
    # 查找图片和标签对
    pairs = find_image_label_pairs(image_dir, label_dir)
    
    if not pairs:
        print("没有找到任何图片文件！")
        return
    
    print(f"找到 {len(pairs)} 个图片文件")
    
    # 过滤掉没有标签的图片（如果需要）
    if not show_unlabeled:
        pairs = [p for p in pairs if p['has_label']]
        print(f"其中 {len(pairs)} 个有对应的标签文件")
    
    if not pairs:
        print("没有找到有标签的图片！")
        return
    
    current_index = 0
    
    print("\n操作说明:")
    print("- 按 'f' 下一张图片")
    print("- 按 'd' 上一张图片") 
    print("- 按 'q' 或 'ESC' 退出")
    print("- 按 's' 保存当前图片")
    print("=" * 50)
    
    while True:
        pair = pairs[current_index]
        img_file = pair['image']
        label_file = pair['label']
        has_label = pair['has_label']
        
        # 读取图片
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"无法读取图片: {img_file}")
                current_index = (current_index + 1) % len(pairs)
                continue
        except Exception as e:
            print(f"读取图片出错 {img_file}: {e}")
            current_index = (current_index + 1) % len(pairs)
            continue
        
        # 加载标注
        annotations = []
        if has_label:
            annotations = load_yolo_annotations(str(label_file), classes)
        
        # 绘制标注
        if annotations:
            annotated_img = draw_annotations(img, annotations, colors)
        else:
            annotated_img = img.copy()
        
        # 添加信息文本（使用英文避免乱码）
        info_text = f"Image: {img_file.name} ({current_index + 1}/{len(pairs)})"
        if has_label:
            info_text += f" | Objects: {len(annotations)}"
        else:
            info_text += " | No labels"
        
        # 在图片上显示信息
        cv2.putText(annotated_img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 调整图片大小以适应屏幕
        screen_height = 800
        img_height, img_width = annotated_img.shape[:2]
        if img_height > screen_height:
            scale = screen_height / img_height
            new_width = int(img_width * scale)
            annotated_img = cv2.resize(annotated_img, (new_width, screen_height))
        
        # 显示图片
        cv2.imshow('YOLO Dataset Viewer', annotated_img)
        
        # 处理按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            break
        elif key == ord('f'):  # 'f' - 下一张
            current_index = (current_index + 1) % len(pairs)
        elif key == ord('d'):  # 'd' - 上一张
            current_index = (current_index - 1) % len(pairs)
        elif key == ord('s'):  # 's' 保存
            save_path = f"annotated_{img_file.name}"
            cv2.imwrite(save_path, annotated_img)
            print(f"已保存: {save_path}")
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO标签可视化工具")
    parser.add_argument("--images", "-i", required=True, help="图片目录路径")
    parser.add_argument("--labels", "-l", required=True, help="YOLO标签目录路径")
    parser.add_argument("--classes", "-c", default="classes.txt", help="类别文件路径 (默认: classes.txt)")
    parser.add_argument("--show-unlabeled", action="store_true", help="显示没有标签的图片")
    
    args = parser.parse_args()
    
    try:
        print("=" * 50)
        print("YOLO标签可视化工具")
        print("=" * 50)
        print(f"图片目录: {args.images}")
        print(f"标签目录: {args.labels}")
        print(f"类别文件: {args.classes}")
        print(f"显示无标签图片: {args.show_unlabeled}")
        print("-" * 50)
        
        visualize_dataset(args.images, args.labels, args.classes, args.show_unlabeled)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 