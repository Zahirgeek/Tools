#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML标注文件转YOLO格式转换器
支持深度遍历目录，保持原有目录结构
"""

import os
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import shutil


def load_classes(classes_file):
    """
    加载类别文件
    
    Args:
        classes_file (str): classes.txt文件路径
    
    Returns:
        dict: 类别名称到索引的映射
    """
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"找不到类别文件: {classes_file}")
    
    class_names = {}
    with open(classes_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            if class_name:
                class_names[class_name] = idx
    
    print(f"加载了 {len(class_names)} 个类别: {list(class_names.keys())}")
    return class_names


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    将XML格式的边界框转换为YOLO格式
    
    Args:
        bbox (dict): 包含xmin, ymin, xmax, ymax的字典
        img_width (int): 图片宽度
        img_height (int): 图片高度
    
    Returns:
        tuple: (center_x, center_y, width, height) 相对坐标
    """
    xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    
    # 计算中心点和宽高
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # 转换为相对坐标
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return center_x, center_y, width, height


def parse_xml_file(xml_file, class_names):
    """
    解析XML标注文件
    
    Args:
        xml_file (str): XML文件路径
        class_names (dict): 类别名称映射
    
    Returns:
        list: YOLO格式的标注列表
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        yolo_annotations = []
        
        # 解析所有对象
        for obj in root.findall('object'):
            # 获取类别名称（使用<name>标签）
            class_name = obj.find('name').text.strip()
            
            if class_name not in class_names:
                print(f"警告: 未知类别 '{class_name}' 在文件 {xml_file}")
                continue
            
            class_id = class_names[class_name]
            
            # 获取边界框坐标
            bndbox = obj.find('bndbox')
            bbox = {
                'xmin': float(bndbox.find('xmin').text),
                'ymin': float(bndbox.find('ymin').text),
                'xmax': float(bndbox.find('xmax').text),
                'ymax': float(bndbox.find('ymax').text)
            }
            
            # 转换为YOLO格式
            center_x, center_y, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    except Exception as e:
        print(f"解析XML文件 {xml_file} 时出错: {e}")
        return []


def convert_xml_to_yolo(input_dir, output_dir, classes_file):
    """
    将XML标注文件转换为YOLO格式
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        classes_file (str): 类别文件路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载类别
    class_names = load_classes(classes_file)
    
    # 复制classes.txt到输出目录
    shutil.copy2(classes_file, output_path / "classes.txt")
    
    xml_count = 0
    success_count = 0
    
    # 深度遍历输入目录
    for xml_file in input_path.rglob("*.xml"):
        xml_count += 1
        
        # 计算相对路径，保持目录结构
        relative_path = xml_file.relative_to(input_path)
        
        # 创建对应的输出目录
        output_file_dir = output_path / relative_path.parent
        output_file_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件路径（.xml -> .txt）
        output_file = output_file_dir / (xml_file.stem + ".txt")
        
        # 转换XML到YOLO格式
        yolo_annotations = parse_xml_file(xml_file, class_names)
        
        # 写入YOLO标注文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')
        
        success_count += 1
        print(f"已转换: {relative_path} -> {output_file.relative_to(output_path)}")
    
    print(f"\n转换完成！")
    print(f"总共处理: {xml_count} 个XML文件")
    print(f"成功转换: {success_count} 个文件")
    print(f"输出目录: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="XML标注文件转YOLO格式转换器")
    parser.add_argument("--input", "-i", required=True, help="输入目录路径（包含XML文件）")
    parser.add_argument("--output", "-o", required=True, help="输出目录路径")
    parser.add_argument("--classes", "-c", default="classes.txt", help="类别文件路径 (默认: classes.txt)")
    
    args = parser.parse_args()
    
    try:
        print("=" * 50)
        print("XML转YOLO格式转换器")
        print("=" * 50)
        print(f"输入目录: {args.input}")
        print(f"输出目录: {args.output}")
        print(f"类别文件: {args.classes}")
        print("-" * 50)
        
        convert_xml_to_yolo(args.input, args.output, args.classes)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 