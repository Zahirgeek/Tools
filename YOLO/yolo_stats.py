#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO标注统计脚本
统计标注图片数量和标注框（实例）数量
"""

import os
import glob
import argparse
from collections import defaultdict

def count_yolo_annotations(directory_path="."):
    """
    统计YOLO标注数据
    
    Args:
        directory_path: 标注文件所在目录路径，默认为当前目录
    
    Returns:
        dict: 包含统计信息的字典
    """
    # 获取所有txt标注文件
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    
    # 统计变量
    total_images = 0  # 有标注的图片数量
    total_boxes = 0   # 总标注框数量
    class_counts = defaultdict(int)  # 各类别标注框数量
    empty_images = 0  # 空标注文件数量
    
    print("正在扫描标注文件...")
    
    for txt_file in txt_files:
        # 检查对应的图片文件是否存在
        base_name = os.path.splitext(txt_file)[0]
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_exists = False
        
        for ext in image_extensions:
            if os.path.exists(base_name + ext):
                image_exists = True
                break
        
        if not image_exists:
            continue  # 如果没有对应的图片文件，跳过这个标注文件
        
        total_images += 1
        
        # 读取标注文件内容
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 统计当前文件中的标注框
            file_boxes = 0
            for line in lines:
                line = line.strip()
                if line:  # 非空行
                    parts = line.split()
                    if len(parts) >= 5:  # 确保格式正确
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        file_boxes += 1
                        total_boxes += 1
            
            if file_boxes == 0:
                empty_images += 1
                
        except Exception as e:
            print(f"读取文件 {txt_file} 时出错: {e}")
            continue
    
    return {
        'total_images': total_images,
        'total_boxes': total_boxes,
        'class_counts': dict(class_counts),
        'empty_images': empty_images
    }

def print_statistics(stats):
    """
    打印统计结果
    
    Args:
        stats: 统计信息字典
    """
    print("\n" + "="*50)
    print("YOLO标注统计结果")
    print("="*50)
    
    print(f"📊 标注图片数量: {stats['total_images']} 张")
    print(f"📦 总标注框数量: {stats['total_boxes']} 个")
    print(f"📭 空标注文件: {stats['empty_images']} 个")
    
    if stats['total_images'] > 0:
        avg_boxes = stats['total_boxes'] / stats['total_images']
        print(f"📈 平均每张图片标注框数: {avg_boxes:.2f} 个")
    
    print("\n📋 各类别标注框统计:")
    if stats['class_counts']:
        for class_id in sorted(stats['class_counts'].keys()):
            count = stats['class_counts'][class_id]
            percentage = (count / stats['total_boxes']) * 100 if stats['total_boxes'] > 0 else 0
            print(f"   类别 {class_id}: {count} 个 ({percentage:.1f}%)")
    else:
        print("   未找到任何标注框")
    
    print("="*50)

def save_statistics_to_file(stats, output_file):
    """
    保存统计结果到文件
    
    Args:
        stats: 统计信息字典
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("YOLO标注统计结果\n")
            f.write("="*50 + "\n")
            f.write(f"标注图片数量: {stats['total_images']} 张\n")
            f.write(f"总标注框数量: {stats['total_boxes']} 个\n")
            f.write(f"空标注文件: {stats['empty_images']} 个\n")
            
            if stats['total_images'] > 0:
                avg_boxes = stats['total_boxes'] / stats['total_images']
                f.write(f"平均每张图片标注框数: {avg_boxes:.2f} 个\n")
            
            f.write("\n各类别标注框统计:\n")
            if stats['class_counts']:
                for class_id in sorted(stats['class_counts'].keys()):
                    count = stats['class_counts'][class_id]
                    percentage = (count / stats['total_boxes']) * 100 if stats['total_boxes'] > 0 else 0
                    f.write(f"类别 {class_id}: {count} 个 ({percentage:.1f}%)\n")
            else:
                f.write("未找到任何标注框\n")
        
        print(f"\n💾 统计结果已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"保存结果文件时出错: {e}")
        return False

def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='YOLO标注统计工具')
    parser.add_argument('-d', '--directory', type=str, default='.', 
                       help='要统计的文件夹目录路径 (默认: 当前目录)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出文件路径 (默认: 不输出文件)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果到文件，只在控制台显示')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.directory):
        print(f"❌ 错误: 目录 '{args.directory}' 不存在")
        return
    
    if not os.path.isdir(args.directory):
        print(f"❌ 错误: '{args.directory}' 不是一个目录")
        return
    
    print("YOLO标注统计工具")
    print(f"正在分析目录: {os.path.abspath(args.directory)}")
    
    # 执行统计
    stats = count_yolo_annotations(args.directory)
    
    # 打印结果
    print_statistics(stats)
    
    # 保存结果到文件（如果指定了输出文件且没有设置不保存）
    if not args.no_save and args.output:
        save_statistics_to_file(stats, args.output)
    elif not args.no_save and not args.output:
        # 如果没有指定输出文件，询问用户是否要保存
        try:
            save_choice = input("\n是否要保存统计结果到文件? (y/n): ").lower().strip()
            if save_choice in ['y', 'yes', '是']:
                default_output = "yolo_statistics.txt"
                output_file = input(f"请输入输出文件名 (直接回车使用默认名称 '{default_output}'): ").strip()
                if not output_file:
                    output_file = default_output
                save_statistics_to_file(stats, output_file)
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            print(f"\n输入处理出错: {e}")

if __name__ == "__main__":
    main()
