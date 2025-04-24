#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import shutil
from math import ceil
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将文件夹中的文件分割为n份')
    
    # 必需参数
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='输入文件夹路径')
    
    # 可选参数
    parser.add_argument('-o', '--output_dir', type=str, default='output',
                        help='输出文件夹路径，默认为当前目录下的output文件夹')
    parser.add_argument('-n', '--num_splits', type=int, default=2,
                        help='分割份数，默认为2')
    parser.add_argument('-r', '--ratios', type=str, default=None,
                        help='各份占比，用逗号分隔，如"0.7,0.3"，默认为平均分配')
    parser.add_argument('-s', '--random_split', action='store_true',
                        help='是否随机分割，默认按文件名顺序分割')
    parser.add_argument('-m', '--move', action='store_true',
                        help='是否剪切文件（移动），默认为复制')
    parser.add_argument('-d', '--recursive', action='store_true',
                        help='是否递归处理子文件夹，默认不处理')
    
    return parser.parse_args()


def get_all_files(directory, recursive=False):
    """获取目录下的所有文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归处理子目录
        
    Returns:
        文件路径列表
    """
    files = []
    directory = Path(directory)
    
    if recursive:
        # 递归遍历所有子目录
        for item in directory.glob('**/*'):
            if item.is_file():
                files.append(item)
    else:
        # 只处理当前目录下的文件
        for item in directory.iterdir():
            if item.is_file():
                files.append(item)
    
    return files


def split_files(files, num_splits, ratios=None):
    """将文件列表分割为指定份数
    
    Args:
        files: 文件列表
        num_splits: 分割份数
        ratios: 各份占比，如果为None则平均分配
        
    Returns:
        分割后的文件列表的列表
    """
    if not files:
        return []
    
    total_files = len(files)
    
    # 如果未指定比例，则平均分配
    if ratios is None:
        ratios = [1/num_splits] * num_splits
    
    # 确保比例数量与分割份数一致
    if len(ratios) != num_splits:
        raise ValueError(f"比例数量({len(ratios)})与分割份数({num_splits})不一致")
    
    # 确保比例之和为1
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 0.01:  # 允许小误差
        # 归一化
        ratios = [r / ratio_sum for r in ratios]
    
    # 计算每份的文件数量
    file_counts = []
    remaining = total_files
    
    for i in range(num_splits - 1):
        count = int(total_files * ratios[i])
        file_counts.append(count)
        remaining -= count
    
    file_counts.append(remaining)  # 最后一份使用剩余文件，避免舍入误差
    
    # 分割文件
    result = []
    start_idx = 0
    
    for count in file_counts:
        end_idx = start_idx + count
        result.append(files[start_idx:end_idx])
        start_idx = end_idx
    
    return result


def process_files(input_dir, output_dir, num_splits=2, ratios=None, 
                 random_split=False, move=False, recursive=False):
    """处理文件分割
    
    Args:
        input_dir: 输入文件夹
        output_dir: 输出文件夹
        num_splits: 分割份数
        ratios: 各份占比
        random_split: 是否随机分割
        move: 是否剪切文件
        recursive: 是否递归处理子文件夹
    """
    # 获取所有文件
    files = get_all_files(input_dir, recursive)
    
    if not files:
        print(f"警告: 在 {input_dir} 中没有找到文件" + (" (包括子文件夹)" if recursive else ""))
        return
    
    # 如果随机分割，则打乱文件顺序
    if random_split:
        random.shuffle(files)
    else:
        # 按文件名排序
        files.sort(key=lambda x: x.name)
    
    # 解析比例
    split_ratios = None
    if ratios:
        try:
            split_ratios = [float(r) for r in ratios.split(',')]
        except ValueError:
            print(f"警告: 无法解析比例 '{ratios}'，将使用平均分配")
    
    # 分割文件
    split_files_list = split_files(files, num_splits, split_ratios)
    
    # 创建输出目录
    output_base = Path(output_dir)
    if not output_base.exists():
        output_base.mkdir(parents=True)
    
    # 处理每一份
    for i, file_list in enumerate(split_files_list):
        # 为每一份创建子目录
        split_dir = output_base / f"split_{i+1}"
        if not split_dir.exists():
            split_dir.mkdir(parents=True)
        
        # 复制或移动文件
        for file_path in file_list:
            # 如果递归处理，保持目录结构
            if recursive:
                # 获取文件相对于输入目录的路径
                rel_path = file_path.relative_to(input_dir)
                # 创建目标文件夹
                target_dir = split_dir / rel_path.parent
                if not target_dir.exists():
                    target_dir.mkdir(parents=True, exist_ok=True)
                # 目标文件路径
                dest_path = target_dir / file_path.name
            else:
                # 简单地将文件放在目标文件夹下
                dest_path = split_dir / file_path.name
            
            # 复制或移动文件
            try:
                if move:
                    shutil.move(str(file_path), str(dest_path))
                else:
                    shutil.copy2(str(file_path), str(dest_path))
            except (shutil.Error, OSError) as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    
    # 打印结果统计
    for i, file_list in enumerate(split_files_list):
        print(f"分割 {i+1}: {len(file_list)} 个文件")


def main():
    """主函数"""
    args = parse_args()
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"分割份数: {args.num_splits}")
    print(f"分割比例: {args.ratios if args.ratios else '平均分配'}")
    print(f"分割方式: {'随机' if args.random_split else '按文件名顺序'}")
    print(f"处理方式: {'移动' if args.move else '复制'}")
    print(f"递归处理: {'是' if args.recursive else '否'}")
    
    process_files(
        args.input_dir,
        args.output_dir,
        args.num_splits,
        args.ratios,
        args.random_split,
        args.move,
        args.recursive
    )
    
    print("处理完成!")


if __name__ == "__main__":
    main() 