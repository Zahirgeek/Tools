#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机复制/剪切文件脚本
根据用户输入的路径，深入遍历其每个子文件夹，找到最深的含有文件的子文件夹路径。
根据用户输入的随机选取的文件数n，对每个子文件夹路径中的文件进行随机获取，
每个子文件夹获取n个文件，再按照用户输入的模式(复制或剪切)，将选中的文件输出到用户输入的输出文件夹中
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class RandomFileProcessor:
    """随机文件处理器类"""
    
    def __init__(self):
        self.supported_modes = ['copy', 'cut', '复制', '剪切']
        
    def find_deepest_file_folders(self, root_path: str) -> List[str]:
        """
        找到最深的含有文件的子文件夹路径
        
        Args:
            root_path: 根目录路径
            
        Returns:
            包含最深层级文件夹路径的列表
        """
        deepest_folders = []
        
        for root, dirs, files in os.walk(root_path):
            # 如果当前目录有文件，并且没有子目录，则认为是最深的文件夹
            if files and not dirs:
                deepest_folders.append(root)
            # 如果当前目录有文件，但子目录中都没有文件，也认为是最深的文件夹
            elif files:
                has_files_in_subdirs = False
                for subdir in dirs:
                    subdir_path = os.path.join(root, subdir)
                    if self._has_files_in_subdirs(subdir_path):
                        has_files_in_subdirs = True
                        break
                
                if not has_files_in_subdirs:
                    deepest_folders.append(root)
        
        return deepest_folders
    
    def _has_files_in_subdirs(self, path: str) -> bool:
        """检查路径及其子目录中是否有文件"""
        for root, dirs, files in os.walk(path):
            if files:
                return True
        return False
    
    def get_random_files(self, folder_path: str, n: int) -> List[str]:
        """
        从指定文件夹中随机获取n个文件
        
        Args:
            folder_path: 文件夹路径
            n: 要获取的文件数量
            
        Returns:
            随机选择的文件路径列表
        """
        try:
            # 获取文件夹中的所有文件
            files = [f for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f))]
            
            if not files:
                return []
            
            # 如果文件数量少于n，则返回所有文件
            if len(files) <= n:
                return [os.path.join(folder_path, f) for f in files]
            
            # 随机选择n个文件
            selected_files = random.sample(files, n)
            return [os.path.join(folder_path, f) for f in selected_files]
            
        except Exception as e:
            print(f"获取文件列表时出错：{folder_path} - {e}")
            return []
    
    def create_output_structure(self, input_path: str, output_path: str, 
                               selected_files: Dict[str, List[str]]) -> None:
        """
        创建输出目录结构
        
        Args:
            input_path: 输入根目录路径
            output_path: 输出根目录路径
            selected_files: 选择的文件字典 {folder_path: [file_paths]}
        """
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        # 创建输出根目录
        os.makedirs(output_path, exist_ok=True)
        
        for folder_path, file_paths in selected_files.items():
            # 计算相对路径
            relative_path = os.path.relpath(folder_path, input_path)
            
            # 创建对应的输出目录
            output_folder = os.path.join(output_path, relative_path)
            os.makedirs(output_folder, exist_ok=True)
    
    def process_files(self, input_path: str, output_path: str, 
                     selected_files: Dict[str, List[str]], mode: str) -> None:
        """
        处理文件（复制或剪切）
        
        Args:
            input_path: 输入根目录路径
            output_path: 输出根目录路径
            selected_files: 选择的文件字典
            mode: 处理模式（copy/cut/复制/剪切）
        """
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        total_files = sum(len(files) for files in selected_files.values())
        processed_files = 0
        
        print(f"开始处理文件，总共 {total_files} 个文件...")
        
        for folder_path, file_paths in selected_files.items():
            # 计算相对路径
            relative_path = os.path.relpath(folder_path, input_path)
            output_folder = os.path.join(output_path, relative_path)
            
            for file_path in file_paths:
                try:
                    filename = os.path.basename(file_path)
                    output_file_path = os.path.join(output_folder, filename)
                    
                    if mode.lower() in ['copy', '复制']:
                        shutil.copy2(file_path, output_file_path)
                        print(f"复制: {file_path} -> {output_file_path}")
                    elif mode.lower() in ['cut', '剪切']:
                        shutil.move(file_path, output_file_path)
                        print(f"剪切: {file_path} -> {output_file_path}")
                    
                    processed_files += 1
                    
                    # 显示进度
                    if processed_files % 10 == 0 or processed_files == total_files:
                        print(f"进度: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
                        
                except Exception as e:
                    print(f"处理文件时出错：{file_path} - {e}")
        
        print(f"处理完成！共处理了 {processed_files} 个文件")
    
    def run(self, input_path: str, count: int, mode: str, output_path: str, 
            confirm: bool = False) -> None:
        """
        主运行函数
        
        Args:
            input_path: 输入文件夹路径
            count: 每个子文件夹选取的文件数
            mode: 处理模式
            output_path: 输出文件夹路径
            confirm: 是否跳过确认提示
        """
        print("=" * 50)
        print("随机文件复制/剪切脚本")
        print("=" * 50)
        
        # 验证输入参数
        if not os.path.exists(input_path):
            print(f"错误：输入路径不存在：{input_path}")
            return
        
        if count <= 0:
            print("错误：文件数必须大于0！")
            return
        
        if mode not in self.supported_modes:
            print(f"错误：不支持的模式 '{mode}'，请使用: {', '.join(self.supported_modes)}")
            return
        
        # 显示操作信息
        print("操作信息:")
        print(f"输入路径: {input_path}")
        print(f"每个子文件夹选取文件数: {count}")
        print(f"处理模式: {mode}")
        print(f"输出路径: {output_path}")
        print("=" * 50)
        
        # 确认操作（如果需要）
        if not confirm:
            user_input = input("是否继续执行？(y/n): ").strip().lower()
            if user_input not in ['y', 'yes', '是']:
                print("操作已取消")
                return
        
        # 开始处理
        print("\n正在查找最深的文件夹...")
        deepest_folders = self.find_deepest_file_folders(input_path)
        
        if not deepest_folders:
            print("未找到包含文件的文件夹！")
            return
        
        print(f"找到 {len(deepest_folders)} 个最深的文件夹")
        
        # 从每个文件夹中随机选择文件
        selected_files = {}
        total_selected = 0
        
        print("\n正在从每个文件夹中随机选择文件...")
        for folder in deepest_folders:
            files = self.get_random_files(folder, count)
            if files:
                selected_files[folder] = files
                total_selected += len(files)
                print(f"从 {folder} 中选择了 {len(files)} 个文件")
        
        if not selected_files:
            print("未选择到任何文件！")
            return
        
        print(f"\n总共选择了 {total_selected} 个文件")
        
        # 创建输出目录结构
        print("\n正在创建输出目录结构...")
        self.create_output_structure(input_path, output_path, selected_files)
        
        # 处理文件
        print("\n开始处理文件...")
        self.process_files(input_path, output_path, selected_files, mode)
        
        print("\n处理完成！")


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="随机复制/剪切文件脚本 - 从每个最深的子文件夹中随机选择指定数量的文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s -i /path/to/input -c 5 -m copy -o /path/to/output
  %(prog)s -i ./photos -c 3 -m 剪切 -o ./selected_photos --yes
  %(prog)s --input-path "C:\\Users\\photos" --count 10 --mode cut --output-path "C:\\Users\\output"

支持的模式:
  copy, 复制    - 复制文件到输出目录
  cut, 剪切     - 移动文件到输出目录
        """
    )
    
    parser.add_argument(
        '-i', '--input-path',
        required=True,
        help='输入文件夹路径'
    )
    
    parser.add_argument(
        '-c', '--count',
        type=int,
        required=True,
        help='每个子文件夹随机选取的文件数量'
    )
    
    parser.add_argument(
        '-m', '--mode',
        required=True,
        choices=['copy', 'cut', '复制', '剪切'],
        help='处理模式：copy/复制（复制文件）或 cut/剪切（移动文件）'
    )
    
    parser.add_argument(
        '-o', '--output-path',
        required=True,
        help='输出文件夹路径'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='跳过确认提示，直接执行'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    processor = RandomFileProcessor()
    processor.run(
        input_path=args.input_path,
        count=args.count,
        mode=args.mode,
        output_path=args.output_path,
        confirm=args.yes
    )


if __name__ == "__main__":
    main() 