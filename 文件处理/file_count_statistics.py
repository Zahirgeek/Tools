#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件数量统计脚本
功能：根据用户输入的路径和第几级子目录，统计子目录同级所有目录及其子目录中的文件数量
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileCountStatistics:
    """文件数量统计器"""
    
    def __init__(self, root_path: str, target_level: int = 1):
        """
        初始化统计器
        
        Args:
            root_path: 根路径
            target_level: 目标子目录级别（从1开始）
        """
        self.root_path = Path(root_path)
        self.target_level = target_level
        self.statistics = defaultdict(int)
        
        if not self.root_path.exists():
            raise ValueError(f"路径不存在: {root_path}")
        
        if not self.root_path.is_dir():
            raise ValueError(f"路径不是目录: {root_path}")
    
    def get_subdirectories_at_level(self) -> List[Path]:
        """
        获取指定级别的所有子目录
        
        Returns:
            指定级别的子目录列表
        """
        subdirs = []
        
        def traverse(current_path: Path, current_level: int):
            if current_level == self.target_level:
                # 到达目标级别，收集所有同级目录
                parent = current_path.parent
                if parent.exists():
                    for item in parent.iterdir():
                        if item.is_dir():
                            subdirs.append(item)
                return
            
            # 继续遍历下一级
            for item in current_path.iterdir():
                if item.is_dir():
                    traverse(item, current_level + 1)
        
        traverse(self.root_path, 0)
        return list(set(subdirs))  # 去重
    
    def count_files_in_directory(self, directory: Path) -> int:
        """
        递归统计目录及其所有子目录中的文件数量
        
        Args:
            directory: 目标目录
            
        Returns:
            文件总数
        """
        file_count = 0
        
        try:
            # 使用os.walk递归遍历所有子目录，统计文件数量
            for root, dirs, files in os.walk(directory):
                file_count += len(files)
        except PermissionError:
            logger.warning(f"无法访问目录: {directory}")
            return 0
        except Exception as e:
            logger.error(f"统计目录 {directory} 时出错: {e}")
            return 0
        
        return file_count
    
    def get_relative_path(self, full_path: Path) -> str:
        """
        获取相对于根路径的相对路径
        
        Args:
            full_path: 完整路径
            
        Returns:
            相对路径字符串
        """
        try:
            return str(full_path.relative_to(self.root_path))
        except ValueError:
            return str(full_path)
    
    def natural_sort_key(self, path_str: str) -> List:
        """
        自然排序键，用于按数字顺序排序路径
        
        Args:
            path_str: 路径字符串
            
        Returns:
            排序键列表
        """
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        
        return [convert(c) for c in re.split('([0-9]+)', path_str)]
    
    def run_statistics(self) -> Dict[str, int]:
        """
        执行统计
        
        Returns:
            统计结果字典 {相对路径: 文件数量}
        """
        logger.info(f"开始统计路径: {self.root_path}")
        logger.info(f"目标子目录级别: {self.target_level}")
        
        # 获取指定级别的子目录
        target_dirs = self.get_subdirectories_at_level()
        
        if not target_dirs:
            logger.warning(f"在级别 {self.target_level} 未找到任何子目录")
            return {}
        
        logger.info(f"找到 {len(target_dirs)} 个目标级别的子目录")
        
        # 统计每个目录的文件数量
        results = {}
        total_files = 0
        
        for directory in sorted(target_dirs):
            relative_path = self.get_relative_path(directory)
            file_count = self.count_files_in_directory(directory)
            
            results[relative_path] = file_count
            total_files += file_count
            
            logger.info(f"目录: {relative_path} - 文件数量: {file_count}")
        
        logger.info(f"统计完成，总文件数: {total_files}")
        return results
    
    def print_results(self, results: Dict[str, int]):
        """
        打印统计结果
        
        Args:
            results: 统计结果
        """
        if not results:
            print("未找到任何文件")
            return
        
        print("\n" + "="*70)
        print("文件数量统计结果")
        print("="*70)
        
        # 按文件数量排序，然后按路径自然排序
        sorted_results = sorted(results.items(), 
                              key=lambda x: (-x[1], self.natural_sort_key(x[0])))
        
        print(f"{'序号':<4} {'相对路径':<45} {'文件数量':<10}")
        print("-" * 70)
        
        total_files = 0
        for i, (relative_path, file_count) in enumerate(sorted_results, 1):
            print(f"{i:<4} {relative_path:<45} {file_count:<10}")
            total_files += file_count
        
        print("-" * 70)
        print(f"{'总计':<49} {total_files:<10}")
        print("="*70)
        
        # 显示统计摘要
        print(f"\n📊 统计摘要:")
        print(f"   • 统计目录数: {len(results)}")
        print(f"   • 总文件数: {total_files}")
        print(f"   • 平均每个目录文件数: {total_files/len(results):.1f}")
        print(f"   • 最多文件的目录: {sorted_results[0][0]} ({sorted_results[0][1]}个文件)")
        print(f"   • 最少文件的目录: {sorted_results[-1][0]} ({sorted_results[-1][1]}个文件)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统计指定路径下子目录的文件数量")
    parser.add_argument("path", help="要统计的根路径")
    parser.add_argument("-l", "--level", type=int, default=1, 
                       help="目标子目录级别（从1开始，默认为1）")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建统计器
        statistics = FileCountStatistics(args.path, args.level)
        
        # 执行统计
        results = statistics.run_statistics()
        
        # 打印结果
        statistics.print_results(results)
            
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        return 1
    except Exception as e:
        logger.error(f"执行统计时出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 