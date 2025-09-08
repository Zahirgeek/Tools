#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机文件选取和复制/剪切脚本
功能：根据用户输入的路径和文件总数，随机选取文件并复制或剪切到指定路径
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomFileProcessor:
    """随机文件处理器"""
    
    def __init__(self, source_path: str, target_count: int, output_path: str, operation: str = "copy", flat_output: bool = False):
        """
        初始化处理器
        
        Args:
            source_path: 源路径
            target_count: 目标文件总数
            output_path: 输出路径
            operation: 操作类型 ("copy" 或 "cut")
            flat_output: 是否扁平化输出（不保持目录结构）
        """
        self.source_path = Path(source_path)
        self.target_count = target_count
        self.output_path = Path(output_path)
        self.operation = operation.lower()
        self.flat_output = flat_output
        
        if not self.source_path.exists():
            raise ValueError(f"源路径不存在: {source_path}")
        
        if not self.source_path.is_dir():
            raise ValueError(f"源路径不是目录: {source_path}")
        
        if self.operation not in ["copy", "cut"]:
            raise ValueError(f"不支持的操作类型: {operation}，支持的操作: copy, cut")
    
    def get_subdirectories(self) -> List[Path]:
        """
        获取所有子目录
        
        Returns:
            子目录列表
        """
        subdirs = []
        for item in self.source_path.iterdir():
            if item.is_dir():
                subdirs.append(item)
        return sorted(subdirs)
    
    def count_files_in_directory(self, directory: Path) -> int:
        """
        统计目录及其子目录中的文件数量
        
        Args:
            directory: 目标目录
            
        Returns:
            文件总数
        """
        file_count = 0
        try:
            for root, dirs, files in os.walk(directory):
                file_count += len(files)
        except PermissionError:
            logger.warning(f"无法访问目录: {directory}")
            return 0
        except Exception as e:
            logger.error(f"统计目录 {directory} 时出错: {e}")
            return 0
        
        return file_count
    
    def get_all_files_in_directory(self, directory: Path) -> List[Path]:
        """
        获取目录及其子目录中的所有文件路径
        
        Args:
            directory: 目标目录
            
        Returns:
            文件路径列表
        """
        files = []
        try:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = Path(root) / filename
                    files.append(file_path)
        except PermissionError:
            logger.warning(f"无法访问目录: {directory}")
        except Exception as e:
            logger.error(f"获取目录 {directory} 中的文件时出错: {e}")
        
        return files
    
    def analyze_subdirectories(self) -> Dict[str, int]:
        """
        分析子目录的文件分布
        
        Returns:
            子目录文件数量统计 {相对路径: 文件数量}
        """
        logger.info("开始分析子目录文件分布...")
        
        subdirs = self.get_subdirectories()
        if not subdirs:
            logger.warning("未找到任何子目录")
            return {}
        
        file_distribution = {}
        total_files = 0
        
        for subdir in subdirs:
            relative_path = str(subdir.relative_to(self.source_path))
            file_count = self.count_files_in_directory(subdir)
            file_distribution[relative_path] = file_count
            total_files += file_count
            logger.info(f"目录: {relative_path} - 文件数量: {file_count}")
        
        logger.info(f"总文件数: {total_files}")
        return file_distribution
    
    def calculate_file_allocation(self, file_distribution: Dict[str, int]) -> Dict[str, int]:
        """
        计算每个子目录应该分配的文件数量
        
        分配逻辑：
        1. 如果某个文件夹的文件数量不够平均分配数，则全选该文件夹的文件
        2. 剩余的文件在其他文件夹中均匀随机选取
        
        Args:
            file_distribution: 文件分布统计
            
        Returns:
            文件分配方案 {相对路径: 分配数量}
        """
        logger.info("计算文件分配方案...")
        
        subdirs = list(file_distribution.keys())
        total_subdirs = len(subdirs)
        
        if total_subdirs == 0:
            return {}
        
        # 计算每个子目录的基础分配数量
        base_allocation = self.target_count // total_subdirs
        remaining_files = self.target_count % total_subdirs
        
        logger.info(f"基础分配数量: {base_allocation} 个文件/目录")
        logger.info(f"剩余文件数: {remaining_files}")
        
        allocation = {}
        insufficient_dirs = []  # 文件数量不足的目录
        sufficient_dirs = []    # 文件数量充足的目录
        
        # 第一步：识别文件数量不足和充足的目录
        for subdir in subdirs:
            available_files = file_distribution[subdir]
            if available_files <= base_allocation:
                insufficient_dirs.append(subdir)
                logger.info(f"目录 {subdir} 文件不足: {available_files} <= {base_allocation}")
            else:
                sufficient_dirs.append(subdir)
                logger.info(f"目录 {subdir} 文件充足: {available_files} > {base_allocation}")
        
        # 第二步：处理文件数量不足的目录（全选）
        total_allocated = 0
        for subdir in insufficient_dirs:
            available_files = file_distribution[subdir]
            allocation[subdir] = available_files
            total_allocated += available_files
            logger.info(f"全选目录 {subdir}: {available_files} 个文件")
        
        # 第三步：计算剩余需要分配的文件数
        remaining_to_allocate = self.target_count - total_allocated
        logger.info(f"已分配: {total_allocated} 个文件")
        logger.info(f"还需分配: {remaining_to_allocate} 个文件")
        
        # 第四步：在文件充足的目录中均匀分配剩余文件
        if remaining_to_allocate > 0 and sufficient_dirs:
            # 计算每个充足目录应该分配的数量
            files_per_sufficient_dir = remaining_to_allocate // len(sufficient_dirs)
            extra_files = remaining_to_allocate % len(sufficient_dirs)
            
            logger.info(f"每个充足目录分配: {files_per_sufficient_dir} 个文件")
            logger.info(f"额外文件数: {extra_files}")
            
            for i, subdir in enumerate(sufficient_dirs):
                available_files = file_distribution[subdir]
                target_count = files_per_sufficient_dir
                
                # 分配额外文件（前几个目录多分配一个）
                if i < extra_files:
                    target_count += 1
                
                # 确保不超过可用文件数
                actual_count = min(target_count, available_files)
                allocation[subdir] = actual_count
                
                logger.info(f"目录 {subdir} 分配: {actual_count} 个文件 (可用: {available_files})")
        
        # 验证分配结果
        total_allocated_final = sum(allocation.values())
        logger.info(f"最终分配总数: {total_allocated_final}/{self.target_count}")
        
        if total_allocated_final < self.target_count:
            logger.warning(f"分配不足: 需要 {self.target_count} 个文件，实际分配 {total_allocated_final} 个文件")
        
        logger.info("文件分配方案:")
        for subdir, count in allocation.items():
            available = file_distribution.get(subdir, 0)
            logger.info(f"  {subdir}: {count}/{available} 个文件")
        
        return allocation
    
    def select_random_files(self, allocation: Dict[str, int]) -> List[Path]:
        """
        根据分配方案随机选取文件
        
        Args:
            allocation: 文件分配方案
            
        Returns:
            选中的文件路径列表
        """
        logger.info("开始随机选取文件...")
        
        selected_files = []
        subdirs = self.get_subdirectories()
        
        for subdir in subdirs:
            relative_path = str(subdir.relative_to(self.source_path))
            if relative_path not in allocation:
                continue
            
            target_count = allocation[relative_path]
            if target_count == 0:
                continue
            
            # 获取该目录下的所有文件
            all_files = self.get_all_files_in_directory(subdir)
            
            if not all_files:
                logger.warning(f"目录 {relative_path} 中没有找到文件")
                continue
            
            # 随机选取指定数量的文件
            actual_count = min(target_count, len(all_files))
            selected = random.sample(all_files, actual_count)
            selected_files.extend(selected)
            
            logger.info(f"从 {relative_path} 选取了 {actual_count} 个文件")
        
        logger.info(f"总共选取了 {len(selected_files)} 个文件")
        return selected_files
    
    def create_output_directory(self):
        """创建输出目录"""
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录已创建: {self.output_path}")
        except Exception as e:
            logger.error(f"创建输出目录时出错: {e}")
            raise
    
    def generate_unique_filename(self, target_path: Path) -> Path:
        """
        生成唯一的文件名，避免重复
        
        Args:
            target_path: 目标路径
            
        Returns:
            唯一的文件路径
        """
        if not target_path.exists():
            return target_path
        
        # 分离文件名和扩展名
        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    def process_files(self, selected_files: List[Path]) -> Tuple[int, int]:
        """
        处理选中的文件（复制或剪切）
        
        Args:
            selected_files: 选中的文件列表
            
        Returns:
            (成功数量, 失败数量)
        """
        logger.info(f"开始{self.operation}文件...")
        
        self.create_output_directory()
        
        success_count = 0
        failed_count = 0
        
        for file_path in selected_files:
            try:
                if self.flat_output:
                    # 扁平化输出：直接保存到输出根目录
                    target_path = self.output_path / file_path.name
                else:
                    # 保持目录结构
                    relative_path = file_path.relative_to(self.source_path)
                    target_path = self.output_path / relative_path
                
                # 扁平化输出时处理重名文件
                if self.flat_output:
                    target_path = self.generate_unique_filename(target_path)
                else:
                    # 创建目标目录
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                
                if self.operation == "copy":
                    shutil.copy2(file_path, target_path)
                    logger.debug(f"复制: {file_path} -> {target_path}")
                else:  # cut
                    shutil.move(str(file_path), str(target_path))
                    logger.debug(f"剪切: {file_path} -> {target_path}")
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
                failed_count += 1
        
        return success_count, failed_count
    
    def print_summary(self, file_distribution: Dict[str, int], allocation: Dict[str, int], 
                     selected_files: List[Path], success_count: int, failed_count: int):
        """
        打印处理摘要
        
        Args:
            file_distribution: 文件分布
            allocation: 分配方案
            selected_files: 选中的文件
            success_count: 成功数量
            failed_count: 失败数量
        """
        print("\n" + "="*80)
        print("随机文件处理摘要")
        print("="*80)
        
        print(f"源路径: {self.source_path}")
        print(f"目标文件数: {self.target_count}")
        print(f"操作类型: {self.operation}")
        print(f"输出路径: {self.output_path}")
        print(f"输出模式: {'扁平化输出' if self.flat_output else '保持目录结构'}")
        print()
        
        print("📊 文件分布统计:")
        total_available = sum(file_distribution.values())
        base_allocation = self.target_count // len(file_distribution) if file_distribution else 0
        print(f"   • 子目录数量: {len(file_distribution)}")
        print(f"   • 可用文件总数: {total_available}")
        print(f"   • 目标选取文件数: {self.target_count}")
        print(f"   • 平均分配数量: {base_allocation} 个文件/目录")
        print()
        
        # 分析分配策略
        insufficient_dirs = []
        sufficient_dirs = []
        for subdir, count in allocation.items():
            available = file_distribution.get(subdir, 0)
            if available <= base_allocation:
                insufficient_dirs.append((subdir, count, available))
            else:
                sufficient_dirs.append((subdir, count, available))
        
        print("📋 文件分配方案:")
        if insufficient_dirs:
            print("   🔴 全选目录 (文件数量不足):")
            for subdir, count, available in insufficient_dirs:
                print(f"      • {subdir}: {count}/{available} 个文件 (全选)")
        
        if sufficient_dirs:
            print("   🟢 均匀分配目录 (文件数量充足):")
            for subdir, count, available in sufficient_dirs:
                print(f"      • {subdir}: {count}/{available} 个文件 (随机选取)")
        
        print()
        
        # 计算分配统计
        total_allocated = sum(allocation.values())
        allocation_rate = total_allocated / self.target_count * 100 if self.target_count > 0 else 0
        
        print("📈 分配统计:")
        print(f"   • 实际分配文件数: {total_allocated}")
        print(f"   • 分配完成率: {allocation_rate:.1f}%")
        if total_allocated < self.target_count:
            print(f"   • 分配不足: {self.target_count - total_allocated} 个文件")
        print()
        
        print("📁 处理结果:")
        print(f"   • 选中文件数: {len(selected_files)}")
        print(f"   • 成功处理: {success_count}")
        print(f"   • 处理失败: {failed_count}")
        success_rate = success_count/(success_count+failed_count)*100 if (success_count+failed_count) > 0 else 0
        print(f"   • 成功率: {success_rate:.1f}%")
        print("="*80)
    
    def run(self) -> bool:
        """
        执行完整的处理流程
        
        Returns:
            是否成功
        """
        try:
            # 1. 分析子目录文件分布
            file_distribution = self.analyze_subdirectories()
            if not file_distribution:
                logger.error("没有找到可处理的子目录")
                return False
            
            # 2. 计算文件分配方案
            allocation = self.calculate_file_allocation(file_distribution)
            if not allocation:
                logger.error("无法计算文件分配方案")
                return False
            
            # 3. 随机选取文件
            selected_files = self.select_random_files(allocation)
            if not selected_files:
                logger.error("没有选中任何文件")
                return False
            
            # 4. 处理文件（复制或剪切）
            success_count, failed_count = self.process_files(selected_files)
            
            # 5. 打印摘要
            self.print_summary(file_distribution, allocation, selected_files, success_count, failed_count)
            
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="随机选取文件并复制或剪切")
    parser.add_argument("source_path", help="源路径")
    parser.add_argument("target_count", type=int, help="目标文件总数")
    parser.add_argument("output_path", help="输出路径")
    parser.add_argument("-o", "--operation", choices=["copy", "cut"], default="copy",
                       help="操作类型: copy(复制) 或 cut(剪切), 默认为copy")
    parser.add_argument("-f", "--flat", action="store_true",
                       help="扁平化输出：不保持目录结构，所有文件直接保存到输出根目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建处理器
        processor = RandomFileProcessor(
            args.source_path, 
            args.target_count, 
            args.output_path, 
            args.operation,
            args.flat
        )
        
        # 执行处理
        success = processor.run()
        
        if success:
            logger.info("文件处理完成")
            return 0
        else:
            logger.error("文件处理失败")
            return 1
            
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        return 1
    except Exception as e:
        logger.error(f"执行处理时出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 