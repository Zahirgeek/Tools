#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件传输脚本
功能：将用户输入的路径下所有子文件剪切或复制到用户输入的输出路径中，并处理同名文件冲突
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from datetime import datetime
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileTransfer:
    """文件传输器"""
    
    def __init__(self, source_path: str, output_path: str, operation: str = "copy", overwrite: bool = False):
        """
        初始化传输器
        
        Args:
            source_path: 源路径
            output_path: 输出路径
            operation: 操作类型 ("copy" 或 "cut")
            overwrite: 是否覆盖同名文件
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.operation = operation.lower()
        self.overwrite = overwrite
        
        if not self.source_path.exists():
            raise ValueError(f"源路径不存在: {source_path}")
        
        if not self.source_path.is_dir():
            raise ValueError(f"源路径不是目录: {source_path}")
        
        if self.operation not in ["copy", "cut"]:
            raise ValueError(f"不支持的操作类型: {operation}，支持的操作: copy, cut")
    
    def get_all_files(self) -> List[Path]:
        """
        获取源路径下所有文件的路径
        
        Returns:
            文件路径列表
        """
        files = []
        try:
            for root, dirs, filenames in os.walk(self.source_path):
                for filename in filenames:
                    file_path = Path(root) / filename
                    files.append(file_path)
        except PermissionError:
            logger.warning(f"无法访问目录: {self.source_path}")
        except Exception as e:
            logger.error(f"获取文件列表时出错: {e}")
        
        return files
    
    def count_files(self) -> int:
        """
        统计源路径下的文件总数
        
        Returns:
            文件总数
        """
        return len(self.get_all_files())
    
    def get_unique_filename(self, target_path: Path) -> Path:
        """
        生成唯一的文件名（避免重名）
        
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
    
    def process_file(self, source_file: Path) -> Tuple[bool, str]:
        """
        处理单个文件
        
        Args:
            source_file: 源文件路径
            
        Returns:
            (是否成功, 消息)
        """
        try:
            # 直接使用文件名，不保持目录结构
            filename = source_file.name
            target_path = self.output_path / filename
            
            # 处理同名文件
            if target_path.exists():
                if self.overwrite:
                    # 覆盖原文件
                    if self.operation == "copy":
                        shutil.copy2(source_file, target_path)
                        return True, f"覆盖复制: {source_file.name}"
                    else:  # cut
                        shutil.move(str(source_file), str(target_path))
                        return True, f"覆盖剪切: {source_file.name}"
                else:
                    # 重命名文件
                    unique_path = self.get_unique_filename(target_path)
                    if self.operation == "copy":
                        shutil.copy2(source_file, unique_path)
                        return True, f"重命名复制: {source_file.name} -> {unique_path.name}"
                    else:  # cut
                        shutil.move(str(source_file), str(unique_path))
                        return True, f"重命名剪切: {source_file.name} -> {unique_path.name}"
            else:
                # 直接复制或剪切
                if self.operation == "copy":
                    shutil.copy2(source_file, target_path)
                    return True, f"复制: {source_file.name}"
                else:  # cut
                    shutil.move(str(source_file), str(target_path))
                    return True, f"剪切: {source_file.name}"
                    
        except Exception as e:
            return False, f"处理文件 {source_file.name} 时出错: {e}"
    
    def run(self) -> Dict[str, int]:
        """
        执行文件传输
        
        Returns:
            处理结果统计
        """
        logger.info(f"开始{self.operation}文件...")
        logger.info(f"源路径: {self.source_path}")
        logger.info(f"输出路径: {self.output_path}")
        logger.info(f"覆盖模式: {'是' if self.overwrite else '否'}")
        
        # 获取所有文件
        files = self.get_all_files()
        if not files:
            logger.warning("源路径下没有找到任何文件")
            return {"total": 0, "success": 0, "failed": 0, "renamed": 0, "overwritten": 0}
        
        logger.info(f"找到 {len(files)} 个文件")
        
        # 创建输出目录
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录已创建: {self.output_path}")
        except Exception as e:
            logger.error(f"创建输出目录时出错: {e}")
            return {"total": len(files), "success": 0, "failed": len(files), "renamed": 0, "overwritten": 0}
        
        # 处理文件
        stats = {
            "total": len(files),
            "success": 0,
            "failed": 0,
            "renamed": 0,
            "overwritten": 0
        }
        
        # 使用tqdm创建进度条
        with tqdm(total=len(files), desc=f"{self.operation.title()}文件", unit="个") as pbar:
            for file_path in files:
                success, message = self.process_file(file_path)
                
                if success:
                    stats["success"] += 1
                    if "重命名" in message:
                        stats["renamed"] += 1
                    elif "覆盖" in message:
                        stats["overwritten"] += 1
                    logger.debug(message)
                else:
                    stats["failed"] += 1
                    logger.error(message)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '成功': stats["success"],
                    '失败': stats["failed"],
                    '重命名': stats["renamed"],
                    '覆盖': stats["overwritten"]
                })
        
        return stats
    
    def print_summary(self, stats: Dict[str, int]):
        """
        打印处理摘要
        
        Args:
            stats: 处理统计
        """
        print("\n" + "="*80)
        print("文件传输摘要")
        print("="*80)
        
        print(f"源路径: {self.source_path}")
        print(f"输出路径: {self.output_path}")
        print(f"操作类型: {self.operation}")
        print(f"覆盖模式: {'是' if self.overwrite else '否'}")
        print()
        
        print("📊 处理统计:")
        print(f"   • 总文件数: {stats['total']}")
        print(f"   • 成功处理: {stats['success']}")
        print(f"   • 处理失败: {stats['failed']}")
        print(f"   • 重命名文件: {stats['renamed']}")
        print(f"   • 覆盖文件: {stats['overwritten']}")
        
        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total'] * 100
            print(f"   • 成功率: {success_rate:.1f}%")
        
        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文件传输脚本 - 复制或剪切文件到指定路径")
    parser.add_argument("source_path", help="源路径")
    parser.add_argument("output_path", help="输出路径")
    parser.add_argument("-o", "--operation", choices=["copy", "cut"], default="copy",
                       help="操作类型: copy(复制) 或 cut(剪切), 默认为copy")
    parser.add_argument("-w", "--overwrite", action="store_true",
                       help="覆盖同名文件，默认为重命名")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建传输器
        transfer = FileTransfer(
            args.source_path,
            args.output_path,
            args.operation,
            args.overwrite
        )
        
        # 执行传输
        stats = transfer.run()
        
        # 打印摘要
        transfer.print_summary(stats)
        
        if stats["failed"] == 0:
            logger.info("文件传输完成")
            return 0
        else:
            logger.warning(f"文件传输完成，但有 {stats['failed']} 个文件处理失败")
            return 1
            
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        return 1
    except Exception as e:
        logger.error(f"执行传输时出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 