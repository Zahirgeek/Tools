#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能音频切分工具
使用多进程并行处理和ffmpeg segment功能大幅提升切分速度
"""

import sys
import argparse
import logging
import subprocess
import multiprocessing as mp
import time
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional


class FastAudioSplitter:
    def __init__(self, ffmpeg_path: str = "ffmpeg.exe", max_workers: int = None):
        """
        初始化高性能音频切分器
        
        Args:
            ffmpeg_path: ffmpeg可执行文件的路径
            max_workers: 最大并行工作进程数，默认为CPU核心数
        """
        self.ffmpeg_path = ffmpeg_path
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # 限制最大8个进程避免过载
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_splitter_fast.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        快速获取音频文件的时长（秒）
        使用ffprobe替代ffmpeg获取更快的结果
        """
        try:
            # 使用ffprobe获取精确时长，比ffmpeg -i更快
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                audio_path
            ]
            
            # 如果ffprobe不存在，回退到原方法
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
                    
            except FileNotFoundError:
                # ffprobe不存在，使用原方法
                pass
                
            # 回退方法：使用ffmpeg
            cmd = [
                self.ffmpeg_path,
                '-i', audio_path,
                '-f', 'null',
                '-'
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # 从ffmpeg输出中解析时长
            output = result.stdout
            for line in output.split('\n'):
                if 'Duration:' in line:
                    duration_str = line.split('Duration:')[1].split(',')[0].strip()
                    time_parts = duration_str.split(':')
                    if len(time_parts) == 3:
                        hours = float(time_parts[0])
                        minutes = float(time_parts[1])
                        seconds = float(time_parts[2])
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                        return total_seconds
                        
        except Exception as e:
            self.logger.error(f"获取音频时长失败 {audio_path}: {e}")
            return None
            
        return None
        
    def split_audio_fast(self, input_path: str, output_dir: str, segment_duration: float) -> List[str]:
        """
        使用ffmpeg segment功能快速切分音频文件
        一次调用切分所有片段，比多次调用快很多
        """
        input_file = Path(input_path)
        output_path = Path(output_dir)
        
        # 确保输出目录存在
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取音频时长
        duration = self.get_audio_duration(input_path)
        if duration is None:
            return []
            
        # 如果音频时长不足指定时长，则不进行切分
        if duration <= segment_duration:
            return []
            
        # 计算需要切分的段数
        num_segments = int(duration // segment_duration)
        
        # 生成输出文件模板
        output_template = str(output_path / f"{input_file.stem}_%03d{input_file.suffix}")
        
        # 使用ffmpeg segment功能一次性切分所有片段
        cmd = [
            self.ffmpeg_path,
            '-i', str(input_path),
            '-f', 'segment',
            '-segment_time', str(segment_duration),
            '-segment_format', input_file.suffix[1:],  # 去掉点号
            '-c', 'copy',  # 复制编码，保持原有格式和质量
            '-avoid_negative_ts', 'make_zero',
            '-reset_timestamps', '1',
            output_template,
            '-y'  # 覆盖已存在的文件
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            if result.returncode == 0:
                # 收集生成的文件
                generated_files = []
                for i in range(num_segments):
                    expected_file = output_path / f"{input_file.stem}_{i:03d}{input_file.suffix}"
                    if expected_file.exists():
                        generated_files.append(str(expected_file))
                        
                return generated_files
            else:
                self.logger.error(f"快速切分失败 {input_file.name}: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"执行快速切分命令失败 {input_file.name}: {e}")
            return []
            
    def process_single_file(self, file_info: Tuple[str, str, float]) -> Tuple[str, int, bool]:
        """
        处理单个音频文件（用于多进程）
        
        Args:
            file_info: (input_path, output_dir, segment_duration)
            
        Returns:
            (filename, generated_count, success)
        """
        input_path, output_dir, segment_duration = file_info
        input_file = Path(input_path)
        
        try:
            generated_files = self.split_audio_fast(input_path, output_dir, segment_duration)
            
            if generated_files:
                return (input_file.name, len(generated_files), True)
            else:
                return (input_file.name, 0, False)
                
        except Exception as e:
            self.logger.error(f"处理文件失败 {input_file.name}: {e}")
            return (input_file.name, 0, False)
            
    def find_audio_files(self, root_dir: str) -> List[str]:
        """递归查找目录中的音频文件"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        audio_files = []
        
        root_path = Path(root_dir)
        if not root_path.exists():
            self.logger.error(f"目录不存在: {root_dir}")
            return []
            
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
                
        self.logger.info(f"在 {root_dir} 中找到 {len(audio_files)} 个音频文件")
        return audio_files
        
    def create_output_structure(self, input_dir: str, output_dir: str, audio_file: str) -> str:
        """创建与输入目录结构对应的输出目录结构"""
        input_path = Path(input_dir).resolve()
        audio_path = Path(audio_file).resolve()
        output_path = Path(output_dir).resolve()
        
        try:
            relative_path = audio_path.parent.relative_to(input_path)
            target_output_dir = output_path / relative_path
            return str(target_output_dir)
        except ValueError:
            return str(output_path)
            
    def process_directory_parallel(self, input_dir: str, output_dir: str, segment_duration: float):
        """
        并行处理整个目录
        """
        start_time = time.time()
        
        self.logger.info(f"开始并行处理目录: {input_dir}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"分段时长: {segment_duration} 秒")
        self.logger.info(f"并行进程数: {self.max_workers}")
        
        # 查找所有音频文件
        audio_files = self.find_audio_files(input_dir)
        
        if not audio_files:
            self.logger.warning("没有找到音频文件")
            return
            
        # 准备任务列表
        tasks = []
        for audio_file in audio_files:
            target_output_dir = self.create_output_structure(input_dir, output_dir, audio_file)
            tasks.append((audio_file, target_output_dir, segment_duration))
            
        self.logger.info(f"准备处理 {len(tasks)} 个文件...")
        
        # 使用进程池并行处理
        total_processed = 0
        total_generated = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(process_file_worker, task, self.ffmpeg_path): task 
                for task in tasks
            }
            
            # 收集结果
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    filename, generated_count, success = future.result()
                    completed += 1
                    
                    if success and generated_count > 0:
                        total_processed += 1
                        total_generated += generated_count
                        self.logger.info(f"[{completed}/{len(tasks)}] ✅ {filename}: 生成 {generated_count} 个片段")
                    else:
                        self.logger.info(f"[{completed}/{len(tasks)}] ⚠️  {filename}: 跳过（时长不足或处理失败）")
                        
                except Exception as e:
                    completed += 1
                    task = future_to_task[future]
                    filename = Path(task[0]).name
                    self.logger.error(f"[{completed}/{len(tasks)}] ❌ {filename}: 处理异常 - {e}")
                    
        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 并行处理完成!")
        self.logger.info(f"📊 处理统计:")
        self.logger.info(f"   • 成功处理文件: {total_processed}/{len(audio_files)}")
        self.logger.info(f"   • 生成片段总数: {total_generated}")
        self.logger.info(f"   • 总耗时: {total_time:.1f} 秒")
        self.logger.info(f"   • 平均速度: {len(audio_files)/total_time:.1f} 文件/秒")
        if total_generated > 0:
            self.logger.info(f"   • 片段生成速度: {total_generated/total_time:.1f} 片段/秒")
        self.logger.info("=" * 60)


def process_file_worker(file_info: Tuple[str, str, float], ffmpeg_path: str) -> Tuple[str, int, bool]:
    """
    工作进程函数（必须在全局范围内定义以支持multiprocessing）
    """
    input_path, output_dir, segment_duration = file_info
    
    # 创建临时的splitter实例
    class TempSplitter:
        def __init__(self, ffmpeg_path):
            self.ffmpeg_path = ffmpeg_path
            
        def get_audio_duration(self, audio_path: str) -> Optional[float]:
            try:
                # 尝试ffprobe
                cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_path]
                try:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         text=True, encoding='utf-8', errors='ignore', timeout=5,
                                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                    if result.returncode == 0 and result.stdout.strip():
                        return float(result.stdout.strip())
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                    
                # 回退到ffmpeg
                cmd = [self.ffmpeg_path, '-i', audio_path, '-f', 'null', '-']
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='ignore', timeout=10,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                
                for line in result.stdout.split('\n'):
                    if 'Duration:' in line:
                        duration_str = line.split('Duration:')[1].split(',')[0].strip()
                        time_parts = duration_str.split(':')
                        if len(time_parts) == 3:
                            hours = float(time_parts[0])
                            minutes = float(time_parts[1])
                            seconds = float(time_parts[2])
                            return hours * 3600 + minutes * 60 + seconds
            except:
                pass
            return None
            
        def split_audio_fast(self, input_path: str, output_dir: str, segment_duration: float) -> List[str]:
            input_file = Path(input_path)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            duration = self.get_audio_duration(input_path)
            if duration is None or duration <= segment_duration:
                return []
                
            num_segments = int(duration // segment_duration)
            output_template = str(output_path / f"{input_file.stem}_%03d{input_file.suffix}")
            
            cmd = [
                self.ffmpeg_path, '-i', str(input_path), '-f', 'segment',
                '-segment_time', str(segment_duration), '-segment_format', input_file.suffix[1:],
                '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-reset_timestamps', '1',
                output_template, '-y'
            ]
            
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding='utf-8', errors='ignore', timeout=60,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                
                if result.returncode == 0:
                    generated_files = []
                    for i in range(num_segments):
                        expected_file = output_path / f"{input_file.stem}_{i:03d}{input_file.suffix}"
                        if expected_file.exists():
                            generated_files.append(str(expected_file))
                    return generated_files
            except:
                pass
            return []
    
    splitter = TempSplitter(ffmpeg_path)
    input_file = Path(input_path)
    
    try:
        generated_files = splitter.split_audio_fast(input_path, output_dir, segment_duration)
        if generated_files:
            return (input_file.name, len(generated_files), True)
        else:
            return (input_file.name, 0, False)
    except Exception:
        return (input_file.name, 0, False)


def main():
    parser = argparse.ArgumentParser(description='高性能音频切分工具')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('-d', '--duration', type=float, default=2.0,
                       help='分段时长（秒），默认为2秒')
    parser.add_argument('-j', '--workers', type=int, default=None,
                       help='并行工作进程数，默认为CPU核心数（最多8个）')
    parser.add_argument('--ffmpeg', default='ffmpeg.exe',
                       help='ffmpeg可执行文件路径，默认为当前目录下的ffmpeg.exe')
    
    args = parser.parse_args()
    
    # 检查ffmpeg是否存在
    ffmpeg_path = Path(args.ffmpeg)
    if not ffmpeg_path.exists():
        print(f"错误: ffmpeg文件不存在: {args.ffmpeg}")
        print("请确保ffmpeg.exe在当前目录下，或使用--ffmpeg参数指定正确路径")
        sys.exit(1)
        
    # 创建高性能音频切分器并开始处理
    splitter = FastAudioSplitter(str(ffmpeg_path), max_workers=args.workers)
    splitter.process_directory_parallel(args.input_dir, args.output_dir, args.duration)


if __name__ == '__main__':
    main() 