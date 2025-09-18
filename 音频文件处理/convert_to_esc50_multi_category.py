import os
import csv
import shutil
import random
import numpy as np
import glob
import argparse
import sys
from pathlib import Path
import librosa
import soundfile as sf

def normalize_audio_length(audio_path, target_sr=16000, target_length_sec=10):
    """
    将音频文件标准化为指定长度
    
    参数:
        audio_path: 音频文件路径
        target_sr: 目标采样率
        target_length_sec: 目标长度(秒)
    
    返回:
        标准化后的音频数据
    """
    # 计算目标采样点数
    target_length = target_sr * target_length_sec
    
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=target_sr)
    
    # 标准化长度
    if len(y) < target_length:
        # 如果音频太短，使用填充
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    elif len(y) > target_length:
        # 如果音频太长，进行截断
        y = y[:target_length]
        
    return y, target_sr

def find_audio_files_recursively(folder_path):
    """
    递归查找文件夹中的所有音频文件
    
    参数:
        folder_path: 文件夹路径
    
    返回:
        音频文件路径列表
    """
    audio_files = []
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"警告: 文件夹'{folder_path}'不存在或不是一个有效的目录")
        return audio_files
    
    # 查找所有wav文件（递归）
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def process_category_files(category_paths, target_dir, meta_file, id_to_name_mapping, target_sr=16000, target_length_sec=10):
    """
    处理多个类别的音频文件
    
    参数:
        category_paths: 类别路径字典 {category_id: path}
        target_dir: 目标目录
        meta_file: 元数据文件路径
        id_to_name_mapping: 类别ID到名称的映射 {category_id: category_name}
        target_sr: 目标采样率
        target_length_sec: 目标音频长度
    """
    # 确保目标路径存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 收集所有文件信息
    all_files = []
    
    # 处理每个类别
    for category_id, folder_path in category_paths.items():
        print(f"处理类别 {category_id}: {folder_path}")
        
        # 递归查找音频文件
        audio_files = find_audio_files_recursively(folder_path)
        
        if not audio_files:
            print(f"  警告: 类别 {category_id} 的文件夹中没有找到wav文件")
            continue
        
        print(f"  找到 {len(audio_files)} 个音频文件")
        
        # 按文件名排序，确保编号的顺序性
        audio_files.sort()
        
        # 为每个文件创建信息
        for i, audio_file in enumerate(audio_files):
            # 提取原始文件名
            original_filename = os.path.basename(audio_file)
            
            # 获取类别名称
            category_name = id_to_name_mapping.get(category_id, f'category_{category_id}')
            
            # 创建文件信息
            file_info = {
                'path': audio_file,
                'original_filename': original_filename,
                'target': category_id,
                'category': category_name,
                'major_category': category_name,
                'src_id': f"{category_id}_{i:08d}"  # 生成源文件ID
            }
            all_files.append(file_info)
    
    if not all_files:
        print("错误: 没有找到任何音频文件")
        return False
    
    # 随机分配fold (1-5)
    random.seed(42)  # 保证可重复性
    for file_info in all_files:
        file_info['fold'] = random.randint(1, 5)
    
    # 写入元数据CSV文件
    with open(meta_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'fold', 'target', 'category', 'major_category', 'src_file', 'take'])
        
        # 复制文件并写入元数据
        for file_info in all_files:
            # 创建新的ESC-50格式文件名
            new_filename = f"{file_info['fold']}-{file_info['src_id']}-A-{file_info['target']}.wav"
            target_path = os.path.join(target_dir, new_filename)
            
            # 处理音频长度标准化
            try:
                # 标准化音频长度
                audio_normalized, sr = normalize_audio_length(
                    file_info['path'],
                    target_sr=target_sr,
                    target_length_sec=target_length_sec
                )
                
                # 保存标准化后的音频
                sf.write(target_path, audio_normalized, sr)
                print(f"  已处理: {os.path.basename(file_info['path'])} -> {new_filename}")
            except Exception as e:
                print(f"  处理文件时出错 {file_info['path']}: {str(e)}")
                continue
            
            # 写入元数据
            writer.writerow([
                new_filename, 
                file_info['fold'], 
                file_info['target'],
                file_info['category'],
                file_info['major_category'],
                file_info['src_id'],
                'A'  # 所有文件都使用A作为take标识
            ])
    
    return True

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将多个类别的音频文件转换为ESC-50格式',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 基本参数
    parser.add_argument(
        '--output-dir',
        default='esc50_format',
        help='输出目录，默认为"esc50_format"'
    )
    
    parser.add_argument(
        '--meta-file',
        default='meta_esc50.csv',
        help='元数据文件路径，默认为"meta_esc50.csv"'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='目标采样率，默认为16000Hz'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='目标音频长度(秒)，默认为10秒'
    )
    
    # 添加示例说明
    parser.epilog = '''
示例:
  python convert_to_esc50_multi_category.py --正常 "正常音频文件夹" --异常 "异常音频文件夹"
  python convert_to_esc50_multi_category.py --类别0 "类别0文件夹" --类别1 "类别1文件夹" --类别2 "类别2文件夹"
  python convert_to_esc50_multi_category.py --正常 "正常" --异常 "异常" --duration 5 --sample-rate 22050
'''
    
    # 解析已知参数
    args, unknown = parser.parse_known_args()
    
    # 解析类别参数 - 支持任意类别名称
    category_paths = {}
    category_names = []
    
    # 从unknown参数中提取类别信息
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--') and i + 1 < len(unknown):
            category_name = arg[2:]  # 去掉--前缀
            folder_path = unknown[i + 1]
            category_paths[category_name] = folder_path
            category_names.append(category_name)
            i += 2
        else:
            i += 1
    
    # 为每个类别分配数字ID
    category_id_mapping = {}
    for idx, category_name in enumerate(category_names):
        category_id_mapping[category_name] = idx
    
    return args, category_paths, category_id_mapping

def interactive_input():
    """交互式输入模式"""
    print("=== 交互式输入模式 ===")
    category_paths = {}
    
    while True:
        print("\n请输入类别信息 (输入空行结束):")
        category_name = input("类别名称: ").strip()
        if not category_name:
            break
            
        folder_path = input(f"类别 '{category_name}' 的文件夹路径: ").strip()
        if folder_path:
            category_paths[category_name] = folder_path
            print(f"已添加类别 '{category_name}': {folder_path}")
    
    if not category_paths:
        print("没有输入任何类别，程序退出")
        return None, None, None, None, None
    
    output_dir = input("输出目录 (直接回车使用默认值'esc50_format'): ").strip()
    if not output_dir:
        output_dir = 'esc50_format'
    
    meta_file = input("元数据文件路径 (直接回车使用默认值'meta_esc50.csv'): ").strip()
    if not meta_file:
        meta_file = 'meta_esc50.csv'
    
    sample_rate = input("采样率 (直接回车使用默认值16000): ").strip()
    if not sample_rate:
        sample_rate = 16000
    else:
        try:
            sample_rate = int(sample_rate)
        except ValueError:
            print("无效的采样率，使用默认值16000")
            sample_rate = 16000
    
    duration = input("音频长度(秒) (直接回车使用默认值10): ").strip()
    if not duration:
        duration = 10
    else:
        try:
            duration = int(duration)
        except ValueError:
            print("无效的音频长度，使用默认值10")
            duration = 10
    
    return category_paths, output_dir, meta_file, sample_rate, duration

def main():
    """主函数"""
    # 首先尝试从命令行参数获取信息
    args, category_paths, category_id_mapping = parse_arguments()
    
    # 如果命令行没有提供类别信息，使用交互式输入
    if not category_paths:
        if len(sys.argv) > 1:  # 用户提供了一些参数但没有类别
            print("错误: 请提供至少一个类别路径，使用 --类别名 \"路径\" 格式")
            print("例如: python script.py --正常 \"正常音频\" --异常 \"异常音频\"")
            return
        
        result = interactive_input()
        if result[0] is None:
            return
        category_paths, output_dir, meta_file, sample_rate, duration = result
        # 为交互式输入创建ID映射
        category_id_mapping = {name: idx for idx, name in enumerate(category_paths.keys())}
    else:
        output_dir = args.output_dir
        meta_file = args.meta_file
        sample_rate = args.sample_rate
        duration = args.duration
    
    print(f"开始处理 {len(category_paths)} 个类别的音频文件...")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {meta_file}")
    print(f"采样率: {sample_rate}Hz, 音频长度: {duration}秒")
    print("-" * 50)
    
    # 显示类别映射信息
    print("类别映射:")
    for category_name, category_id in category_id_mapping.items():
        print(f"  {category_name} -> ID {category_id}")
    print("-" * 50)
    
    # 将类别名称映射转换为ID映射
    category_id_paths = {}
    # 创建ID到名称的反向映射
    id_to_name_mapping = {v: k for k, v in category_id_mapping.items()}
    
    for category_name, folder_path in category_paths.items():
        category_id = category_id_mapping[category_name]
        category_id_paths[category_id] = folder_path
    
    # 处理文件
    success = process_category_files(
        category_id_paths, 
        output_dir, 
        meta_file, 
        id_to_name_mapping,
        target_sr=sample_rate, 
        target_length_sec=duration
    )
    
    if success:
        # 统计信息
        category_counts = {}
        for category_id in category_id_paths.keys():
            category_counts[category_id] = 0
        
        # 统计每个类别的文件数量
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    target = int(row['target'])
                    if target in category_counts:
                        category_counts[target] += 1
        
        print("\n" + "=" * 50)
        print("转换完成!")
        print(f"共处理了 {sum(category_counts.values())} 个文件")
        
        # 显示类别名称而不是ID
        for category_name, category_id in category_id_mapping.items():
            count = category_counts.get(category_id, 0)
            print(f"类别 '{category_name}' (ID {category_id}): {count} 个文件")
        
        print(f"所有音频文件已标准化为 {duration} 秒, 采样率 {sample_rate} Hz")
        print(f"元数据已保存到: {meta_file}")
    else:
        print("转换失败!")

if __name__ == "__main__":
    main()
