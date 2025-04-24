import os
import datetime
import re
from pydub import AudioSegment
import shutil
from collections import defaultdict
import argparse

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='将短音频文件合并为长音频文件')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='输入音频文件夹路径')
    parser.add_argument('-o', '--output_dir', type=str, default='audio_merge', help='输出音频文件夹路径，默认为audio_merge')
    return parser.parse_args()

def main():
    # 获取命令行参数
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有WAV文件
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    # 解析文件名中的时间戳并排序
    def parse_timestamp(filename):
        match = re.match(r'(\d{8})_(\d{6})\.wav', filename)
        if match:
            date_str, time_str = match.groups()
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            return datetime.datetime(year, month, day, hour, minute, second)
        return None

    # 按日期分组
    files_by_date = defaultdict(list)
    for file in wav_files:
        timestamp = parse_timestamp(file)
        if timestamp:
            date_str = timestamp.strftime('%Y%m%d')
            files_by_date[date_str].append((file, timestamp))

    # 为每个日期处理文件
    for date, files_with_timestamps in files_by_date.items():
        # 按时间戳排序
        sorted_files = sorted(files_with_timestamps, key=lambda x: x[1])
        
        # 分组合并文件
        groups = []
        current_group = []
        
        for i, (file, timestamp) in enumerate(sorted_files):
            # 如果是第一个文件或者与前一个文件的时间差不超过1分钟
            if (i == 0 or 
                (timestamp - sorted_files[i-1][1]).total_seconds() <= 60):
                current_group.append((file, timestamp))
            else:
                # 当前文件与前一个文件时间差太大，开始新的分组
                if current_group:
                    groups.append(current_group)
                current_group = [(file, timestamp)]
            
            # 如果当前组已有5个文件，将其加入分组列表并重新开始
            if len(current_group) == 5:
                groups.append(current_group)
                current_group = []
        
        # 处理最后一个不完整的组（如果有）
        if current_group and len(current_group) >= 2:  # 至少2个文件才合并
            groups.append(current_group)
        
        # 合并每个组中的文件
        for group_index, group in enumerate(groups):
            if len(group) < 2:  # 跳过单个文件
                continue
                
            # 获取组的起始和结束时间戳用于命名
            start_time = group[0][1].strftime('%H%M%S')
            end_time = group[-1][1].strftime('%H%M%S')
            output_filename = f"{date}_{start_time}_to_{end_time}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # 合并音频
            combined = AudioSegment.empty()
            for file, _ in group:
                audio_path = os.path.join(input_dir, file)
                audio = AudioSegment.from_wav(audio_path)
                combined += audio
            
            # 检查合并后的音频长度是否大于等于10秒（10000毫秒）
            if len(combined) >= 10000:
                # 保存合并后的文件
                combined.export(output_path, format="wav")
                print(f"已合并 {len(group)} 个文件，长度 {len(combined)/1000:.1f}秒: {output_filename}")
            else:
                print(f"跳过 {len(group)} 个文件的合并，长度不足10秒 ({len(combined)/1000:.1f}秒)")

    print("处理完成，合并后的文件保存在", output_dir, "文件夹中")

if __name__ == "__main__":
    main() 