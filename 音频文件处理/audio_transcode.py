#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path

def transcode_audio(input_dir, output_dir, target_format="mp3"):
    """
    将指定目录下的所有音频文件转换为目标格式
    
    参数:
        input_dir: 输入音频文件目录
        output_dir: 输出音频文件目录
        target_format: 目标音频格式，默认为mp3
    """
    # 确保输入目录存在
    if not Path(input_dir).exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return False
        
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取ffmpeg路径
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        print("错误: 找不到ffmpeg。请确保ffmpeg已安装并在PATH中，或在当前目录的ffmpeg-7.1.1-essentials_build文件夹中。")
        return False
    
    # 获取输入目录中的所有音频文件
    input_files = []
    for ext in ['.wav', '.mp3', '.ogg', '.m4a', '.flac', '.aac']:
        input_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    if not input_files:
        print(f"警告: 在 {input_dir} 中没有找到音频文件")
        return False
    
    # 转换每个文件
    success_count = 0
    for input_file in input_files:
        output_file = Path(output_dir) / f"{input_file.stem}.{target_format}"
        
        try:
            # 构建ffmpeg命令
            cmd = [
                ffmpeg_path, 
                "-i", str(input_file), 
                "-acodec", get_codec_for_format(target_format), 
                "-y",  # 覆盖已存在的文件
                str(output_file)
            ]
            
            # 执行命令并捕获输出
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 检查命令是否成功执行
            if result.returncode == 0:
                print(f"成功: {input_file.name} -> {output_file.name}")
                success_count += 1
            else:
                print(f"错误: 转换 {input_file.name} 失败")
                print(f"错误详情: {result.stderr}")
        except Exception as e:
            print(f"错误: 处理 {input_file.name} 时发生异常: {str(e)}")
    
    print(f"\n转换完成: {success_count}/{len(input_files)} 个文件成功转换")
    return success_count > 0

def get_codec_for_format(format):
    """根据目标格式返回合适的编解码器"""
    codecs = {
        "mp3": "libmp3lame",
        "aac": "aac",
        "ogg": "libvorbis",
        "flac": "flac",
        "wav": "pcm_s16le",
        "m4a": "aac"
    }
    return codecs.get(format.lower(), "copy")

def get_ffmpeg_path():
    """尝试找到ffmpeg可执行文件的路径"""
    # 检查是否在PATH中
    if sys.platform.startswith('win'):
        ffmpeg_cmd = "ffmpeg.exe"
    else:
        ffmpeg_cmd = "ffmpeg"
    
    # 方法1: 检查PATH中是否有ffmpeg
    try:
        result = subprocess.run(
            [ffmpeg_cmd, "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return ffmpeg_cmd
    except:
        pass
    
    # 方法2: 检查当前目录下的ffmpeg文件夹
    current_dir = Path(__file__).parent
    ffmpeg_dir = current_dir / "ffmpeg-7.1.1-essentials_build" / "bin"
    ffmpeg_path = ffmpeg_dir / ffmpeg_cmd
    
    if ffmpeg_path.exists():
        return str(ffmpeg_path)
    
    # 方法3: 检查上级目录的ffmpeg文件夹
    parent_dir = current_dir.parent
    ffmpeg_dir = parent_dir / "ffmpeg-7.1.1-essentials_build" / "bin"
    ffmpeg_path = ffmpeg_dir / ffmpeg_cmd
    
    if ffmpeg_path.exists():
        return str(ffmpeg_path)
    
    return None

def main():
    """主函数"""
    # 设置默认的输入和输出目录
    script_dir = Path(__file__).parent
    default_input_dir = script_dir / "audio"
    default_output_dir = script_dir / "audio_converted"
    
    # 默认转换为MP3格式
    target_format = "mp3"
    input_dir = default_input_dir
    output_dir = default_output_dir
    
    # 解析命令行参数
    argc = len(sys.argv)
    if argc > 1:
        # 第一个参数是输出格式
        target_format = sys.argv[1]
    
    if argc > 2:
        # 第二个参数是输入目录
        input_dir = Path(sys.argv[2])
    
    if argc > 3:
        # 第三个参数是输出目录
        output_dir = Path(sys.argv[3])
    
    # 如果用户没有指定输入目录，提示输入
    if argc <= 2:
        print("请输入音频文件所在目录(按回车使用默认目录 %s):" % default_input_dir)
        user_input = input().strip()
        if user_input:
            input_dir = Path(user_input)
    
    print(f"开始将 {input_dir} 中的音频文件转换为 {target_format} 格式...")
    print(f"转换后的文件将保存到 {output_dir}")
    
    # 执行转码
    success = transcode_audio(input_dir, output_dir, target_format)
    
    if success:
        print(f"转换成功! 转换后的文件保存在: {output_dir}")
    else:
        print("转换失败，请检查错误信息。")

if __name__ == "__main__":
    main() 