#音乐文件按音乐家/专辑分类,需要安装mutagen
import argparse
import os
import shutil
from mutagen import File

# 支持的音频文件扩展名
SUPPORTED_FORMATS = ('.flac', '.mp3', '.m4a', '.aac', '.wav')

def get_args():
    parser = argparse.ArgumentParser(description="音乐文件按音乐家/专辑分类")
    parser.add_argument("--source_directory", "-s", type=str, required=True,
                        help="音乐文件位置")
    parser.add_argument("--target_directory", "-t", type=str, required=True,
                        help="输出位置")


    args = parser.parse_args()
    return args

def sanitize_filename(filename, max_length=100):
    """替换不被支持的字符为下划线，并截断文件名"""
    sanitized = "".join(c if c not in r'\/:*?"<>|' else '_' for c in filename)
    words = sanitized.split()
    if len(sanitized) > max_length:
        # 取前两个完整的单词或文字
        sanitized = " ".join(words[:2])
    return sanitized


def get_audio_metadata(file_path):
    """获取音频文件的元数据"""
    audio = File(file_path, easy=True)
    if audio is None:
        return None, None
    artist = audio.get('artist', ['Unknown Artist'])[0]
    album = audio.get('album', ['Unknown Album'])[0]
    return artist, album


def organize_files(source_dir, target_dir):
    """根据元数据组织音频文件"""
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(root, file)
                artist, album = get_audio_metadata(file_path)

                # 清理文件名
                artist = sanitize_filename(artist)
                album = sanitize_filename(album)

                # 创建目标目录
                artist_dir = os.path.join(target_dir, artist)
                album_dir = os.path.join(artist_dir, album)
                os.makedirs(album_dir, exist_ok=True)

                # 移动文件
                target_path = os.path.join(album_dir, file)
                shutil.copy(file_path, target_path)
                print(f'Moved: {file_path} -> {target_path}')


if __name__ == "__main__":
    args = get_args()
    source_directory = args.source_directory
    target_directory = args.target_directory
    organize_files(source_directory, target_directory)