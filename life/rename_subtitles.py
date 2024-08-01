"""
字幕重命名脚本,字幕重命名后移动到对应的视频目录中
"""
import os
import re
import argparse

def get_episode_number(filename):
    match = re.search(r'\b(\d{2})\b', filename)
    return match.group(1) if match else None

def rename_subtitles(video_path, subtitle_path):
    video_files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
    subtitle_files = [f for f in os.listdir(subtitle_path) if os.path.isfile(os.path.join(subtitle_path, f))]
    
    video_files.sort()
    subtitle_files.sort()
    
    for video_file in video_files:
        video_episode = get_episode_number(video_file)
        if not video_episode:
            print(f"Skipping {video_file}, no episode number found.")
            continue
        
        for subtitle_file in subtitle_files:
            subtitle_episode = get_episode_number(subtitle_file)
            if video_episode == subtitle_episode:
                video_filename_no_ext = os.path.splitext(video_file)[0]
                subtitle_ext = os.path.splitext(subtitle_file)[1]
                new_subtitle_name = f"{video_filename_no_ext}{subtitle_ext}"
                old_subtitle_path = os.path.join(subtitle_path, subtitle_file)
                new_subtitle_path = os.path.join(video_path, new_subtitle_name)
                os.rename(old_subtitle_path, new_subtitle_path)
                print(f"Renamed: {subtitle_file} -> {new_subtitle_name}")
                break
        else:
            print(f"No matching subtitle found for {video_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rename subtitle files to match video files.")
    parser.add_argument("--video_path", "-v", type=str, help="Path to the video files")
    parser.add_argument("--subtitle_path", "-s", type=str, help="Path to the subtitle files")
    
    args = parser.parse_args()
    
    rename_subtitles(args.video_path, args.subtitle_path)
