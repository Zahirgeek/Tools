# 音频转码工具

这个工具用于解决打开音频文件时出现的"Unable to open item for playback (Missing ACM codec)"错误。通过使用FFmpeg将不兼容的音频文件转换为更通用的格式（如MP3），以便在各种播放器中正常播放。

## 使用方法

### Windows用户

1. 双击运行`transcode.bat`批处理文件
   - 默认会将`audio`文件夹中的所有音频文件转换为MP3格式
   - 转换后的文件会保存在`audio_converted`文件夹中

2. 指定转换格式和输入目录：
   ```
   transcode.bat [格式] [输入目录]
   ```
   例如：
   ```
   transcode.bat wav "D:\我的音乐"
   transcode.bat mp3 "C:\Users\用户名\Music"
   ```

### 命令行用户

您可以直接运行Python脚本，支持更多参数：

```
python audio_transcode.py [格式] [输入目录] [输出目录]
```

例如：
```
python audio_transcode.py mp3
python audio_transcode.py wav "D:\我的音乐"
python audio_transcode.py flac "D:\音乐" "E:\转换后的音乐"
```

如果不指定目录，程序会提示您输入。

支持的格式包括：mp3、wav、ogg、m4a、flac、aac等。

## 要求

- Python 3.6+
- FFmpeg (已在本地ffmpeg-7.1.1-essentials_build文件夹中提供，或需要安装在系统PATH中)

## 故障排除

如果遇到问题：

1. 确保已安装Python 3.6或更高版本
2. 确保FFmpeg正确安装或包含在项目目录中
3. 检查音频文件是否存在于指定目录中
4. 检查输出目录是否有写入权限
5. 如果路径中包含空格，请确保用引号括起来

## 常见问题

- Q: 为什么我的音频文件无法播放？
  A: 某些音频文件使用了罕见的编解码器，Windows默认不支持这些编解码器。使用此转码工具可以将这些文件转换为更通用的格式。

- Q: 转换后的文件质量是否会下降？
  A: 转换为有损格式（如MP3）可能会略微降低音质。如果需要保持原始质量，请转换为无损格式如FLAC或WAV。

- Q: 如何批量转换不同目录下的音频文件？
  A: 您可以将所有需要转换的音频文件复制到同一个目录下，然后指定该目录进行转换。或者多次运行程序，每次指定不同的输入目录。

# 音频处理工具集

本仓库包含多个音频处理工具，请参考各工具的专用说明文档：

- [音频合并工具 (merge_audio)](README_MERGE_AUDIO.md) - 用于将多个短音频合并为长音频文件 
- [文件分割工具 (file_splitter)](README_FILE_SPLITTER.md) - 用于将文件夹中的文件分割成多个部分 