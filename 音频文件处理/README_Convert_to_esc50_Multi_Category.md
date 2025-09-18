# ESC-50 多类别音频转换工具

这是一个将多个类别的音频文件转换为ESC-50格式的Python脚本，结合了原有的两个脚本功能。

## 功能特点

- 支持多类别路径输入（-c0, -c1, -c2 等）
- 递归遍历子文件夹查找音频文件
- 自动音频标准化（长度和采样率）
- 生成ESC-50格式的元数据文件
- 支持交互式输入模式
- 支持命令行参数模式

## 安装依赖

```bash
pip install librosa soundfile numpy
```

## 使用方法

### 1. 命令行模式

```bash
# 基本用法 - 使用类别名称
python convert_to_esc50_multi_category.py --正常 "正常音频文件夹" --异常 "异常音频文件夹"

# 多类别示例
python convert_to_esc50_multi_category.py --类别0 "类别0文件夹" --类别1 "类别1文件夹" --类别2 "类别2文件夹"

# 指定输出目录和元数据文件
python convert_to_esc50_multi_category.py --正常 "正常" --异常 "异常" --类别3 "其他" --output-dir "输出文件夹" --meta-file "元数据.csv"

# 自定义音频参数
python convert_to_esc50_multi_category.py --正常 "正常" --异常 "异常" --duration 5 --sample-rate 22050
```

### 2. 交互式模式

```bash
# 直接运行脚本，会进入交互式输入模式
python convert_to_esc50_multi_category.py
```

### 3. 参数说明

- `--类别名称`: 指定类别名称和对应的文件夹路径（支持任意类别名称）
- `--output-dir`: 输出目录（默认: esc50_format）
- `--meta-file`: 元数据文件路径（默认: meta_esc50.csv）
- `--sample-rate`: 目标采样率（默认: 16000Hz）
- `--duration`: 目标音频长度（默认: 10秒）

## 输出格式

### 音频文件命名
转换后的音频文件按照ESC-50格式命名：
```
{fold}-{src_id}-A-{target}.wav
```
- `fold`: 随机分配的fold编号（1-5）
- `src_id`: 源文件ID（格式：类别ID_序号）
- `A`: 固定标识
- `target`: 类别ID

### 元数据文件
生成的CSV文件包含以下列：
- `filename`: 文件名
- `fold`: fold编号
- `target`: 类别ID
- `category`: 类别名称
- `major_category`: 主类别名称
- `src_file`: 源文件ID
- `take`: 固定为'A'

## 示例

假设有以下文件夹结构：
```
音频数据/
├── 正常/
│   ├── 子文件夹1/
│   │   ├── 音频1.wav
│   │   └── 音频2.wav
│   └── 子文件夹2/
│       └── 音频3.wav
└── 异常/
    ├── 异常音频1.wav
    └── 异常音频2.wav
```

运行命令：
```bash
python convert_to_esc50_multi_category.py --正常 "音频数据/正常" --异常 "音频数据/异常"
```

输出结果：
```
esc50_format/
├── 1-0_00000000-A-0.wav
├── 2-0_00000001-A-0.wav
├── 3-0_00000002-A-0.wav
├── 4-1_00000000-A-1.wav
└── 5-1_00000001-A-1.wav
meta_esc50.csv
```

## 注意事项

1. 脚本会递归遍历所有子文件夹查找.wav文件
2. 所有音频文件会被标准化为指定的长度和采样率
3. 如果音频太短会进行填充，太长会进行截断
4. fold编号是随机分配的，但使用固定种子保证可重复性
5. 如果某个类别文件夹不存在或没有音频文件，会显示警告但不会中断处理

## 错误处理

- 文件夹不存在：显示警告并跳过
- 音频文件处理失败：显示错误信息并跳过该文件
- 没有找到任何音频文件：程序退出并显示错误信息
