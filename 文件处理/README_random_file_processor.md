# 随机文件复制/剪切脚本使用说明

## 功能介绍

这个Python脚本可以帮助您：
1. 深入遍历输入文件夹的所有子文件夹
2. 找到最深的含有文件的子文件夹路径
3. 从每个子文件夹中随机选取指定数量的文件
4. 按照指定模式（复制或剪切）处理这些文件
5. 保持原有的目录结构输出到目标文件夹

## 使用方法

### 1. 基本用法

```bash
python random_file_processor.py -i <输入路径> -c <文件数> -m <模式> -o <输出路径>
```

### 2. 命令行参数

#### 必需参数

- `-i, --input-path`: 输入文件夹路径
- `-c, --count`: 每个子文件夹中要随机选取的文件数量
- `-m, --mode`: 处理模式（copy/复制 或 cut/剪切）
- `-o, --output-path`: 输出文件夹路径

#### 可选参数

- `-y, --yes`: 跳过确认提示，直接执行
- `--version`: 显示版本信息
- `-h, --help`: 显示帮助信息

### 3. 支持的模式

- `copy` 或 `复制`: 复制文件到目标文件夹
- `cut` 或 `剪切`: 移动文件到目标文件夹

## 使用示例

### 示例1：基本复制操作
```bash
python random_file_processor.py -i D:\photos -c 5 -m copy -o D:\selected_photos
```

### 示例2：剪切操作（使用中文）
```bash
python random_file_processor.py -i ./input_folder -c 3 -m 剪切 -o ./output_folder
```

### 示例3：跳过确认提示
```bash
python random_file_processor.py -i /path/to/input -c 10 -m copy -o /path/to/output --yes
```

### 示例4：使用完整参数名
```bash
python random_file_processor.py --input-path "C:\Users\photos" --count 5 --mode cut --output-path "C:\Users\output"
```

### 示例5：查看帮助
```bash
python random_file_processor.py -h
```

## 完整的使用流程示例

```bash
# 1. 查看帮助信息
python random_file_processor.py -h

# 2. 执行复制操作
python random_file_processor.py -i D:\photos -c 5 -m copy -o D:\selected_photos

# 输出示例:
==================================================
随机文件复制/剪切脚本
==================================================
操作信息:
输入路径: D:\photos
每个子文件夹选取文件数: 5
处理模式: copy
输出路径: D:\selected_photos
==================================================
是否继续执行？(y/n): y

正在查找最深的文件夹...
找到 8 个最深的文件夹

正在从每个文件夹中随机选择文件...
从 D:\photos\2023\春天 中选择了 5 个文件
从 D:\photos\2023\夏天 中选择了 3 个文件
从 D:\photos\2024\旅行 中选择了 5 个文件
...

总共选择了 25 个文件

正在创建输出目录结构...

开始处理文件...
开始处理文件，总共 25 个文件...
复制: D:\photos\2023\春天\IMG_001.jpg -> D:\selected_photos\2023\春天\IMG_001.jpg
...
进度: 25/25 (100.0%)
处理完成！共处理了 25 个文件

处理完成！
```

## 脚本特点

### 1. 智能文件夹识别
- 自动识别最深的含有文件的子文件夹
- 避免处理空文件夹或只包含子文件夹的中间层级

### 2. 随机选择算法
- 使用Python的 `random.sample()` 方法确保真正的随机选择
- 如果文件夹中的文件数量少于指定数量，则选择所有文件

### 3. 目录结构保持
- 输出文件夹的结构与输入文件夹完全一致
- 保持原有的相对路径关系

### 4. 进度显示
- 实时显示处理进度
- 每处理10个文件或处理完成时显示进度百分比

### 5. 错误处理
- 包含完整的错误处理机制
- 对于无法处理的文件会显示错误信息但继续处理其他文件

### 6. 命令行友好
- 支持长参数名和短参数名
- 提供详细的帮助信息
- 支持跳过确认提示的自动化模式

## 高级用法

### 1. 批处理脚本
创建批处理文件 `batch_process.bat`（Windows）：
```batch
@echo off
python random_file_processor.py -i "C:\Photos\2023" -c 5 -m copy -o "C:\Selected\2023" --yes
python random_file_processor.py -i "C:\Photos\2024" -c 3 -m copy -o "C:\Selected\2024" --yes
pause
```

### 2. Shell脚本
创建Shell脚本 `batch_process.sh`（Linux/macOS）：
```bash
#!/bin/bash
python random_file_processor.py -i "/home/user/photos/2023" -c 5 -m copy -o "/home/user/selected/2023" --yes
python random_file_processor.py -i "/home/user/photos/2024" -c 3 -m copy -o "/home/user/selected/2024" --yes
```

### 3. 配合其他工具使用
```bash
# 先创建输出目录
mkdir -p ./output

# 处理文件
python random_file_processor.py -i ./input -c 10 -m copy -o ./output --yes

# 统计处理结果
find ./output -type f | wc -l
```

## 注意事项

1. **路径格式**：支持绝对路径和相对路径
2. **文件权限**：确保对输入和输出文件夹有足够的读写权限
3. **磁盘空间**：复制模式需要确保有足够的磁盘空间
4. **备份建议**：使用剪切模式前建议备份重要文件
5. **中文支持**：脚本完全支持中文路径和文件名
6. **参数验证**：脚本会自动验证所有输入参数的有效性

## 依赖库

脚本使用的都是Python标准库，无需额外安装：
- `os`：文件系统操作
- `shutil`：文件复制和移动
- `random`：随机选择
- `argparse`：命令行参数解析
- `pathlib`：路径处理
- `typing`：类型提示

## 故障排除

### 常见错误及解决方案：

1. **参数错误**
   ```bash
   # 错误示例
   python random_file_processor.py -i /path/to/input
   # 解决方案：提供所有必需参数
   python random_file_processor.py -i /path/to/input -c 5 -m copy -o /path/to/output
   ```

2. **路径不存在**
   ```
   错误：输入路径不存在：/nonexistent/path
   ```
   - 检查输入路径是否正确
   - 确保路径格式正确（Windows使用反斜杠或双反斜杠）

3. **权限不足**
   - 以管理员身份运行脚本
   - 检查文件夹的读写权限

4. **磁盘空间不足**
   - 检查目标磁盘的可用空间
   - 清理不必要的文件释放空间

5. **文件被占用**
   - 关闭可能正在使用这些文件的程序
   - 等待文件操作完成后再次尝试

### 获取帮助

```bash
# 查看所有参数说明
python random_file_processor.py -h

# 查看版本信息
python random_file_processor.py --version
```
