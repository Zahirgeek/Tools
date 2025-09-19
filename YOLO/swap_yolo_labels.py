#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# 可选进度条支持
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def read_classes(classes_path: Path) -> List[str]:
    with open(classes_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_classes(classes_path: Path, classes: List[str]) -> None:
    with open(classes_path, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")


def backup_file(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + '.bak')
    try:
        if not backup_path.exists():
            backup_path.write_bytes(path.read_bytes())
    except Exception:
        pass
    return backup_path


def find_label_files(paths: List[Path]) -> List[Path]:
    label_files: List[Path] = []
    for p in paths:
        if p.is_file():
            if p.suffix.lower() == '.txt':
                label_files.append(p)
        elif p.is_dir():
            for txt in p.rglob('*.txt'):
                label_files.append(txt)
    # 去重并保持顺序
    seen = set()
    unique_files = []
    for f in label_files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def swap_indices_in_line(line: str, from_idx: int, to_idx: int) -> str:
    line = line.strip()
    if not line:
        return line
    parts = line.split()
    try:
        cls_id = int(parts[0])
    except Exception:
        return line
    if cls_id == from_idx:
        parts[0] = str(to_idx)
    elif cls_id == to_idx:
        parts[0] = str(from_idx)
    return ' '.join(parts)


def process_label_file(label_file: Path, from_idx: int, to_idx: int, do_backup: bool = False) -> bool:
    try:
        content = label_file.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return False
    changed = False
    new_lines: List[str] = []
    for line in content:
        new_line = swap_indices_in_line(line, from_idx, to_idx)
        if new_line != line:
            changed = True
        new_lines.append(new_line)
    if changed:
        if do_backup:
            backup_file(label_file)
        label_file.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
    return True


def progress_iter(iterable, desc: str):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, unit='file')
    return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='调换YOLO标签顺序，同时更新classes.txt与标签文件中的类别索引',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-c', '--classes', required=True, help='classes.txt 路径')
    parser.add_argument('-l', '--labels', required=True, nargs='+', help='一个或多个标签文件路径，或标签目录路径（将递归处理*.txt）')
    parser.add_argument('-f', '--from-index', required=True, type=int, help='源类别索引（将与目标索引互换）')
    parser.add_argument('-t', '--to-index', required=True, type=int, help='目标类别索引（将与源索引互换）')
    parser.add_argument('--dry-run', action='store_true', help='试运行，仅打印将要进行的修改，不写入文件')
    parser.add_argument('--backup', action='store_true', help='启用备份：写入前为classes与每个标签文件创建.bak备份（默认关闭）')
    return parser.parse_args()


def main():
    args = parse_args()

    classes_path = Path(args.classes)
    if not classes_path.exists():
        print(f"错误: classes文件不存在: {classes_path}")
        sys.exit(1)

    label_inputs = [Path(p) for p in args.labels]
    for p in label_inputs:
        if not p.exists():
            print(f"错误: 标签路径不存在: {p}")
            sys.exit(1)

    # 读取classes并校验索引
    classes = read_classes(classes_path)
    num_classes = len(classes)
    f_idx = args.from_index
    t_idx = args.to_index
    if f_idx < 0 or f_idx >= num_classes or t_idx < 0 or t_idx >= num_classes:
        print(f"错误: 索引超出范围。classes共 {num_classes} 类，收到 from={f_idx}, to={t_idx}")
        sys.exit(1)
    if f_idx == t_idx:
        print("提示: 源与目标索引相同，无需修改。")
        sys.exit(0)

    # 预览classes交换
    new_classes = classes.copy()
    new_classes[f_idx], new_classes[t_idx] = new_classes[t_idx], new_classes[f_idx]

    print("将交换 classes.txt 中的类别:")
    print(f"  {f_idx}: {classes[f_idx]} <-> {t_idx}: {classes[t_idx]}")

    # 收集所有标签文件
    label_files = find_label_files(label_inputs)
    print(f"将处理标签文件数量: {len(label_files)}")

    if args.dry_run:
        print("dry-run: 不会写入任何文件。")
        return

    # 写入classes（可选备份）
    if args.backup:
        backup_file(classes_path)
    write_classes(classes_path, new_classes)

    # 处理每个label文件（带进度）
    updated = 0
    for lf in progress_iter(label_files, desc='Updating label files'):
        if process_label_file(lf, f_idx, t_idx, do_backup=args.backup):
            updated += 1
    print(f"完成。已更新标签文件: {updated}/{len(label_files)}")


if __name__ == '__main__':
    main()
