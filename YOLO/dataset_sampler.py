#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 进度条（缺失时优雅降级）
try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(iterable, **kwargs):
        return iterable


def _generate_unique_filename(
    dest_dir: Path, base_name: str, used_names: Optional[set] = None
) -> str:
    """
    在 dest_dir 下为 base_name 生成不重复的文件名；如存在同名，则追加 _0001, _0002 ...。
    used_names 用于同一批次内的冲突跟踪（避免尚未落盘的重名）。
    返回最终的文件名（仅文件名，不含目录）。
    """
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    candidate = base_name
    counter = 1
    if used_names is None:
        used_names = set()
    # 统一大小写跟踪，尽量兼容大小写不敏感文件系统
    key = candidate.lower()
    while (dest_dir / candidate).exists() or key in used_names:
        candidate = f"{stem}_{counter:04d}{suffix}"
        key = candidate.lower()
        counter += 1
    used_names.add(key)
    return candidate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTENSION = ".txt"


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    label_path: Path
    # 相对路径：相对于 images_root 的相对路径（用于在目标处保留结构）
    relative_path_from_images_root: Path
    # 分组名：images_root 下的一级子目录名；若文件直接在根下，则为 "."
    top_level_group: str


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


@dataclass
class ProcessStats:
    num_samples: int = 0
    num_images: int = 0
    num_labels: int = 0
    num_samples_train: int = 0
    num_samples_val: int = 0


def find_all_pairs(images_root: Path, labels_root: Path) -> List[SamplePair]:
    pairs: List[SamplePair] = []
    for image_file in _tqdm(
        images_root.rglob("*"), desc="扫描图片", unit="文件", leave=False
    ):
        if not image_file.is_file():
            continue
        if not is_image_file(image_file):
            continue

        relative = image_file.relative_to(images_root)
        label_relative = relative.with_suffix(LABEL_EXTENSION)
        label_file = labels_root / label_relative
        if not label_file.is_file():
            # 跳过无对应标注的图片
            continue

        group = relative.parts[0] if len(relative.parts) > 1 else "."
        pairs.append(
            SamplePair(
                image_path=image_file,
                label_path=label_file,
                relative_path_from_images_root=relative,
                top_level_group=group,
            )
        )
    return pairs


def group_pairs_by_top_level(pairs: Sequence[SamplePair]) -> Dict[str, List[SamplePair]]:
    grouped: Dict[str, List[SamplePair]] = {}
    for p in pairs:
        grouped.setdefault(p.top_level_group, []).append(p)
    return grouped


def balanced_select_across_groups(
    grouped_pairs: Dict[str, List[SamplePair]], total_count: int, seed: int
) -> List[SamplePair]:
    if total_count <= 0:
        return []

    random_instance = random.Random(seed)

    # 仅保留非空组
    non_empty_groups = {g: lst[:] for g, lst in grouped_pairs.items() if lst}
    if not non_empty_groups:
        return []

    group_names = sorted(non_empty_groups.keys())
    group_sizes = {g: len(non_empty_groups[g]) for g in group_names}
    num_groups = len(group_names)

    # 目标：尽量在各组间均匀分配；若组不足以满足配额，则在其它组补齐
    base = total_count // num_groups
    remainder = total_count % num_groups

    selected: List[SamplePair] = []

    # 第一轮：每组分配 base 或 base+1
    per_group_take: Dict[str, int] = {}
    for idx, g in enumerate(group_names):
        want = base + (1 if idx < remainder else 0)
        take = min(want, group_sizes[g])
        per_group_take[g] = take
        if take > 0:
            selected.extend(random_instance.sample(non_empty_groups[g], take))

    # 第二轮：若总量不足 total_count，则在剩余容量的组中补齐
    still_need = total_count - len(selected)
    if still_need > 0:
        # 计算每组剩余可选数量
        remaining_capacity = {
            g: group_sizes[g] - per_group_take[g] for g in group_names
        }
        # 构建可补齐池
        pool: List[SamplePair] = []
        for g in group_names:
            if remaining_capacity[g] <= 0:
                continue
            # 从该组剩余样本中加入池
            already_selected_set = set(selected)
            leftovers = [
                p for p in non_empty_groups[g] if p not in already_selected_set
            ]
            pool.extend(leftovers)

        if pool:
            take_more = min(still_need, len(pool))
            selected.extend(random_instance.sample(pool, take_more))

    # 如果仍不足，说明总可用样本小于 total_count
    return selected


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    ensure_directory(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        # 复制文件元数据可选：copy2
        shutil.copy2(src, dst)


def place_in_yolo_structure(
    selected: Sequence[SamplePair],
    dest_root: Path,
    move_files: bool,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> ProcessStats:
    # 目录：dest_root/images/{train,val}/..., dest_root/labels/{train,val}/...
    images_dir = dest_root / "images"
    labels_dir = dest_root / "labels"

    random_instance = random.Random(seed)
    items = list(selected)
    random_instance.shuffle(items)

    total = len(items)
    # 优先使用 val_ratio，若 train+val 不为 1，进行归一
    if train_ratio <= 0 and val_ratio <= 0:
        train_ratio = 1.0
        val_ratio = 0.0
    s = train_ratio + val_ratio
    if s <= 0:
        train_ratio, val_ratio = 1.0, 0.0
        s = 1.0
    train_ratio /= s
    val_ratio /= s

    num_val = int(round(total * val_ratio))
    num_val = min(max(num_val, 0), total)
    num_train = total - num_val

    train_items = items[:num_train]
    val_items = items[num_train:]

    stats = ProcessStats(num_samples=total, num_samples_train=num_train, num_samples_val=num_val)

    # 跟踪各 split 下已使用的图片文件名，确保扁平化后不重名
    used_names_per_split: Dict[str, set] = {"train": set(), "val": set()}

    for split_name, subset in (("train", train_items), ("val", val_items)):
        for pair in _tqdm(
            subset,
            desc=f"写入 {split_name} 样本",
            unit="样本",
            leave=False,
        ):
            # YOLO 结构扁平化：不保留子目录，放入同一目录，避免重名
            split_images_dir = images_dir / split_name
            split_labels_dir = labels_dir / split_name

            base_image_name = pair.relative_path_from_images_root.name
            unique_image_name = _generate_unique_filename(
                split_images_dir, base_image_name, used_names_per_split[split_name]
            )
            unique_label_name = Path(unique_image_name).with_suffix(LABEL_EXTENSION).name

            image_dst = split_images_dir / unique_image_name
            label_dst = split_labels_dir / unique_label_name
            copy_or_move(pair.image_path, image_dst, move_files)
            copy_or_move(pair.label_path, label_dst, move_files)
            stats.num_images += 1
            stats.num_labels += 1

    return stats


def place_in_mirrored_structure(
    selected: Sequence[SamplePair],
    dest_root: Path,
    images_root: Path,
    labels_root: Path,
    move_files: bool,
) -> ProcessStats:
    # 在 dest_root 下镜像输入的目录结构：
    #   dest_root/<images_root_name>/...  和  dest_root/<labels_root_name>/...
    images_root_name = images_root.name
    labels_root_name = labels_root.name

    stats = ProcessStats(num_samples=len(selected))
    for pair in _tqdm(selected, desc="写入样本", unit="样本", leave=False):
        image_dst = dest_root / images_root_name / pair.relative_path_from_images_root
        label_relative = pair.relative_path_from_images_root.with_suffix(LABEL_EXTENSION)
        label_dst = dest_root / labels_root_name / label_relative
        copy_or_move(pair.image_path, image_dst, move_files)
        copy_or_move(pair.label_path, label_dst, move_files)
        stats.num_images += 1
        stats.num_labels += 1

    return stats


def print_directory_dirs_with_counts(
    root: Path, max_depth: int = 3, max_dirs_per_dir: int = 200
) -> None:
    print(f"输出目录结构（仅目录与文件数，最多深度 {max_depth}）：{root}")

    root = root.resolve()

    # 构建：目录 -> 其子树内文件总数（递归）
    from collections import defaultdict

    dir_file_counts: Dict[Path, int] = defaultdict(int)
    try:
        for path in root.rglob("*"):
            if path.is_file():
                parent = path.parent.resolve()
                while True:
                    dir_file_counts[parent] += 1
                    if parent == root:
                        break
                    parent = parent.parent
    except Exception:
        pass

    def recurse(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            subdirs = [p for p in current.iterdir() if p.is_dir()]
        except Exception:
            return
        subdirs.sort(key=lambda p: p.name.lower())

        more = 0
        if max_dirs_per_dir and len(subdirs) > max_dirs_per_dir:
            more = len(subdirs) - max_dirs_per_dir
            subdirs = subdirs[:max_dirs_per_dir]

        for idx, d in enumerate(subdirs):
            is_last = idx == len(subdirs) - 1 and more == 0
            connector = "└── " if is_last else "├── "
            count = dir_file_counts.get(d.resolve(), 0)
            print(prefix + connector + f"{d.name}/ ({count} 文件)")
            if depth < max_depth:
                child_prefix = prefix + ("    " if is_last else "│   ")
                recurse(d, child_prefix, depth + 1)

        if more > 0:
            print(prefix + f"└── … 其余 {more} 个目录")

    # 从根开始，仅打印其子目录（不重复打印根本身）
    recurse(root, prefix="", depth=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "随机抽取图像与对应标注，按子目录均衡分配，复制或移动到目标目录。"
        )
    )

    parser.add_argument("--images-root", "-i", type=Path, required=True, help="图片根目录")
    parser.add_argument("--labels-root", "-l", type=Path, required=True, help="标注根目录（YOLO .txt）")
    parser.add_argument("--dest-root", "-o", type=Path, required=True, help="输出根目录")

    parser.add_argument("--count", "-c", type=int, required=True, help="随机选取的样本总数")
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="随机种子（可复现）"
    )
    parser.add_argument(
        "--move", "-m", action="store_true", help="将文件移动到目标目录（默认复制）"
    )

    # YOLO 相关
    parser.add_argument(
        "--yolo", action="store_true", help="以 YOLO 目录结构输出（images/labels, train/val）"
    )
    parser.add_argument(
        "--val-ratio", "-v",
        type=float,
        default=None,
        help="验证集比例（0-1），指定则进行 train/val 切分；未指定则全部进 train",
    )
    parser.add_argument(
        "--train-ratio", "-t",
        type=float,
        default=None,
        help="训练集比例（0-1，可与 --val-ratio 一起使用；若未指定则为 1 - val-ratio）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_root: Path = args.images_root.resolve()
    labels_root: Path = args.labels_root.resolve()
    dest_root: Path = args.dest_root.resolve()
    total_count: int = args.count
    seed: int = args.seed
    move_files: bool = args.move
    use_yolo: bool = args.yolo
    val_ratio_arg: Optional[float] = args.val_ratio
    train_ratio_arg: Optional[float] = args.train_ratio

    if not images_root.exists() or not images_root.is_dir():
        raise SystemExit(f"图片根目录不存在或不可用：{images_root}")
    if not labels_root.exists() or not labels_root.is_dir():
        raise SystemExit(f"标注根目录不存在或不可用：{labels_root}")
    if total_count <= 0:
        raise SystemExit("--count 必须为正整数")

    print(f"扫描数据对：{images_root} ↔ {labels_root} ...")
    all_pairs = find_all_pairs(images_root, labels_root)
    if not all_pairs:
        raise SystemExit("未找到任何成对的 图片-标注(.txt) 文件。")

    grouped = group_pairs_by_top_level(all_pairs)
    print(
        f"发现可用样本 {len(all_pairs)}，分布于 {len([g for g in grouped if grouped[g]])} 个子目录组。"
    )

    selected = balanced_select_across_groups(grouped, total_count, seed)
    if len(selected) < total_count:
        print(
            f"警告：可用样本不足，目标 {total_count}，实际仅选取 {len(selected)}。"
        )
    else:
        print(f"已均衡抽样 {len(selected)} 个样本。")

    if use_yolo:
        # 处理 train/val 比例
        train_ratio: float
        val_ratio: float
        if val_ratio_arg is None and train_ratio_arg is None:
            train_ratio, val_ratio = 1.0, 0.0
        elif val_ratio_arg is not None and train_ratio_arg is None:
            if not (0.0 <= val_ratio_arg <= 1.0):
                raise SystemExit("--val-ratio 需在 [0,1] 范围内")
            val_ratio = val_ratio_arg
            train_ratio = 1.0 - val_ratio
        elif val_ratio_arg is None and train_ratio_arg is not None:
            if not (0.0 <= train_ratio_arg <= 1.0):
                raise SystemExit("--train-ratio 需在 [0,1] 范围内")
            train_ratio = train_ratio_arg
            val_ratio = 1.0 - train_ratio
        else:
            # 两者都提供时，允许不等于 1，后续会归一化
            if not (0.0 <= val_ratio_arg <= 1.0 and 0.0 <= train_ratio_arg <= 1.0):
                raise SystemExit("--train-ratio 与 --val-ratio 需在 [0,1] 范围内")
            train_ratio = float(train_ratio_arg)
            val_ratio = float(val_ratio_arg)

        print(
            f"以 YOLO 结构输出至：{dest_root} （train:val ≈ {train_ratio:.2f}:{val_ratio:.2f}）"
        )
        stats = place_in_yolo_structure(
            selected=selected,
            dest_root=dest_root,
            move_files=move_files,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        print(f"完成：train={stats.num_samples_train}, val={stats.num_samples_val}")
        action = "移动" if move_files else "复制"
        total_files = stats.num_images + stats.num_labels
        print(f"共{action} {total_files} 个文件（图片 {stats.num_images}、标注 {stats.num_labels}）。")
        print_directory_dirs_with_counts(dest_root, max_depth=4, max_dirs_per_dir=300)
    else:
        print(f"以镜像目录结构输出至：{dest_root}")
        stats = place_in_mirrored_structure(
            selected=selected,
            dest_root=dest_root,
            images_root=images_root,
            labels_root=labels_root,
            move_files=move_files,
        )
        print(f"完成：导出样本 {stats.num_samples}。")
        action = "移动" if move_files else "复制"
        total_files = stats.num_images + stats.num_labels
        print(f"共{action} {total_files} 个文件（图片 {stats.num_images}、标注 {stats.num_labels}）。")
        print_directory_dirs_with_counts(dest_root, max_depth=4, max_dirs_per_dir=300)


if __name__ == "__main__":
    main()


