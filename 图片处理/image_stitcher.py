"""
图片快速拼接脚本（按文件名排序，按 N 张一组从上到下拼接）。

需求要点：
1) 用户输入：图片文件夹路径、输出目录、N（N>1）
2) 选择逻辑：按文件名排序（文件名为时间戳）
3) 拼接逻辑：每 N 张为一组顺序拼接；若最后剩余不足 N 张，则使用“最后 N 张”再拼接一次；
   若文件夹图片总数 < N，则控制台警告并退出。

使用示例：
python image_stitcher.py -i "dist\save\13" -o "dist\save\13_stitched" -n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def _clean_user_path(s: str) -> str:
    """去除用户输入路径两端空白和引号。"""
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def _iter_image_files(input_dir: Path) -> List[Path]:
    """仅扫描一级目录下的图片文件（不递归），按文件名排序。"""
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是文件夹：{input_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def _build_groups(total: int, n: int) -> List[Tuple[int, int]]:
    """
    生成拼接分组 [start, end)。
    - 正常：0..n, n..2n...
    - 若最后不足 n：追加 (total-n, total) 并结束（与前一组可能重叠，符合需求）
    """
    if n <= 1:
        raise ValueError("n 必须为大于 1 的整数")

    if total < n:
        return []

    groups: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = start + n
        if end <= total:
            groups.append((start, end))
            start = end
            continue

        # 剩余不足 n：选择最后 n 张
        groups.append((total - n, total))
        break
    return groups


def _open_image_rgb(path: Path):
    """
    读取图片并转为 RGB。
    - Pillow 对中文路径天然友好
    - 额外做一次 exif_transpose，避免少量图片带旋转信息导致方向不一致
    """
    from PIL import Image, ImageOps  # lazy import

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def _vstack_images_pil(images_rgb: Sequence, align: str = "left", background=(0, 0, 0)):
    """将多张 RGB 图片按从上到下拼接；允许宽度不一致（自动补底色）。"""
    from PIL import Image  # lazy import

    if not images_rgb:
        raise ValueError("images_rgb 不能为空")

    widths = [int(im.size[0]) for im in images_rgb]
    heights = [int(im.size[1]) for im in images_rgb]
    max_w = max(widths)
    total_h = sum(heights)

    out = Image.new("RGB", (max_w, total_h), color=background)
    y = 0
    for im in images_rgb:
        w, h = im.size
        if align == "center":
            x = (max_w - w) // 2
        elif align == "right":
            x = max_w - w
        else:
            x = 0
        out.paste(im, (int(x), int(y)))
        y += h

    return out


def _save_image_pil(path: Path, img) -> None:
    """保存图片（按扩展名决定编码参数）。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = (path.suffix or ".jpg").lower()
    save_kwargs = {}
    if ext in {".jpg", ".jpeg"}:
        # JPEG 只支持 RGB
        if getattr(img, "mode", "") != "RGB":
            img = img.convert("RGB")
        save_kwargs.update({"quality": 95, "subsampling": 0})
    elif ext == ".png":
        save_kwargs.update({"compress_level": 3})

    img.save(str(path), **save_kwargs)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按文件名排序，将图片按 N 张一组纵向拼接输出。")
    parser.add_argument("--input", "-i", dest="input_dir", help="输入图片文件夹路径")
    parser.add_argument("--output", "-o", dest="output_dir", help="输出目录")
    parser.add_argument("--n", "-n", dest="n", type=int, help="每张拼接使用的图片数量（>1）")
    parser.add_argument(
        "--ext",
        dest="ext",
        default=".jpg",
        help="输出图片扩展名（如 .jpg / .png），默认 .jpg",
    )
    parser.add_argument(
        "--align",
        dest="align",
        default="left",
        choices=["left", "center", "right"],
        help="宽度不一致时的水平对齐方式，默认 left",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse_args(argv)

    input_dir_str = _clean_user_path(args.input_dir) if args.input_dir else ""
    output_dir_str = _clean_user_path(args.output_dir) if args.output_dir else ""

    if not input_dir_str:
        input_dir_str = _clean_user_path(input("请输入要拼接的图片文件夹路径："))
    if not output_dir_str:
        output_dir_str = _clean_user_path(input("请输入拼接后图片输出目录："))

    # N
    n = args.n
    while n is None:
        s = _clean_user_path(input("请输入每次拼接使用的图片数量 n（n>1）："))
        try:
            n = int(s)
        except Exception:
            print("n 必须是整数，请重新输入。")
            n = None
            continue
        if n <= 1:
            print("n 必须大于 1，请重新输入。")
            n = None

    ext = (args.ext or ".jpg").strip().lower()
    if not ext.startswith("."):
        ext = "." + ext

    input_dir = Path(input_dir_str).expanduser()
    output_dir = Path(output_dir_str).expanduser()

    try:
        files = _iter_image_files(input_dir)
    except Exception as e:
        print(f"[错误] {e}")
        return 2

    total = len(files)
    print(f"找到图片数量：{total}")

    if total < n:
        print(f"[警告] 图片数量({total})小于 n({n})，无法拼接。程序退出。")
        return 1

    groups = _build_groups(total, n)
    if not groups:
        print(f"[警告] 未生成任何拼接分组。程序退出。")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (start, end) in enumerate(groups, start=1):
        batch = files[start:end]
        first_name = batch[0].stem
        last_name = batch[-1].stem

        # 读取并统一为 RGB
        images_rgb = []
        for p in batch:
            try:
                img = _open_image_rgb(p)
            except Exception:
                print(f"[错误] 读取/解析图片失败：{p}")
                return 3
            images_rgb.append(img)

        merged = _vstack_images_pil(images_rgb, align=args.align, background=(0, 0, 0))

        out_name = f"{idx:04d}_{first_name}__{last_name}{ext}"
        out_path = output_dir / out_name
        _save_image_pil(out_path, merged)

        print(
            f"[{idx}/{len(groups)}] 拼接 {len(batch)} 张：{batch[0].name} ~ {batch[-1].name} -> {out_path}"
        )

    print("全部拼接完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

