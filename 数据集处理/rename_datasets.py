# 重命名图像和标注文件，按数字递增顺序命名
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def rename_files(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if input_path.is_file():
        files = [input_path]
        output_subdir = output_dir
    else:
        files = list(input_path.glob('*'))
        output_subdir = output_dir / input_path.name

    output_subdir.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files, start=1):
        if file.is_file():
            new_name = f"{i}{file.suffix}"
            shutil.copy2(file, output_subdir / new_name)
            yield file


def process_files(input_paths, output_dir):
    total_files = sum(len(list(Path(path).glob('*'))) if Path(path).is_dir() else 1 for path in input_paths)

    with tqdm(total=total_files, desc="处理进度") as pbar:
        for path in input_paths:
            for file in rename_files(path, output_dir):
                pbar.update(1)
                # pbar.set_postfix({"当前文件": file.name}, refresh=True)

def get_args():
    parser = argparse.ArgumentParser(description="重命名图像和标注文件，按数字递增顺序命名")
    parser.add_argument("--images", "-i", help="图像文件或目录路径")
    parser.add_argument("--annotations", "-a", help="标注文件或目录路径")
    parser.add_argument("--output", "-o", required=True, help="输出目录路径")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    input_paths = [path for path in [args.images, args.annotations] if path]

    assert input_paths, "至少需要提供一个输入路径（图像或标注）"

    process_files(input_paths, args.output)

    if args.images and args.annotations:
        print("注意：图像和标注文件已按原始顺序重命名，请确保它们的对应关系正确。")


if __name__ == "__main__":
    main()