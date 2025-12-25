'''
用户输入基准音频和声音目录，将其中的音频两两混音合成
前期准备：
1.pip install pydub
2.下载ffmpeg
示例：
python mix_audio.py \
  --base "/path/to/base.mp3" \
  --src_dir "/path/to/src_folder" \
  --out_dir "/path/to/output_folder" \
  --num 50 \
  --seed 123 \
  --base_gain -6 \
  --target_gain -3 \
  --overwrite
'''
import argparse
import random
import sys
from pathlib import Path

from pydub import AudioSegment


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus", ".aiff", ".aif"}


def is_audio_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in AUDIO_EXTS


def load_audio(p: Path) -> AudioSegment:
    # pydub 会调用 ffmpeg 解码
    return AudioSegment.from_file(p)


def match_duration(base: AudioSegment, target: AudioSegment) -> AudioSegment:
    """
    让 base 的时长匹配 target：
    - base 短：循环拼接直到 >= target，再截断
    - base 长：直接截断
    """
    if len(base) == 0:
        return base
    if len(base) < len(target):
        times = (len(target) // len(base)) + 1
        base = base * times
    return base[: len(target)]


def mix_two(base: AudioSegment, target: AudioSegment, base_gain_db: float, target_gain_db: float) -> AudioSegment:
    """
    将 base 与 target 混音：
    - 默认保持 target 的时长
    - base 做时长匹配后，从头 overlay 到 target 上
    """
    base2 = match_duration(base, target)

    # 为避免削波/爆音，通常会把两路都降一点；这里交给参数控制
    base2 = base2.apply_gain(base_gain_db)
    target2 = target.apply_gain(target_gain_db)

    mixed = target2.overlay(base2, position=0)
    return mixed


def export_audio(seg: AudioSegment, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = out_path.suffix.lower().lstrip(".")
    if not fmt:
        fmt = "wav"
        out_path = out_path.with_suffix(".wav")

    # 常见格式直接按后缀导出；如遇不支持可改成统一 wav
    seg.export(out_path, format=fmt)


def main():
    parser = argparse.ArgumentParser(
        description="将一个基准音频与目录内随机选取的音频逐个混音合成，并保持目录结构输出。"
    )
    parser.add_argument("--base", required=True, help="基准音频文件路径（要混进去的那个音频）")
    parser.add_argument("--src_dir", required=True, help="要遍历并抽取进行混音的目录")
    parser.add_argument("--out_dir", required=True, help="输出目录（保持 src_dir 的目录结构）")
    parser.add_argument("--num", type=int, required=True, help="随机抽取合成的数量 N（不足则全选）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选，用于可复现抽样）")
    parser.add_argument("--base_gain", type=float, default=-6.0, help="基准音频音量增益(dB)，默认 -6")
    parser.add_argument("--target_gain", type=float, default=-3.0, help="目标音频音量增益(dB)，默认 -3")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    args = parser.parse_args()

    base_path = Path(args.base).expanduser().resolve()
    src_dir = Path(args.src_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not base_path.exists() or not base_path.is_file():
        print(f"[ERROR] base 文件不存在：{base_path}", file=sys.stderr)
        sys.exit(1)
    if not is_audio_file(base_path):
        print(f"[ERROR] base 不是支持的音频格式：{base_path}", file=sys.stderr)
        sys.exit(1)
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"[ERROR] src_dir 目录不存在：{src_dir}", file=sys.stderr)
        sys.exit(1)

    # 收集音频
    all_audio = [p for p in src_dir.rglob("*") if is_audio_file(p)]
    # 如果 base 恰好也在 src_dir 内，避免自己跟自己混
    all_audio = [p for p in all_audio if p.resolve() != base_path]

    if not all_audio:
        print(f"[WARN] 在目录中未找到音频：{src_dir}")
        sys.exit(0)

    n = args.num
    if n <= 0:
        print("[ERROR] num 必须为正整数", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    if n >= len(all_audio):
        chosen = all_audio
    else:
        chosen = random.sample(all_audio, n)

    print(f"[INFO] 找到音频 {len(all_audio)} 个，抽取 {len(chosen)} 个进行合成。")

    # 读取 base
    try:
        base_audio = load_audio(base_path)
    except Exception as e:
        print(f"[ERROR] 读取 base 失败：{base_path}\n{e}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    failed = 0

    for i, target_path in enumerate(chosen, 1):
        rel = target_path.relative_to(src_dir)  # 保持目录结构
        out_path = out_dir / rel

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] 已存在且不覆盖：{out_path}")
            continue

        try:
            target_audio = load_audio(target_path)

            # 统一采样率/声道，减少 overlay 时的奇怪问题
            # （以 target 为准）
            base2 = base_audio.set_frame_rate(target_audio.frame_rate).set_channels(target_audio.channels)

            mixed = mix_two(
                base=base2,
                target=target_audio,
                base_gain_db=args.base_gain,
                target_gain_db=args.target_gain,
            )

            export_audio(mixed, out_path)
            ok += 1
            print(f"[OK {i}/{len(chosen)}] {target_path}  ->  {out_path}")
        except Exception as e:
            failed += 1
            print(f"[FAIL {i}/{len(chosen)}] {target_path}\n  {e}", file=sys.stderr)

    print(f"[DONE] 成功：{ok} 失败：{failed} 输出目录：{out_dir}")


if __name__ == "__main__":
    main()
