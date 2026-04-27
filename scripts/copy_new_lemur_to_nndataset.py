#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def _default_nn_dataset_root() -> Path:
    env_root = Path(str(Path.cwd()))
    sibling = env_root.parent / "nn-dataset" / "ab" / "nn"
    if sibling.exists():
        return sibling
    return Path("/home/s471802/nn-dataset/ab/nn")


def _copy_matching_files(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            count += 1
    return count


def _copy_matching_dirs(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob(pattern)):
        if src.is_dir():
            shutil.copytree(src, dst_dir / src.name, dirs_exist_ok=True)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy successful NNEval new_lemur outputs into nn-dataset.")
    parser.add_argument("--prefix", default="rl-bb-struct1")
    parser.add_argument("--nngpt-root", type=Path, default=Path.cwd())
    parser.add_argument("--nn-dataset-root", type=Path, default=_default_nn_dataset_root())
    args = parser.parse_args()

    new_lemur_root = args.nngpt_root / "out" / "nngpt" / "new_lemur"
    nn_count = _copy_matching_files(
        new_lemur_root / "nn",
        args.nn_dataset_root / "nn",
        f"{args.prefix}-*.py",
    )
    stat_count = _copy_matching_dirs(
        new_lemur_root / "train",
        args.nn_dataset_root / "stat" / "train",
        f"*_{args.prefix}-*",
    )
    print(f"Copied {nn_count} NN files and {stat_count} stat directories for prefix {args.prefix!r}.")


if __name__ == "__main__":
    main()
