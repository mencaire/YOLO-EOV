#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ovd.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Ultralytics dataset yaml from converted meta json files.")
    parser.add_argument("--train-meta", required=True)
    parser.add_argument("--val-meta", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--dataset-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_meta = load_json(args.train_meta)
    val_meta = load_json(args.val_meta)
    if train_meta["names"] != val_meta["names"]:
        raise ValueError("Train/val category names do not match.")

    out_yaml = Path(args.output_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    names = train_meta["names"]
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {Path(args.dataset_root).resolve()}\n")
        f.write(f"train: {Path(train_meta['image_list']).resolve()}\n")
        f.write(f"val: {Path(val_meta['image_list']).resolve()}\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(names):
            safe_name = str(name).replace("\"", "'")
            f.write(f"  {idx}: \"{safe_name}\"\n")

    print(f"Saved dataset yaml to {out_yaml}")


if __name__ == "__main__":
    main()
