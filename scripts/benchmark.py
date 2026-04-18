#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOE

from ovd.evaluation import benchmark_model
from ovd.utils import load_json, read_lines, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOE FPS on a list of images.")
    parser.add_argument("--weights", default="yoloe-11l-seg.pt")
    parser.add_argument("--image-list", required=True, help="Txt file containing one image path per line.")
    parser.add_argument("--classes", nargs="*", default=["person", "car", "dog"])
    parser.add_argument("--class-file", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out-json", default="runs/ovd/benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = read_lines(args.image_list)
    prompts = list(args.classes)
    if args.class_file:
        prompts.extend(read_lines(args.class_file))

    model = YOLOE(args.weights)
    result = benchmark_model(
        model=model,
        images=images,
        class_names=prompts,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    save_json(result, args.out_json)
    print(result)


if __name__ == "__main__":
    main()
