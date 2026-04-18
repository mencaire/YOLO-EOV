#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOE

from ovd.utils import read_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-vocabulary inference with YOLOE.")
    parser.add_argument("--weights", default="yoloe-11l-seg.pt")
    parser.add_argument("--source", required=True, help="Image / dir / video path.")
    parser.add_argument("--classes", nargs="*", default=None, help="Text prompts, e.g. --classes person bus helmet")
    parser.add_argument("--class-file", default=None, help="One prompt per line.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--project", default="runs/ovd")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.classes or []
    if args.class_file:
        prompts.extend(read_lines(args.class_file))
    if not prompts:
        raise ValueError("Please provide --classes or --class-file for open-vocabulary inference.")

    model = YOLOE(args.weights)
    model.set_classes(prompts)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        save=args.save,
        project=args.project,
        name=args.name,
    )
    print(results)


if __name__ == "__main__":
    main()
