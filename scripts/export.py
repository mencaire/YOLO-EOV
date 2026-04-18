#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ultralytics import YOLOE

from ovd.utils import read_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOE with fixed open-vocabulary prompts.")
    parser.add_argument("--weights", default="yoloe-11l-seg.pt")
    parser.add_argument("--classes", nargs="*", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLOE(args.weights)
    model.set_classes(args.classes)
    out = model.export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)
    print(out)


if __name__ == "__main__":
    main()
