#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ultralytics import YOLOE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print model information and parameter count.")
    parser.add_argument("--weights", default="yoloe-11l-seg.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLOE(args.weights)
    try:
        info = model.info(verbose=True)
        print(info)
    except TypeError:
        print(model.info())


if __name__ == "__main__":
    main()
