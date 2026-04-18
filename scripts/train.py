#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOE for open-vocabulary object detection.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset yaml file.")
    parser.add_argument("--model-cfg", default="yoloe-11l.yaml", help="Detection model config.")
    parser.add_argument("--pretrained", default="yoloe-11l-seg.pt", help="Pretrained segmentation checkpoint.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/ovd")
    parser.add_argument("--name", default="yoloe_11l_det")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLOE(args.model_cfg)
    model.load(args.pretrained)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        trainer=YOLOEPETrainer,
        freeze=args.freeze,
        resume=args.resume,
        amp=args.amp,
    )
    print(results)


if __name__ == "__main__":
    main()
