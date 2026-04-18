#!/usr/bin/env bash
set -e

python scripts/val.py \
  --weights runs/ovd/coco_yoloe_11l_det/weights/best.pt \
  --gt-json /path/to/coco/annotations/instances_val2017.json \
  --image-root /path/to/coco/val2017 \
  --dataset coco \
  --out-json runs/ovd/coco_preds.json \
  --metrics-json runs/ovd/coco_metrics.json \
  --imgsz 640 \
  --device 0 \
  --half
