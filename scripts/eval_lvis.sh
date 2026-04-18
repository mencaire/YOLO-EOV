#!/usr/bin/env bash
set -e

python scripts/val.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --gt-json /path/to/lvis/annotations/lvis_v1_val.json \
  --image-root /path/to/coco \
  --dataset lvis \
  --out-json runs/ovd/lvis_preds.json \
  --metrics-json runs/ovd/lvis_metrics.json \
  --imgsz 640 \
  --device 0 \
  --half
