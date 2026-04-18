#!/usr/bin/env bash
set -e

python scripts/export.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --classes person bicycle car bus truck traffic_light stop_sign \
  --format onnx \
  --imgsz 640 \
  --half \
  --dynamic
