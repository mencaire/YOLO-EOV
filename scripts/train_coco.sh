#!/usr/bin/env bash
set -e

python scripts/train.py \
  --data /path/to/coco_ovd.yaml \
  --model-cfg yoloe-11l.yaml \
  --pretrained yoloe-11l-seg.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --project runs/ovd \
  --name coco_yoloe_11l_det \
  --amp
