#!/usr/bin/env bash
set -e

python scripts/train.py \
  --data /path/to/lvis_ovd.yaml \
  --model-cfg yoloe-11l.yaml \
  --pretrained runs/ovd/coco_yoloe_11l_det/weights/best.pt \
  --epochs 40 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --project runs/ovd \
  --name lvis_yoloe_11l_det \
  --amp
