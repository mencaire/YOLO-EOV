#!/usr/bin/env bash
set -e

python scripts/benchmark.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --image-list /path/to/bench_images.txt \
  --classes person car dog backpack helmet \
  --imgsz 640 \
  --device 0 \
  --half \
  --warmup 20 \
  --repeats 200 \
  --out-json runs/ovd/benchmark.json
