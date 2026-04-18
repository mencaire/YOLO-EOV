#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLOE

from ovd.evaluation import eval_coco
from ovd.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--dataset", choices=["coco", "lvis"], required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def load_categories(gt_json: str):
    with open(gt_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda x: x["id"])
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    names = [c["name"] for c in cats]
    cat_ids = [c["id"] for c in cats]
    return cat_id_to_name, names, cat_ids


def main():
    args = parse_args()

    cat_id_to_name, names, cat_ids = load_categories(args.gt_json)
    model = YOLOE(args.weights)

    # 显式设置开放词汇类别
    model.set_classes(names, model.get_text_pe(names))

    image_root = Path(args.image_root)
    image_paths = sorted(image_root.glob("*.jpg"))

    preds = []
    for i, img_path in enumerate(image_paths, 1):
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            half=args.half,
            verbose=False,
        )

        if not results:
            continue

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        image_id = int(img_path.stem)

        for box, score, cls_idx in zip(boxes, scores, classes):
            if cls_idx < 0 or cls_idx >= len(cat_ids):
                continue

            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(cat_ids[cls_idx]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

        if i % 500 == 0:
            print(f"processed {i}/{len(image_paths)} images, current preds={len(preds)}", flush=True)

    save_json(preds, args.out_json)
    print(f"Saved {len(preds)} predictions to {args.out_json}")

    if args.dataset == "coco":
        metrics = eval_coco(args.gt_json, args.out_json)
    else:
        raise NotImplementedError("LVIS eval not patched here yet.")

    save_json(metrics, args.metrics_json)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
