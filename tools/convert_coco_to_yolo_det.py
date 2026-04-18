#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ovd.data import build_category_map
from ovd.utils import ensure_dir, load_json, save_json


def xywh_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / width, cy / height, w / width, h / height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO detection json to Ultralytics YOLO txt labels.")
    parser.add_argument("--ann-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", required=True, help="train2017 / val2017 / test-dev, only used for folder naming.")
    parser.add_argument("--copy-images", action="store_true", help="Create image list txt instead of copying images by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.ann_json)
    categories = data["categories"]
    images = data["images"]
    annotations = data.get("annotations", [])

    cat_map = build_category_map(categories)
    img_by_id = {im["id"]: im for im in images}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        anns_by_img[int(ann["image_id"])] .append(ann)

    out_root = Path(args.output_root)
    labels_dir = out_root / "labels" / args.split
    images_txt_dir = out_root / "images"
    ensure_dir(labels_dir)
    ensure_dir(images_txt_dir)

    image_list_path = images_txt_dir / f"{args.split}.txt"
    with open(image_list_path, "w", encoding="utf-8") as list_f:
        for im in images:
            image_id = int(im["id"])
            width, height = int(im["width"]), int(im["height"])
            image_path = Path(args.image_root) / im["file_name"]
            list_f.write(str(image_path.resolve()) + "\n")

            label_path = labels_dir / (Path(im["file_name"]).stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as lf:
                for ann in anns_by_img.get(image_id, []):
                    bbox = ann["bbox"]
                    if bbox[2] <= 1 or bbox[3] <= 1:
                        continue
                    cls_idx = cat_map.cat_id_to_contiguous[int(ann["category_id"])]
                    cx, cy, w, h = xywh_to_yolo(bbox, width, height)
                    lf.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    meta = {
        "names": cat_map.names,
        "contiguous_to_cat_id": cat_map.contiguous_to_cat_id,
        "cat_id_to_contiguous": cat_map.cat_id_to_contiguous,
        "split": args.split,
        "image_list": str(image_list_path.resolve()),
        "image_root": str(Path(args.image_root).resolve()),
        "ann_json": str(Path(args.ann_json).resolve()),
    }
    save_json(meta, out_root / f"meta_{args.split}.json")
    print(f"Done. Labels: {labels_dir}")
    print(f"Image list: {image_list_path}")


if __name__ == "__main__":
    main()
