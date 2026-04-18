from __future__ import annotations

from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_coco(gt_json, pred_json):
    coco_gt = COCO(str(gt_json))

    pred_json = Path(pred_json)
    if (not pred_json.exists()) or pred_json.stat().st_size == 0:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "ARs": 0.0,
            "ARm": 0.0,
            "ARl": 0.0,
            "num_predictions": 0,
        }

    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    if not preds:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "ARs": 0.0,
            "ARm": 0.0,
            "ARl": 0.0,
            "num_predictions": 0,
        }

    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    s = coco_eval.stats
    return {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "APs": float(s[3]),
        "APm": float(s[4]),
        "APl": float(s[5]),
        "AR1": float(s[6]),
        "AR10": float(s[7]),
        "AR100": float(s[8]),
        "ARs": float(s[9]),
        "ARm": float(s[10]),
        "ARl": float(s[11]),
        "num_predictions": len(preds),
    }
