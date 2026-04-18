from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json, normalize_name, resolve_image_path


@dataclass
class CategoryMap:
    contiguous_to_cat_id: dict[int, int]
    cat_id_to_contiguous: dict[int, int]
    contiguous_to_name: dict[int, str]

    @property
    def names(self) -> list[str]:
        return [self.contiguous_to_name[i] for i in sorted(self.contiguous_to_name)]


def build_category_map(categories: list[dict[str, Any]]) -> CategoryMap:
    categories_sorted = sorted(categories, key=lambda x: x["id"])
    contiguous_to_cat_id = {i: c["id"] for i, c in enumerate(categories_sorted)}
    cat_id_to_contiguous = {c["id"]: i for i, c in enumerate(categories_sorted)}
    contiguous_to_name = {i: normalize_name(c["name"]) for i, c in enumerate(categories_sorted)}
    return CategoryMap(
        contiguous_to_cat_id=contiguous_to_cat_id,
        cat_id_to_contiguous=cat_id_to_contiguous,
        contiguous_to_name=contiguous_to_name,
    )


def load_detection_gt(gt_json: str | Path, image_root: str | Path) -> tuple[list[dict[str, Any]], CategoryMap]:
    data = load_json(gt_json)
    images = data["images"]
    categories = data["categories"]
    cat_map = build_category_map(categories)

    resolved = []
    for im in images:
        file_name = im["file_name"]
        path = resolve_image_path(image_root, file_name)
        item = dict(im)
        item["path"] = str(path)
        resolved.append(item)
    return resolved, cat_map
