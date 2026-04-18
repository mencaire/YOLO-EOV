from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().split())


def resolve_image_path(image_root: str | os.PathLike, file_name: str) -> Path:
    image_root = Path(image_root)
    path = image_root / file_name
    if path.exists():
        return path

    # LVIS often stores file_name like train2017/0000001.jpg while image_root may already point at coco root.
    if "/" in file_name:
        alt = image_root / Path(file_name).name
        if alt.exists():
            return alt

    # COCO/LVIS sometimes need a broader search fallback.
    candidate = image_root / Path(file_name).name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Cannot resolve image path: root={image_root}, file_name={file_name}")


def read_lines(path: str | os.PathLike) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
