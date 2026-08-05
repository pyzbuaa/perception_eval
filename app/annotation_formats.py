from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


VISDRONE_CATEGORIES = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "people"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "van"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "tricycle"},
    {"id": 7, "name": "awning-tricycle"},
    {"id": 8, "name": "bus"},
    {"id": 9, "name": "motor"},
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def is_visdrone_label_directory(directory: Path) -> bool:
    for path in directory.rglob("*.txt"):
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.strip():
                return len(line.split(",")) >= 8
    return False


def convert_visdrone_to_coco(
    image_directory: Path,
    label_directory: Path,
    output_path: Path,
    description: str,
) -> dict[str, Any]:
    label_files = {
        path.relative_to(label_directory).with_suffix("").as_posix(): path
        for path in sorted(label_directory.rglob("*.txt"))
    }
    image_paths = [
        path
        for path in sorted(image_directory.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "annotations" not in path.relative_to(image_directory).parts
    ]
    if not image_paths:
        raise ValueError("VisDrone 转换没有找到图像")
    if not label_files:
        raise ValueError("VisDrone 转换没有找到 TXT 标注")

    images = []
    annotations = []
    annotation_id = 1
    ignored_count = 0
    missing_label_count = 0
    for image_id, image_path in enumerate(image_paths, start=1):
        relative_image = image_path.relative_to(image_directory)
        with Image.open(image_path) as image:
            width, height = image.width, image.height
        images.append(
            {
                "id": image_id,
                "file_name": relative_image.as_posix(),
                "width": width,
                "height": height,
            }
        )
        label_key = relative_image.with_suffix("").as_posix()
        label_path = label_files.get(label_key)
        if not label_path:
            path_matches = [
                path
                for key, path in label_files.items()
                if key.endswith(f"/{label_key}")
                or label_key.endswith(f"/{key}")
            ]
            label_path = path_matches[0] if len(path_matches) == 1 else None
        if not label_path:
            basename_matches = [
                path
                for key, path in label_files.items()
                if Path(key).name == relative_image.stem
            ]
            label_path = (
                basename_matches[0]
                if len(basename_matches) == 1
                else None
            )
        if not label_path:
            missing_label_count += 1
            continue
        for line in label_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = line.strip().split(",")
            if len(fields) < 8:
                continue
            try:
                x, y, box_width, box_height = (
                    float(value) for value in fields[:4]
                )
                score = int(float(fields[4]))
                source_category_id = int(float(fields[5]))
            except ValueError:
                continue
            if (
                score <= 0
                or source_category_id not in range(1, 11)
                or box_width <= 0
                or box_height <= 0
            ):
                ignored_count += 1
                continue
            x = max(0.0, min(x, float(width)))
            y = max(0.0, min(y, float(height)))
            box_width = max(0.0, min(box_width, width - x))
            box_height = max(0.0, min(box_height, height - y))
            if box_width == 0 or box_height == 0:
                ignored_count += 1
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": source_category_id - 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": description,
            "source_format": "VisDrone",
        },
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_CATEGORIES,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return {
        "images": len(images),
        "annotations": len(annotations),
        "ignored": ignored_count,
        "missing_labels": missing_label_count,
        "path": str(output_path),
    }
