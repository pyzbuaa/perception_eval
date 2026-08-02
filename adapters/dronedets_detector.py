#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ALIASES = {
    "motor": "motorcycle",
}


def dronedets_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "DroneDets"
    return Path(os.environ.get("DRONEDETS_ROOT", default)).expanduser().resolve()


def load_dronedets(root: Path):
    source = root / "src"
    if not (source / "aerial_det" / "catalog.py").is_file():
        raise FileNotFoundError(f"DroneDets 项目不存在或不完整: {root}")
    source_value = str(source)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    from aerial_det.catalog import Catalog
    from aerial_det.weights import resolve_weights

    return Catalog, resolve_weights


def normalize_label(value: str) -> str:
    return value.strip().casefold().replace("_", "-").replace(" ", "-")


def category_mapping(
    categories: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
) -> dict[str, int]:
    by_name = {
        normalize_label(str(category["name"])): int(category["id"])
        for category in categories
    }
    mapping = dict(by_name)
    for source, target in {**DEFAULT_ALIASES, **(aliases or {})}.items():
        target_id = by_name.get(normalize_label(target))
        if target_id is not None:
            mapping[normalize_label(source)] = target_id
    return mapping


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "ultralytics", "aerial-det"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def health(result_path: Path | None) -> None:
    root = dronedets_root()
    Catalog, _ = load_dronedets(root)
    import torch
    import ultralytics

    catalog = Catalog()
    model = catalog.get("yolov8m_visdrone")
    result = {
        "protocol_version": "1.0",
        "adapter_id": "adapter_dronedets_yolov8m",
        "status": "healthy",
        "root": str(root),
        "model": model.id,
        "backend": model.backend,
        "cuda_available": torch.cuda.is_available(),
        "versions": _package_versions(),
        "ultralytics_license": "AGPL-3.0",
    }
    if result_path:
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, ensure_ascii=False))


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版检测 Adapter 协议")
    model_request = request.get("model", {})
    catalog_model_id = model_request.get("catalog_model_id")
    if catalog_model_id != "yolov8m_visdrone":
        raise ValueError(f"当前 Adapter 不支持模型: {catalog_model_id}")

    root = dronedets_root()
    Catalog, resolve_weights = load_dronedets(root)
    catalog = Catalog()
    spec = catalog.get(catalog_model_id)
    checkpoint = model_request.get("weight_path")
    weight_path = resolve_weights(spec.weights, checkpoint)
    if checkpoint and not Path(weight_path).is_file():
        raise FileNotFoundError(f"模型权重不存在: {weight_path}")

    dataset = request.get("dataset", {})
    image_directory = Path(dataset["image_directory"]).resolve()
    ground_truth_path = Path(dataset["annotation_path"]).resolve()
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    categories = ground_truth.get("categories", [])
    mapping = category_mapping(
        categories,
        request.get("category_aliases"),
    )
    images = sorted(ground_truth.get("images", []), key=lambda item: int(item["id"]))
    if not images:
        raise ValueError("COCO 真值中没有图片")
    image_paths = []
    for image in images:
        path = (image_directory / image["file_name"]).resolve()
        if not path.is_relative_to(image_directory):
            raise ValueError(f"图片路径超出数据集目录: {image['file_name']}")
        if not path.is_file():
            raise FileNotFoundError(f"缺少评测图片: {path}")
        image_paths.append(path)

    inference = request.get("inference", {})
    if int(inference.get("batch_size", 1)) != 1:
        raise ValueError("DroneDets YOLOv8m 首版仅支持 batch_size=1")
    device = str(inference.get("device", "cuda:0"))
    precision = str(inference.get("precision", "FP16"))
    if precision not in {"FP16", "FP32"}:
        raise ValueError("DroneDets YOLOv8m 当前只支持 FP16 或 FP32")
    confidence = float(inference.get("confidence", 0.001))
    iou = float(inference.get("nms_iou", 0.7))
    image_size = int(inference.get("image_size", 1280))
    max_detections = int(inference.get("max_detections", 300))
    warmup = int(inference.get("warmup", 0))
    seed = int(request.get("seed", 1001))

    import torch
    from ultralytics import YOLO

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法执行 DroneDets YOLOv8m 推理")

    output_directory = Path(request["output_directory"]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    model = YOLO(weight_path)
    model_load_ms = (time.perf_counter() - load_started) * 1000
    options = {
        "save": False,
        "verbose": False,
        "conf": confidence,
        "iou": iou,
        "imgsz": image_size,
        "max_det": max_detections,
        "device": device,
        "half": precision == "FP16",
    }
    for _ in range(warmup):
        model.predict(source=str(image_paths[0]), **options)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    predictions = []
    unmatched: Counter[str] = Counter()
    preprocess_ms = []
    inference_ms = []
    postprocess_ms = []
    started = time.perf_counter()
    for index, (image, image_path) in enumerate(zip(images, image_paths), start=1):
        prediction = model.predict(source=str(image_path), **options)[0]
        speed = prediction.speed or {}
        preprocess_ms.append(float(speed.get("preprocess", 0.0)))
        inference_ms.append(float(speed.get("inference", 0.0)))
        postprocess_ms.append(float(speed.get("postprocess", 0.0)))
        if prediction.boxes is not None:
            for box, score, label in zip(
                prediction.boxes.xyxy.cpu().tolist(),
                prediction.boxes.conf.cpu().tolist(),
                prediction.boxes.cls.cpu().tolist(),
            ):
                label_id = int(label)
                label_name = str(prediction.names.get(label_id, label_id))
                category_id = mapping.get(normalize_label(label_name))
                if category_id is None:
                    unmatched[label_name] += 1
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
        print(
            json.dumps(
                {
                    "type": "progress",
                    "stage": "目标检测",
                    "current": index,
                    "total": len(images),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration_ms = (time.perf_counter() - started) * 1000
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    predictions_path = output_directory / "predictions.json"
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "protocol_version": "1.0",
        "job_id": request["job_id"],
        "run_id": request["run_id"],
        "status": "succeeded",
        "predictions_path": predictions_path.name,
        "image_count": len(images),
        "prediction_count": len(predictions),
        "unmatched_labels": dict(unmatched),
        "runtime": {
            "model_load_ms": round(model_load_ms, 3),
            "duration_ms": round(duration_ms, 3),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "peak_memory_mb": round(peak_memory_mb, 3),
            "warmup": warmup,
            "batch_size": 1,
            "precision": precision,
        },
        "environment": {
            "versions": _package_versions(),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            ),
            "cuda": torch.version.cuda,
        },
        "warnings": (
            [f"未参与评测的模型类别: {dict(unmatched)}"] if unmatched else []
        ),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["health", "run"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.command == "health":
        health(args.result)
        return
    if not args.request or not args.result:
        parser.error("run 需要 --request 和 --result")
    run(args.request, args.result)


if __name__ == "__main__":
    main()
