from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def evaluate_coco_predictions(
    annotation_path: Path,
    predictions_path: Path,
    runtime: dict[str, Any],
    category_ids: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(annotation_path.read_text(encoding="utf-8"))
    recall_curve = [round(index / 100, 2) for index in range(101)]
    precision_curve = [0.0] * 101
    stats = [0.0] * 12
    if predictions:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_ground_truth = COCO(str(annotation_path))
            coco_predictions = coco_ground_truth.loadRes(predictions)
            evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
            evaluator.params.imgIds = [
                int(image["id"]) for image in ground_truth.get("images", [])
            ]
            evaluator.params.catIds = category_ids or [
                int(category["id"])
                for category in ground_truth.get("categories", [])
            ]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        stats = [max(0.0, float(value)) for value in evaluator.stats]
        raw_precision = evaluator.eval["precision"][0, :, :, 0, -1]
        for recall_index in range(raw_precision.shape[0]):
            values = raw_precision[recall_index]
            valid = values[values >= 0]
            if valid.size:
                precision_curve[recall_index] = float(valid.mean())

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for recall, precision in zip(recall_curve, precision_curve):
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    preprocess = [float(value) for value in runtime.get("preprocess_ms", [])]
    inference = [float(value) for value in runtime.get("inference_ms", [])]
    postprocess = [float(value) for value in runtime.get("postprocess_ms", [])]
    count = max(len(preprocess), len(inference), len(postprocess))
    end_to_end = [
        (preprocess[index] if index < len(preprocess) else 0.0)
        + (inference[index] if index < len(inference) else 0.0)
        + (postprocess[index] if index < len(postprocess) else 0.0)
        for index in range(count)
    ]
    if not end_to_end and runtime.get("duration_ms") and ground_truth.get("images"):
        average = float(runtime["duration_ms"]) / len(ground_truth["images"])
        end_to_end = [average] * len(ground_truth["images"])
    mean_latency = sum(end_to_end) / len(end_to_end) if end_to_end else 0.0
    metrics = {
        "map": round(stats[0], 6),
        "map50": round(stats[1], 6),
        "map75": round(stats[2], 6),
        "map_small": round(stats[3], 6),
        "map_medium": round(stats[4], 6),
        "map_large": round(stats[5], 6),
        "ar100": round(stats[8], 6),
        "precision": round(best_precision, 6),
        "recall": round(best_recall, 6),
        "f1": round(best_f1, 6),
        "latency_p50": round(_percentile(end_to_end, 0.5), 3),
        "latency_p95": round(_percentile(end_to_end, 0.95), 3),
        "fps": round(1000 / mean_latency, 3) if mean_latency else 0.0,
        "peak_memory": round(float(runtime.get("peak_memory_mb", 0.0)), 3),
        "model_load_ms": round(float(runtime.get("model_load_ms", 0.0)), 3),
        "metric_protocol": "pycocotools-2.0.11",
    }
    curves = {
        "recall": recall_curve,
        "precision": [round(value, 6) for value in precision_curve],
    }
    return metrics, curves
