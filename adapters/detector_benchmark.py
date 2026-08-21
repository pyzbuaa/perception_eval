from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


MB = 1024**2


def load_project_adapter() -> ModuleType:
    path = Path.cwd() / "tools" / "inference" / "platform_coco.py"
    if not path.is_file():
        raise FileNotFoundError(f"模型项目缺少推理入口: {path}")
    specification = importlib.util.spec_from_file_location(
        "perception_eval_project_detector", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载模型推理入口: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_elapsed(
    device: torch.device,
    operation: Any,
) -> tuple[Any, float]:
    if device.type != "cuda":
        started = time.perf_counter()
        value = operation()
        return value, (time.perf_counter() - started) * 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = operation()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end))


def model_complexity(
    model: torch.nn.Module,
    sample: torch.Tensor,
    backend: str,
) -> dict[str, Any]:
    parameters_total = sum(parameter.numel() for parameter in model.parameters())
    parameters_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    flops = 0.0
    profiler = "torch.profiler"
    unsupported_ops: list[str] = []
    if backend == "ultralytics":
        try:
            from ultralytics.utils.torch_utils import get_flops

            flops = float(get_flops(model, imgsz=list(sample.shape[-2:]))) * 1e9
            profiler = "ultralytics.get_flops(thop)"
        except Exception as exc:
            unsupported_ops.append(f"Ultralytics FLOPs 分析失败: {exc}")
    else:
        unsupported_ops.append(
            "torch.profiler 仅统计支持 FLOPs 的算子，自定义算子可能未计入"
        )
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if sample.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
            ) as profile:
                with torch.inference_mode():
                    model(sample)
                synchronize(sample.device)
            flops = float(sum(event.flops or 0 for event in profile.key_averages()))
        except Exception as exc:
            unsupported_ops.append(f"PyTorch FLOPs 分析失败: {exc}")
    if flops <= 0:
        unsupported_ops.append("分析器未统计到完整算子 FLOPs")
        flops = 0.0
    return {
        "parameters_total": int(parameters_total),
        "parameters_trainable": int(parameters_trainable),
        "input_shape": list(sample.shape),
        "macs": flops / 2 if flops else None,
        "flops": flops or None,
        "scope": "forward_only",
        "profiler": profiler,
        "flop_convention": "1_MAC_equals_2_FLOPs",
        "unsupported_ops": unsupported_ops,
    }


def append_ultralytics_predictions(
    predictions: list[dict[str, Any]],
    batch_images: list[dict[str, Any]],
    results: list[Any],
    label_mapping: dict[int, int],
    names: dict[int, str],
    confidence: float,
    unmatched: dict[str, int],
) -> None:
    for image, result in zip(batch_images, results):
        height, width = result.orig_shape
        boxes = result.boxes
        for label, box, score in zip(
            boxes.cls.detach().cpu().tolist(),
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
        ):
            label = int(label)
            score = float(score)
            if score < confidence or not math.isfinite(score):
                continue
            category_id = label_mapping.get(label)
            if category_id is None:
                name = names.get(label, str(label))
                unmatched[name] = unmatched.get(name, 0) + 1
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            x1 = min(max(x1, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            x2 = min(max(x2, 0.0), float(width))
            y2 = min(max(y2, 0.0), float(height))
            if x2 > x1 and y2 > y1:
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )


def run_ultralytics(
    args: argparse.Namespace,
    project: ModuleType,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, int], list[str]]:
    images, categories = project.load_coco_inputs(args.annotations, args.images)
    load_started = time.perf_counter()
    model = project.YOLO(args.weights, task="detect")
    model_load_ms = (time.perf_counter() - load_started) * 1000
    names = project.model_names(model.names)
    label_mapping, unmatched_labels = project.category_mapping(categories, names)
    predict_kwargs = project.prediction_arguments(args)
    sample = torch.empty(
        (1, 3, args.input_height, args.input_width),
        device=device,
        dtype=torch.float16 if args.precision == "FP16" else torch.float32,
    )
    complexity = model_complexity(model.model, sample, "ultralytics")
    for _ in range(args.warmup):
        model.predict(source=[str(images[0]["resolved_path"])], **predict_kwargs)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runtime = {
        "timing_source": "ultralytics_speed+cuda_synchronized_wall_clock",
        "batch_size": args.batch_size,
        "model_load_ms": model_load_ms,
        "preprocess_ms": [],
        "inference_ms": [],
        "postprocess_ms": [],
        "end_to_end_ms": [],
        "batch_duration_ms": [],
        "batch_image_counts": [],
    }
    predictions: list[dict[str, Any]] = []
    unmatched: dict[str, int] = {}
    for offset in range(0, len(images), args.batch_size):
        batch_images = images[offset : offset + args.batch_size]
        synchronize(device)
        started = time.perf_counter()
        results = model.predict(
            source=[str(image["resolved_path"]) for image in batch_images],
            **predict_kwargs,
        )
        synchronize(device)
        batch_ms = (time.perf_counter() - started) * 1000
        if len(results) != len(batch_images):
            raise RuntimeError("Ultralytics 返回的结果数量与输入图片数量不一致")
        runtime["batch_duration_ms"].append(batch_ms)
        runtime["batch_image_counts"].append(len(batch_images))
        runtime["end_to_end_ms"].extend([batch_ms / len(batch_images)] * len(batch_images))
        for result in results:
            speed = result.speed or {}
            runtime["preprocess_ms"].append(float(speed.get("preprocess", 0.0)))
            runtime["inference_ms"].append(float(speed.get("inference", 0.0)))
            runtime["postprocess_ms"].append(float(speed.get("postprocess", 0.0)))
        append_ultralytics_predictions(
            predictions,
            batch_images,
            results,
            label_mapping,
            names,
            args.confidence,
            unmatched,
        )
        emit_progress("Ultralytics 目标检测", offset + len(batch_images), len(images))
    return predictions, runtime, complexity, unmatched, unmatched_labels


def append_transformer_predictions(
    predictions: list[dict[str, Any]],
    batch_images: list[dict[str, Any]],
    sizes: list[tuple[int, int]],
    results: list[dict[str, torch.Tensor]],
    label_mapping: dict[int, int],
    labels: tuple[str, ...],
    confidence: float,
    unmatched: dict[str, int],
) -> None:
    for image, size, result in zip(batch_images, sizes, results):
        width, height = size
        for label, box, score in zip(
            result["labels"].detach().cpu().tolist(),
            result["boxes"].detach().cpu().tolist(),
            result["scores"].detach().cpu().tolist(),
        ):
            label = int(label)
            score = float(score)
            if score < confidence or not math.isfinite(score):
                continue
            category_id = label_mapping.get(label)
            if category_id is None:
                name = labels[label] if 0 <= label < len(labels) else str(label)
                unmatched[name] = unmatched.get(name, 0) + 1
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            x1 = min(max(x1, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            x2 = min(max(x2, 0.0), float(width))
            y2 = min(max(y2, 0.0), float(height))
            if x2 > x1 and y2 > y1:
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )


def run_transformer(
    args: argparse.Namespace,
    project: ModuleType,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, int], list[str]]:
    images, label_mapping, unmatched_labels = project.load_coco_inputs(
        args.annotations, args.images
    )
    input_size = (args.input_height, args.input_width)
    load_started = time.perf_counter()
    if args.backend == "rtdetrv2":
        cfg, model, postprocessor = project.load_model(
            args.config, args.weights, device, input_size, args.max_detections
        )
    else:
        cfg, model, postprocessor = project.load_model(
            args.config, args.weights, device, input_size
        )
    model_load_ms = (time.perf_counter() - load_started) * 1000
    spatial_size = cfg.yaml_cfg.get("eval_spatial_size", list(input_size))
    transform = project.T.Compose(
        [project.T.Resize((int(spatial_size[0]), int(spatial_size[1]))), project.T.ToTensor()]
    )
    warmup_tensor, warmup_size = project.preprocess_image(
        images[0]["resolved_path"], transform
    )
    sample = warmup_tensor.unsqueeze(0).to(device)
    sample_sizes = torch.tensor([warmup_size], dtype=torch.int64, device=device)
    complexity = model_complexity(model, sample, args.backend)
    for _ in range(args.warmup):
        project.run_batch(
            model, postprocessor, sample, sample_sizes, device, args.precision
        )
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runtime = {
        "timing_source": "cuda_event+cuda_synchronized_wall_clock",
        "batch_size": args.batch_size,
        "model_load_ms": model_load_ms,
        "preprocess_ms": [],
        "h2d_ms": [],
        "inference_ms": [],
        "postprocess_ms": [],
        "end_to_end_ms": [],
        "batch_duration_ms": [],
        "batch_image_counts": [],
    }
    predictions: list[dict[str, Any]] = []
    unmatched: dict[str, int] = {}
    for offset in range(0, len(images), args.batch_size):
        batch_images = images[offset : offset + args.batch_size]
        synchronize(device)
        batch_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        prepared = [
            project.preprocess_image(image["resolved_path"], transform)
            for image in batch_images
        ]
        tensors = [value[0] for value in prepared]
        sizes = [value[1] for value in prepared]
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        def transfer() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.stack(tensors).to(device),
                torch.tensor(sizes, dtype=torch.int64, device=device),
            )

        (samples, original_sizes), h2d_ms = cuda_elapsed(device, transfer)

        def forward() -> Any:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.precision == "FP16",
            ):
                return model(samples)

        outputs, inference_ms = cuda_elapsed(device, forward)

        def postprocess() -> Any:
            with torch.inference_mode():
                return postprocessor(outputs, original_sizes)

        results, gpu_postprocess_ms = cuda_elapsed(device, postprocess)
        cpu_postprocess_started = time.perf_counter()
        append_transformer_predictions(
            predictions,
            batch_images,
            sizes,
            results,
            label_mapping,
            project.VISDRONE_LABELS,
            args.confidence,
            unmatched,
        )
        cpu_postprocess_ms = (time.perf_counter() - cpu_postprocess_started) * 1000
        batch_ms = (time.perf_counter() - batch_started) * 1000
        image_count = len(batch_images)
        runtime["preprocess_ms"].extend([preprocess_ms / image_count] * image_count)
        runtime["h2d_ms"].extend([h2d_ms / image_count] * image_count)
        runtime["inference_ms"].extend([inference_ms / image_count] * image_count)
        runtime["postprocess_ms"].extend(
            [(gpu_postprocess_ms + cpu_postprocess_ms) / image_count] * image_count
        )
        runtime["end_to_end_ms"].extend([batch_ms / image_count] * image_count)
        runtime["batch_duration_ms"].append(batch_ms)
        runtime["batch_image_counts"].append(image_count)
        emit_progress(
            "RT-DETRv2 目标检测" if args.backend == "rtdetrv2" else "D-FINE 目标检测",
            offset + image_count,
            len(images),
        )
    return predictions, runtime, complexity, unmatched, unmatched_labels


def emit_progress(stage: str, current: int, total: int) -> None:
    print(
        json.dumps(
            {"type": "progress", "stage": stage, "current": current, "total": total},
            ensure_ascii=False,
        ),
        flush=True,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    request = json.loads(args.request.read_text(encoding="utf-8"))
    project = load_project_adapter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法执行性能评测")
    if args.precision == "FP16" and device.type != "cuda":
        raise ValueError("FP16 性能评测需要 CUDA")
    args.weights = args.weights.expanduser().resolve()
    args.images = args.images.expanduser().resolve()
    args.annotations = args.annotations.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.config is not None:
        args.config = args.config.expanduser().resolve()
    if args.backend == "ultralytics":
        predictions, runtime, complexity, unmatched, unmatched_labels = run_ultralytics(
            args, project, device
        )
    else:
        predictions, runtime, complexity, unmatched, unmatched_labels = run_transformer(
            args, project, device
        )
    synchronize(device)
    memory = {
        "torch_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / MB if device.type == "cuda" else 0.0
        ),
        "torch_peak_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / MB if device.type == "cuda" else 0.0
        ),
    }
    runtime.update(memory)
    write_json(args.output, predictions)
    write_json(
        args.result,
        {
            "protocol_version": "1.0",
            "benchmark_schema_version": "1.0",
            "job_id": request["job_id"],
            "run_id": request["run_id"],
            "status": "succeeded",
            "predictions_path": args.output.name,
            "image_count": len(runtime["end_to_end_ms"]),
            "runtime": runtime,
            "memory": memory,
            "complexity": complexity,
            "unmatched_labels": {
                "dataset": unmatched_labels,
                "predictions": unmatched,
            },
            "warnings": complexity["unsupported_ops"],
            "environment": {
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "backend": args.backend,
                "timing": runtime["timing_source"],
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ultralytics", "rtdetrv2", "dfine"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("FP16", "FP32"), default="FP16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--input-height", type=int, default=1280)
    parser.add_argument("--input-width", type=int, default=1280)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
