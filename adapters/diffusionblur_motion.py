#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MOTIONS = {
    "forward",
    "backward",
    "fly-left",
    "fly-right",
    "ascend",
    "descend",
    "yaw-left",
    "yaw-right",
    "tilt-up",
    "tilt-down",
    "tilt-left",
    "tilt-right",
    "vibration",
}


def project_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "DiffusionBlur"
    return Path(os.environ.get("DIFFUSION_BLUR_ROOT", default)).expanduser().resolve()


def checkpoint_path() -> Path:
    default = project_root() / "weights" / "ID_Blau.pth"
    return Path(os.environ.get("DIFFUSION_BLUR_CHECKPOINT", default)).expanduser().resolve()


def validate_installation() -> tuple[Path, Path]:
    root = project_root()
    checkpoint = checkpoint_path()
    if not (root / "reblur" / "pipeline.py").is_file():
        raise FileNotFoundError(f"DiffusionBlur 项目不存在或不完整: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ID-Blau 权重不存在: {checkpoint}")
    return root, checkpoint


def load_components(root: Path, checkpoint: Path):
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)

    import torch
    from reblur.conditions.camera_motion import (
        CameraMotionConditionGenerator,
        CameraMotionConfig,
    )
    from reblur.pipeline import IDBlauPipeline, pad_image

    if not torch.cuda.is_available():
        raise RuntimeError("DiffusionBlur 无人机运动模糊仅支持 CUDA，当前进程无法访问 GPU")
    pipeline = IDBlauPipeline.from_checkpoint(checkpoint, "cuda", sampler="DDIM")
    torch.backends.cudnn.benchmark = True
    return CameraMotionConditionGenerator, CameraMotionConfig, pad_image, pipeline


def parse_parameters(request: dict[str, Any]) -> tuple[str, float, int]:
    parameters = request.get("model_parameters", {})
    if parameters.get("effect") != "motion_blur":
        raise ValueError("本适配器当前仅支持 motion_blur")
    if parameters.get("domain") != "uav_aerial":
        raise ValueError("本适配器当前仅支持无人机航拍域")
    motion = str(parameters.get("motion", "forward"))
    if motion not in MOTIONS:
        raise ValueError(f"不支持的无人机运动类型: {motion}")
    strength = float(parameters.get("strength", 0.14))
    if not 0.01 <= strength <= 0.35:
        raise ValueError("运动模糊强度必须位于 0.01 到 0.35 之间")
    sample_timesteps = int(parameters.get("sample_timesteps", 20))
    if sample_timesteps != 20:
        raise ValueError("当前平台固定使用 20 个 DDIM 采样步")
    return motion, strength, sample_timesteps


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版 Adapter 协议")
    motion, strength, sample_timesteps = parse_parameters(request)
    input_links = [
        Path(value).expanduser().absolute()
        for value in request.get("input_images", [])
    ]
    inputs = [path.resolve() for path in input_links]
    count = int(request.get("sample_count", len(inputs)))
    if count < 1 or count != len(inputs):
        raise ValueError("无人机运动模糊必须处理输入数据集中的全部图像")
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("输入数据集中存在无法读取的图像")
    input_directory = Path(request["input_directory"]).expanduser().resolve()
    relatives = [path.relative_to(input_directory) for path in input_links]

    output_directory = Path(request["output_directory"]).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    root, checkpoint = validate_installation()
    generator_class, config_class, pad_image, pipeline = load_components(
        root, checkpoint
    )
    motion_generator = generator_class(
        config_class(motion=motion, mean_strength=strength)
    )

    seeds = request.get("seeds") or [request.get("seed", 2023)]
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for index, (input_path, relative) in enumerate(zip(inputs, relatives)):
        seed = int(seeds[index % len(seeds)]) + index
        with Image.open(input_path) as opened:
            input_image = np.array(opened.convert("RGB"), dtype=np.uint8, copy=True)
        height, width = input_image.shape[:2]
        padded, _ = pad_image(input_image)
        condition = motion_generator.generate(padded.shape[0], padded.shape[1])
        output = pipeline.generate(
            padded,
            condition,
            sample_timesteps=sample_timesteps,
            seed=seed,
        )[:height, :width]
        output_path = output_directory / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(output_path)
        samples.append(
            {
                "sample_id": f"{request['job_id']}-{index + 1}",
                "image_path": relative.as_posix(),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "width": width,
                "height": height,
                "seed": seed,
                "annotation_status": (
                    "CANDIDATE" if request.get("has_source_annotations") else "UNLABELED"
                ),
                "source_path": str(input_path),
                "motion": motion,
                "mean_strength": float(condition[2, :height, :width].mean()),
            }
        )
        print(
            json.dumps(
                {
                    "type": "progress",
                    "stage": "无人机运动模糊生成",
                    "current": index + 1,
                    "total": count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result_path.write_text(
        json.dumps(
            {
                "protocol_version": "1.0",
                "job_id": request["job_id"],
                "status": "succeeded",
                "samples": samples,
                "has_candidate_annotations": bool(request.get("has_source_annotations")),
                "runtime": {
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "model": "ID-Blau UAV Motion Blur",
                    "checkpoint": checkpoint.name,
                    "motion": motion,
                    "strength": strength,
                    "sample_timesteps": sample_timesteps,
                    "precision": "FP32",
                },
                "warnings": [
                    (
                        "运动模糊保持输出尺寸和文件名；继承标注作为候选真值，冻结前仍需抽查。"
                        if request.get("has_source_annotations")
                        else "源数据未提供 COCO 标注，运动模糊结果保持未标注状态。"
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def health() -> None:
    root, checkpoint = validate_installation()
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    import cv2
    import torch
    from reblur.pipeline import IDBlauPipeline

    print(
        json.dumps(
            {
                "project_root": str(root),
                "checkpoint": str(checkpoint),
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "pipeline": IDBlauPipeline.__name__,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "health"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.command == "health":
        health()
        return
    if not args.request or not args.result:
        parser.error("run 需要 --request 和 --result")
    run(args.request, args.result)


if __name__ == "__main__":
    main()
