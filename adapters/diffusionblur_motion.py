#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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


def parse_parameters(request: dict[str, Any]) -> tuple[str, float, int, Path | None]:
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
    condition_value = str(parameters.get("condition_directory") or "").strip()
    condition_directory = (
        Path(condition_value).expanduser().resolve() if condition_value else None
    )
    if condition_directory and not condition_directory.is_dir():
        raise FileNotFoundError(f"运动条件目录不存在: {condition_directory}")
    return motion, strength, sample_timesteps, condition_directory


def index_condition_files(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.npy")):
        key = path.stem
        if key.endswith("_condition"):
            key = key[: -len("_condition")]
        if key in indexed:
            raise ValueError(
                f"运动条件目录存在重复文件名: {indexed[key].name} / {path.name}"
            )
        indexed[key] = path
    return indexed


def load_motion_condition(
    path: Path,
    height: int,
    width: int,
    flow_norm: float = 147.0,
) -> tuple[np.ndarray, str]:
    condition = np.asarray(np.load(path, allow_pickle=False))
    if condition.ndim != 3:
        raise ValueError(f"运动条件必须是三维数组: {path}")
    if condition.shape[0] == 3:
        pass
    elif condition.shape[-1] == 3:
        condition = np.moveaxis(condition, -1, 0)
    else:
        raise ValueError(f"运动条件必须为 [3,H,W] 或 [H,W,3]: {path}")
    expected = (3, height, width)
    if condition.shape != expected:
        raise ValueError(
            f"运动条件尺寸 {condition.shape} 与图片尺寸 {expected} 不一致: {path.name}"
        )
    condition = condition.astype(np.float32, copy=True)
    if not np.isfinite(condition).all():
        raise ValueError(f"运动条件包含 NaN 或 Inf: {path.name}")
    direction_max = float(np.abs(condition[:2]).max())
    if direction_max > 1.001:
        raise ValueError(f"运动条件方向分量必须位于 [-1,1]: {path.name}")
    if np.any(condition[2] < 0):
        raise ValueError(f"运动条件幅度不能为负数: {path.name}")
    magnitude_max = float(condition[2].max())
    mode = "raft-raw" if magnitude_max > 1.000001 else "normalized"
    if mode == "raft-raw":
        condition[2] = np.clip(condition[2] / flow_norm, 0.0, 1.0)
    else:
        condition[2] = np.clip(condition[2], 0.0, 1.0)
    condition[:2] = np.clip(condition[:2], -1.0, 1.0)
    condition[:2, condition[2] <= 1e-8] = 0.0
    if float(condition[2].max()) <= 0:
        raise ValueError(f"运动条件幅度全部为零: {path.name}")
    return condition, mode


def pad_condition(condition: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
    pad_height, pad_width = padding
    if not pad_height and not pad_width:
        return condition
    return np.pad(
        condition,
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode="reflect",
    )


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版 Adapter 协议")
    motion, strength, sample_timesteps, condition_directory = parse_parameters(request)
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
    motion_generator = (
        generator_class(config_class(motion=motion, mean_strength=strength))
        if condition_directory is None
        else None
    )
    condition_files = (
        index_condition_files(condition_directory) if condition_directory else {}
    )

    seeds = request.get("seeds") or [request.get("seed", 2023)]
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    matched_conditions = 0
    fallback_conditions = 0
    for index, (input_path, relative) in enumerate(zip(inputs, relatives)):
        seed = int(seeds[index % len(seeds)]) + index
        with Image.open(input_path) as opened:
            input_image = np.array(opened.convert("RGB"), dtype=np.uint8, copy=True)
        height, width = input_image.shape[:2]
        padded, padding = pad_image(input_image)
        condition_path = condition_files.get(relative.stem)
        condition_mode = None
        used_motion = motion
        if condition_path:
            condition, condition_mode = load_motion_condition(
                condition_path,
                height,
                width,
            )
            condition = pad_condition(condition, padding)
            condition_source = "file"
            matched_conditions += 1
        else:
            if condition_directory:
                used_motion = random.Random(seed).choice(sorted(MOTIONS))
                fallback_conditions += 1
            assert motion_generator is not None or condition_directory is not None
            generator = motion_generator or generator_class(
                config_class(motion=used_motion, mean_strength=strength)
            )
            condition = generator.generate(padded.shape[0], padded.shape[1])
            condition_source = (
                "random-preset-fallback" if condition_directory else "preset"
            )
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
                "motion": used_motion,
                "mean_strength": float(condition[2, :height, :width].mean()),
                "condition_source": condition_source,
                "condition_file": (
                    str(condition_path.relative_to(condition_directory))
                    if condition_path and condition_directory
                    else None
                ),
                "condition_mode": condition_mode,
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
                    "motion": motion if condition_directory is None else "condition-files",
                    "strength": strength,
                    "sample_timesteps": sample_timesteps,
                    "precision": "FP32",
                    "condition_matching": "filename",
                    "matched_conditions": matched_conditions,
                    "fallback_conditions": fallback_conditions,
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
