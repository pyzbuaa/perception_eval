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

from PIL import Image, ImageOps


PROMPT = "a nighttime RGB UAV aerial image"
MODEL_SIZE = 640
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 256


def project_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "DiffusionDegrade"
    return Path(os.environ.get("DIFFUSION_DEGRADE_ROOT", default)).expanduser().resolve()


def checkpoint_path() -> Path:
    default = (
        project_root()
        / "outputs"
        / "uav_daynight_sichuan_3125"
        / "checkpoints"
        / "model_3125.pkl"
    )
    return Path(
        os.environ.get("DIFFUSION_DEGRADE_UAV_DAY_TO_NIGHT_CHECKPOINT", default)
    ).expanduser().resolve()


def validate_installation() -> tuple[Path, Path]:
    root = project_root()
    checkpoint = checkpoint_path()
    if not (root / "scripts" / "infer_uav_daynight_tiled.py").is_file():
        raise FileNotFoundError(f"DiffusionDegrade 无人机弱光入口不存在: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"无人机弱光权重不存在: {checkpoint}")
    return root, checkpoint


def load_translator(root: Path, checkpoint: Path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from scripts.infer_uav_daynight_tiled import TileTranslator

    if not torch.cuda.is_available():
        raise RuntimeError("DiffusionDegrade 无人机弱光仅支持 CUDA，当前进程无法访问 GPU")
    translator = TileTranslator(
        model_path=checkpoint,
        direction="a2b",
        prompt=PROMPT,
        model_size=MODEL_SIZE,
        use_fp16=True,
    )
    return torch, translator


def translate_tiled_image(
    root: Path,
    image: Image.Image,
    translator,
    tile_size: int,
    overlap: int,
) -> Image.Image:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.infer_uav_daynight_tiled import translate_tiled

    return translate_tiled(
        image,
        translator,
        tile_size=tile_size,
        overlap=overlap,
        description="tiles",
    )


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版 Adapter 协议")
    parameters = request.get("model_parameters", {})
    if parameters.get("effect") != "day_to_night":
        raise ValueError("本适配器仅支持 day_to_night")
    if parameters.get("domain") != "uav_aerial":
        raise ValueError("本适配器仅支持无人机航拍域")
    inference_mode = str(parameters.get("inference_mode", "fixed_resolution"))
    if inference_mode not in {"fixed_resolution", "tiled"}:
        raise ValueError("无人机弱光推理方式仅支持 fixed_resolution 或 tiled")
    try:
        tile_size = int(parameters.get("tile_size", DEFAULT_TILE_SIZE))
        overlap = int(parameters.get("overlap", DEFAULT_OVERLAP))
    except (TypeError, ValueError) as exc:
        raise ValueError("分块大小和重叠像素必须是整数") from exc
    if inference_mode == "tiled" and tile_size <= 0:
        raise ValueError("分块大小必须大于 0")
    if inference_mode == "tiled" and not 0 <= overlap < tile_size:
        raise ValueError("重叠像素必须大于等于 0 且小于分块大小")

    input_links = [
        Path(value).expanduser().absolute()
        for value in request.get("input_images", [])
    ]
    inputs = [path.resolve() for path in input_links]
    count = int(request.get("sample_count", len(inputs)))
    if count < 1 or count != len(inputs):
        raise ValueError("无人机弱光必须处理输入数据集中的全部图像")
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("输入数据集中存在无法读取的图像")
    input_directory = Path(request["input_directory"]).expanduser().resolve()
    relatives = [path.relative_to(input_directory) for path in input_links]

    output_directory = Path(request["output_directory"]).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    root, checkpoint = validate_installation()
    torch, translator = load_translator(root, checkpoint)

    seeds = request.get("seeds") or [request.get("seed", 42)]
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for index, (input_path, relative) in enumerate(zip(inputs, relatives)):
        seed = int(seeds[index % len(seeds)])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with Image.open(input_path) as opened:
            input_image = ImageOps.exif_transpose(opened).convert("RGB")
        output_image = (
            translate_tiled_image(
                root,
                input_image,
                translator,
                tile_size,
                overlap,
            )
            if inference_mode == "tiled"
            else translator(input_image)
        )
        output_path = output_directory / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            output_image.save(output_path, quality=95, subsampling=0)
        else:
            output_image.save(output_path)
        samples.append(
            {
                "sample_id": f"{request['job_id']}-{index + 1}",
                "image_path": relative.as_posix(),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "width": output_image.width,
                "height": output_image.height,
                "seed": seed,
                "annotation_status": (
                    "CANDIDATE" if request.get("has_source_annotations") else "UNLABELED"
                ),
                "source_path": str(input_path),
            }
        )
        print(
            json.dumps(
                {
                    "type": "progress",
                    "stage": "无人机弱光生成",
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
                    "model": "DiffusionDegrade CycleGAN-Turbo Sichuan UAV Low-Light",
                    "checkpoint": checkpoint.name,
                    "direction": "a2b",
                    "inference_mode": inference_mode,
                    "image_prep": (
                        "overlap_tiled" if inference_mode == "tiled" else "resize_640x640"
                    ),
                    "model_size": MODEL_SIZE,
                    "output_restore": "original_size",
                    "precision": "FP16",
                    **(
                        {"tile_size": tile_size, "overlap": overlap}
                        if inference_mode == "tiled"
                        else {}
                    ),
                },
                "warnings": [
                    (
                        "弱光生成保持输出尺寸和文件名；继承标注作为候选真值，冻结前需抽查目标几何是否保持。"
                        if request.get("has_source_annotations")
                        else "源数据未提供 COCO 标注，弱光结果保持未标注状态。"
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
    import diffusers
    import torch
    import transformers

    print(
        json.dumps(
            {
                "project_root": str(root),
                "checkpoint": str(checkpoint),
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "transformers": transformers.__version__,
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
