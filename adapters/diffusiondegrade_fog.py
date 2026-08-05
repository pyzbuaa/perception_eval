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

from PIL import Image


PROMPT = "a foggy UAV aerial image"
IMAGE_PREP = "resize_512x512"


def blend_fog(source: Image.Image, fogged: Image.Image, strength: float) -> Image.Image:
    if not 0 <= strength <= 1:
        raise ValueError("fog_strength 必须位于 0 到 1 之间")
    return Image.blend(source.convert("RGB"), fogged.convert("RGB"), strength)


def project_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "DiffusionDegrade"
    return Path(os.environ.get("DIFFUSION_DEGRADE_ROOT", default)).expanduser().resolve()


def checkpoint_path() -> Path:
    default = (
        project_root()
        / "outputs"
        / "uav_fog_8gpu_3125_content15"
        / "checkpoints"
        / "model_2501.pkl"
    )
    return Path(
        os.environ.get("DIFFUSION_DEGRADE_UAV_FOG_CHECKPOINT", default)
    ).expanduser().resolve()


def validate_installation() -> tuple[Path, Path]:
    root = project_root()
    checkpoint = checkpoint_path()
    if not (root / "src" / "cyclegan_turbo.py").is_file():
        raise FileNotFoundError(f"DiffusionDegrade 项目不存在或不完整: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"无人机加雾权重不存在: {checkpoint}")
    return root, checkpoint


def load_model(root: Path, checkpoint: Path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_directory = str(root / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)

    import torch
    from cyclegan_turbo import CycleGAN_Turbo
    from my_utils.training_utils import build_transform
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("DiffusionDegrade 无人机加雾仅支持 CUDA，当前进程无法访问 GPU")
    model = CycleGAN_Turbo(pretrained_path=str(checkpoint))
    model.eval()
    if torch.cuda.get_device_capability()[0] < 12:
        model.unet.enable_xformers_memory_efficient_attention()
    model.half()
    return torch, transforms, build_transform(IMAGE_PREP), model


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版 Adapter 协议")
    parameters = request.get("model_parameters", {})
    if parameters.get("effect") != "fog":
        raise ValueError("本适配器当前仅支持 fog 加雾")
    if parameters.get("domain") != "uav_aerial":
        raise ValueError("本适配器当前仅支持无人机航拍域")
    strength = float(parameters.get("fog_strength", 1.0))
    if not 0 <= strength <= 1:
        raise ValueError("fog_strength 必须位于 0 到 1 之间")

    input_links = [
        Path(value).expanduser().absolute()
        for value in request.get("input_images", [])
    ]
    inputs = [path.resolve() for path in input_links]
    count = int(request.get("sample_count", len(inputs)))
    if count < 1 or count != len(inputs):
        raise ValueError("无人机加雾必须处理输入数据集中的全部图像")
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("输入数据集中存在无法读取的图像")
    input_directory = Path(request["input_directory"]).expanduser().resolve()
    relatives = [path.relative_to(input_directory) for path in input_links]

    output_directory = Path(request["output_directory"]).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    root, checkpoint = validate_installation()
    torch, transforms, transform, model = load_model(root, checkpoint)

    seeds = request.get("seeds") or [request.get("seed", 1001)]
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for index, (input_path, relative) in enumerate(zip(inputs, relatives)):
        seed = int(seeds[index % len(seeds)])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with Image.open(input_path) as opened:
            input_image = opened.convert("RGB")
        prepared = transform(input_image)
        tensor = transforms.ToTensor()(prepared)
        tensor = transforms.Normalize([0.5], [0.5])(tensor).unsqueeze(0).cuda().half()
        with torch.no_grad():
            output = model(tensor, direction="a2b", caption=PROMPT)
        output_image = transforms.ToPILImage()(output[0].float().cpu() * 0.5 + 0.5)
        output_image = output_image.resize(input_image.size, Image.Resampling.LANCZOS)
        output_image = blend_fog(input_image, output_image, strength)
        output_path = output_directory / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
                    "stage": "无人机航拍域加雾",
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
                    "model": "DiffusionDegrade CycleGAN-Turbo UAV Fog",
                    "checkpoint": checkpoint.name,
                    "image_prep": IMAGE_PREP,
                    "precision": "FP16",
                    "fog_strength": strength,
                },
                "warnings": [
                    (
                        "加雾保持输出尺寸和文件名；继承标注作为候选真值，冻结前仍需抽查。"
                        if request.get("has_source_annotations")
                        else "源数据未提供 COCO 标注，加雾结果保持未标注状态。"
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
    import torch
    import diffusers
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
