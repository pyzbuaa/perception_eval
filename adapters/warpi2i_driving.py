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


FOG_PROMPT = (
    "Relit with dense fog in a muted outdoor setting, casting soft diffused "
    "shadows and surrounding the subject in pale gray light to create a quiet, "
    "atmospheric mood."
)
NIGHT_PROMPT = "driving in the night"


def project_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "WarpI2I"
    return Path(os.environ.get("WARPI2I_ROOT", default)).expanduser().resolve()


def checkpoint_path(effect: str) -> Path:
    root = project_root()
    if effect == "fog":
        default = (
            root
            / "weights"
            / "pix2pix_turbo"
            / "2_24_drive_v2_warped_128"
            / "foggy_1.pkl"
        )
        variable = "WARPI2I_DRIVING_FOG_CHECKPOINT"
    elif effect == "day_to_night":
        default = (
            root
            / "weights"
            / "cyclegan_turbo"
            / "BDD100K_day2night.pkl"
        )
        variable = "WARPI2I_DRIVING_DAY_TO_NIGHT_CHECKPOINT"
    else:
        raise ValueError(f"不支持的 WarpI2I 效果: {effect}")
    return Path(os.environ.get(variable, default)).expanduser().resolve()


def validate_installation(effect: str) -> tuple[Path, Path]:
    root = project_root()
    checkpoint = checkpoint_path(effect)
    required = "pix2pix_turbo.py" if effect == "fog" else "cyclegan_turbo.py"
    if not (root / "src" / required).is_file():
        raise FileNotFoundError(f"WarpI2I 项目不存在或不完整: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WarpI2I 权重不存在: {checkpoint}")
    return root, checkpoint


def prepare_imports(root: Path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_directory = str(root / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    import torch
    from torchvision import transforms
    import torchvision.transforms.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("WarpI2I 自动驾驶场景转换仅支持 CUDA，当前进程无法访问 GPU")
    return torch, transforms, functional


def load_model(effect: str, root: Path, checkpoint: Path):
    torch, transforms, functional = prepare_imports(root)
    if effect == "fog":
        from pix2pix_turbo import Pix2Pix_Turbo

        model = Pix2Pix_Turbo(pretrained_path=str(checkpoint))
        model.set_eval()
        model.half()
        transform = None
    else:
        from cyclegan_turbo import CycleGAN_Turbo
        from my_utils.training_utils import build_transform

        model = CycleGAN_Turbo(pretrained_path=str(checkpoint))
        model.eval()
        if torch.cuda.get_device_capability()[0] < 12:
            model.unet.enable_xformers_memory_efficient_attention()
        model.half()
        transform = build_transform("resize_512x512")
    return torch, transforms, functional, transform, model


def translate_fog(image: Image.Image, torch, transforms, functional, model) -> Image.Image:
    width = image.width - image.width % 8
    height = image.height - image.height % 8
    if width < 8 or height < 8:
        raise ValueError("WarpI2I 自动驾驶气雾要求图像宽高均不小于 8 像素")
    prepared = image.resize((width, height), Image.Resampling.LANCZOS)
    tensor = functional.to_tensor(prepared).unsqueeze(0).cuda().half()
    with torch.no_grad():
        output = model(tensor, FOG_PROMPT)
    result = transforms.ToPILImage()(output[0].float().cpu() * 0.5 + 0.5)
    return result.resize(image.size, Image.Resampling.LANCZOS)


def translate_day_to_night(image: Image.Image, torch, transforms, transform, model) -> Image.Image:
    prepared = transform(image)
    tensor = transforms.ToTensor()(prepared)
    tensor = transforms.Normalize([0.5], [0.5])(tensor).unsqueeze(0).cuda().half()
    with torch.no_grad():
        output = model(tensor, direction="a2b", caption=NIGHT_PROMPT)
    result = transforms.ToPILImage()(output[0].float().cpu() * 0.5 + 0.5)
    return result.resize(image.size, Image.Resampling.LANCZOS)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0)
    else:
        image.save(path)


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版 Adapter 协议")
    parameters = request.get("model_parameters", {})
    effect = str(parameters.get("effect", ""))
    if effect not in {"fog", "day_to_night"}:
        raise ValueError("WarpI2I 自动驾驶适配器仅支持 fog 或 day_to_night")
    if parameters.get("domain") != "autonomous_driving":
        raise ValueError("WarpI2I 自动驾驶适配器仅支持自动驾驶场景域")
    expected_method = "paired" if effect == "fog" else "unpaired"
    if parameters.get("method") != expected_method:
        raise ValueError(f"{effect} 必须使用 {expected_method} 方法")

    input_links = [
        Path(value).expanduser().absolute()
        for value in request.get("input_images", [])
    ]
    inputs = [path.resolve() for path in input_links]
    count = int(request.get("sample_count", len(inputs)))
    if count < 1 or count != len(inputs):
        raise ValueError("WarpI2I 必须处理输入数据集中的全部图像")
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("输入数据集中存在无法读取的图像")
    input_directory = Path(request["input_directory"]).expanduser().resolve()
    relatives = [path.relative_to(input_directory) for path in input_links]

    output_directory = Path(request["output_directory"]).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    root, checkpoint = validate_installation(effect)
    torch, transforms, functional, transform, model = load_model(
        effect, root, checkpoint
    )

    seeds = request.get("seeds") or [request.get("seed", 42)]
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for index, (input_path, relative) in enumerate(zip(inputs, relatives)):
        seed = int(seeds[index % len(seeds)])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with Image.open(input_path) as opened:
            input_image = ImageOps.exif_transpose(opened).convert("RGB")
        if effect == "fog":
            output_image = translate_fog(
                input_image, torch, transforms, functional, model
            )
        else:
            output_image = translate_day_to_night(
                input_image, torch, transforms, transform, model
            )
        output_path = output_directory / relative
        save_image(output_image, output_path)
        samples.append(
            {
                "sample_id": f"{request['job_id']}-{index + 1}",
                "image_path": relative.as_posix(),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "width": output_image.width,
                "height": output_image.height,
                "seed": seed,
                "annotation_status": (
                    "CANDIDATE"
                    if request.get("has_source_annotations")
                    else "UNLABELED"
                ),
                "source_path": str(input_path),
            }
        )
        print(
            json.dumps(
                {
                    "type": "progress",
                    "stage": (
                        "自动驾驶气雾"
                        if effect == "fog"
                        else "自动驾驶弱光"
                    ),
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
                "has_candidate_annotations": bool(
                    request.get("has_source_annotations")
                ),
                "runtime": {
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                    "model": (
                        "WarpI2I 自动驾驶气雾"
                        if effect == "fog"
                        else "WarpI2I 自动驾驶弱光"
                    ),
                    "checkpoint": checkpoint.name,
                    "method": expected_method,
                    "image_prep": (
                        "multiple_of_8" if effect == "fog" else "resize_512x512"
                    ),
                    "precision": "FP16",
                },
                "warnings": [
                    (
                        "场景转换保持输出尺寸和文件名；继承标注作为候选真值，冻结前需抽查目标几何是否保持。"
                        if request.get("has_source_annotations")
                        else "源数据未提供 COCO 标注，场景转换结果保持未标注状态。"
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def health(effect: str) -> None:
    effects = [effect] if effect in {"fog", "day_to_night"} else ["fog", "day_to_night"]
    roots_and_checkpoints = [validate_installation(item) for item in effects]
    import diffusers
    import torch
    import transformers

    print(
        json.dumps(
            {
                "project_root": str(roots_and_checkpoints[0][0]),
                "checkpoints": {
                    item: str(checkpoint)
                    for item, (_, checkpoint) in zip(effects, roots_and_checkpoints)
                },
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
    parser.add_argument("--effect", choices=["fog", "day_to_night", "all"], default="all")
    args = parser.parse_args()
    if args.command == "health":
        health(args.effect)
        return
    if not args.request or not args.result:
        parser.error("run 需要 --request 和 --result")
    run(args.request, args.result)


if __name__ == "__main__":
    main()
