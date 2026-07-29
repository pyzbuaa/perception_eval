#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageFilter


def apply_conditions(image: Image.Image, blur: float, fog: float, noise: float, seed: int) -> Image.Image:
    output = image.convert("RGB")
    if blur > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=max(0.1, blur * 8)))
    if fog > 0:
        veil = Image.new("RGB", output.size, (232, 239, 241))
        output = Image.blend(output, veil, min(0.78, fog * 0.68))
    if noise > 0:
        rng = random.Random(seed)
        # Pillow's effect_noise is deterministic only through its own RNG, so use a
        # small seeded luminance offset for a reproducible protocol fixture.
        offset = rng.randint(-int(noise * 30), int(noise * 30))
        noise_layer = Image.new("RGB", output.size, (128 + offset,) * 3)
        output = Image.blend(output, noise_layer, min(0.22, noise * 0.18))
    return output


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output_directory = Path(request["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    inputs = [Path(value) for value in request.get("input_images", [])]
    if not inputs:
        raise ValueError("input_images 不能为空")
    sensor = request.get("conditions", {}).get("sensor", {})
    blur = float(sensor.get("motion_blur", 0) or 0)
    fog = float(sensor.get("fog_density", 0) or 0)
    noise = float(sensor.get("noise_level", 0) or 0)
    seeds = request.get("seeds") or [request.get("seed", 1001)]
    count = min(int(request.get("sample_count", len(inputs))), len(inputs))
    started = time.perf_counter()
    samples = []
    for index, input_path in enumerate(inputs[:count]):
        seed = int(seeds[index % len(seeds)])
        with Image.open(input_path) as image:
            transformed = apply_conditions(image, blur, fog, noise, seed)
            output_path = output_directory / f"condition-{index + 1:04d}.png"
            transformed.save(output_path, format="PNG", optimize=True)
        samples.append(
            {
                "sample_id": f"{request['job_id']}-{index + 1}",
                "image_path": output_path.name,
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "width": transformed.width,
                "height": transformed.height,
                "seed": seed,
                "annotation_status": "CANDIDATE",
                "source_path": str(input_path),
            }
        )
        print(json.dumps({"type": "progress", "current": index + 1, "total": count}), flush=True)
    result_path.write_text(
        json.dumps(
            {
                "protocol_version": "1.0",
                "job_id": request["job_id"],
                "status": "succeeded",
                "samples": samples,
                "has_candidate_annotations": True,
                "runtime": {"duration_ms": round((time.perf_counter() - started) * 1000, 2)},
                "warnings": ["成像退化不改变几何位置；正式冻结前仍需抽查真值框。"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.result)


if __name__ == "__main__":
    main()

