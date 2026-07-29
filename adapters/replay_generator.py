#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def render_svg(index: int, blur: float, fog: float) -> str:
    offset = (index * 37) % 180
    soft = blur * 7
    veil = min(0.75, fog)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 540">
<defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#63b7cf"/><stop offset="1" stop-color="#d8e9e7"/></linearGradient>
<filter id="motion"><feGaussianBlur stdDeviation="{soft:.2f} 0.5"/></filter></defs>
<rect width="960" height="540" fill="url(#sky)"/><g filter="url(#motion)">
<path d="M0 240L960 190V540H0Z" fill="#567c58"/><path d="M100 540L390 200H540L850 540Z" fill="#dfd3af"/>
<g fill="#d86542"><rect x="{310+offset}" y="335" width="92" height="44" rx="8"/><circle cx="{330+offset}" cy="380" r="11" fill="#263645"/><circle cx="{382+offset}" cy="380" r="11" fill="#263645"/></g>
<g fill="#e8c84f"><rect x="{610-offset}" y="280" width="72" height="38" rx="7"/><circle cx="{626-offset}" cy="320" r="9" fill="#263645"/><circle cx="{666-offset}" cy="320" r="9" fill="#263645"/></g></g>
<rect width="960" height="540" fill="#e9eef0" opacity="{veil:.2f}"/>
<g fill="none" stroke="#19d3ff" stroke-width="5"><rect x="{297+offset}" y="318" width="120" height="78"/><rect x="{598-offset}" y="264" width="98" height="70"/></g>
<g font-family="sans-serif" font-size="20" fill="white"><rect x="{297+offset}" y="288" width="135" height="30" fill="#087f9c"/><text x="{306+offset}" y="310">vehicle 0.91</text></g>
<text x="24" y="510" fill="white" font-family="sans-serif" font-size="20">REPLAY FIXTURE · FLOW VALIDATION ONLY</text>
</svg>"""


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = Path(request["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    sensor = request.get("conditions", {}).get("sensor", {})
    blur = float(sensor.get("motion_blur", sensor.get("motion_blur_level", 0.0)) or 0)
    fog = float(sensor.get("fog_density", 0.0) or 0)
    samples = []
    started = time.perf_counter()
    count = int(request["sample_count"])
    seeds = request.get("seeds") or [request.get("seed", 1001)]
    for index in range(count):
        name = f"sample-{index + 1:04d}.svg"
        path = output / name
        content = render_svg(index, blur, fog)
        path.write_text(content, encoding="utf-8")
        samples.append(
            {
                "sample_id": f"{request['job_id']}-{index + 1}",
                "image_path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": 960,
                "height": 540,
                "seed": seeds[index % len(seeds)],
                "annotation_status": "CANDIDATE",
            }
        )
        print(json.dumps({"type": "progress", "current": index + 1, "total": count}), flush=True)
    result = {
        "protocol_version": "1.0",
        "job_id": request["job_id"],
        "status": "succeeded",
        "samples": samples,
        "has_candidate_annotations": True,
        "runtime": {"duration_ms": round((time.perf_counter() - started) * 1000, 2)},
        "warnings": ["回放数据仅用于软件流程验证，不代表生成模型性能。"],
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.result)


if __name__ == "__main__":
    main()

