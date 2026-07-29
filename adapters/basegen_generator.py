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
from typing import Any, TextIO


DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"
DOMAIN_MAP = {
    "城市驾驶": "autonomous-driving",
    "无人机航拍": "low-altitude-uav",
    "野外自动驾驶": "offroad-autonomous-driving",
    "autonomous-driving": "autonomous-driving",
    "low-altitude-uav": "low-altitude-uav",
    "offroad-autonomous-driving": "offroad-autonomous-driving",
}
WEATHER_MAP = {
    "晴朗": "clear",
    "阴天": "overcast",
    "雾": "fog",
    "雨": "light_rain",
    "雪": "snow",
    "clear": "clear",
    "overcast": "overcast",
    "fog": "fog",
    "light_rain": "light_rain",
    "snow": "snow",
}


def basegen_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "BaseGen"
    return Path(os.environ.get("BASEGEN_ROOT", default)).expanduser().resolve()


def load_basegen(root: Path):
    if not (root / "zimage_gen" / "runner.py").is_file():
        raise FileNotFoundError(f"BaseGen 项目不存在或不完整: {root}")
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    from zimage_gen.prompts import compile_prompt
    from zimage_gen.runner import run_plan
    from zimage_gen.scenes import load_catalog, sample_scene, validate_scene

    return compile_prompt, run_plan, load_catalog, sample_scene, validate_scene


def parse_resolution(value: Any, domain: str) -> tuple[int, int]:
    if value is None:
        return (1024, 1024 if domain == "low-altitude-uav" else 576)
    if isinstance(value, list) and len(value) == 2:
        width, height = (int(item) for item in value)
    elif isinstance(value, str):
        normalized = value.lower().replace("×", "x")
        parts = normalized.split("x")
        if len(parts) != 2:
            raise ValueError(f"无法解析分辨率: {value}")
        width, height = (int(item.strip()) for item in parts)
    else:
        raise ValueError(f"无法解析分辨率: {value}")
    if width < 1 or height < 1 or width % 16 or height % 16:
        raise ValueError("BaseGen 分辨率必须为正数且宽高均能被 16 整除")
    return width, height


def _selection(
    selections: dict[str, Any], field: str, multi: bool = False
) -> dict[str, Any]:
    selection = selections.get(field, {"mode": "random"})
    if not isinstance(selection, dict):
        raise ValueError(f"{field} 的选择规则必须是对象")
    mode = selection.get("mode")
    if mode == "random":
        return {"mode": "random"}
    if mode != "fixed":
        raise ValueError(f"{field} 的 mode 必须是 random 或 fixed")
    key = "values" if multi else "value"
    value = selection.get(key)
    if multi:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{field}.values 必须是字符串数组")
        if len(value) > 4:
            raise ValueError("elements 最多选择四项")
    elif not isinstance(value, str):
        raise ValueError(f"{field}.value 必须是字符串")
    return {"mode": "fixed", key: value}


def _weighted_choice(
    values: list[str], options: dict[str, Any], rng: random.Random
) -> str:
    if not values:
        raise ValueError("没有满足固定条件的随机候选值")
    weights = [options[value].get("weight", 1) for value in values]
    return rng.choices(values, weights=weights, k=1)[0]


def _option(
    domain_catalog: dict[str, Any], field: str, value: str
) -> dict[str, Any]:
    options = domain_catalog["fields"][field]["options"]
    if value not in options:
        raise ValueError(f"{field} 的固定值无效: {value}")
    return options[value]


def resolve_scene(
    domain: str,
    catalog: dict[str, Any],
    rng: random.Random,
    index: int,
    selections: dict[str, Any],
    custom: str,
    validate_scene,
) -> dict[str, Any]:
    domain_catalog = catalog["domains"][domain]
    field_names = set(domain_catalog["fields"])
    unknown = set(selections) - field_names
    if unknown:
        raise ValueError(f"未知场景字段: {', '.join(sorted(unknown))}")
    if not isinstance(custom, str):
        raise ValueError("custom 必须是字符串")

    scalar_fields = [
        field
        for field in domain_catalog["fields"]
        if field not in {"elements", "custom"}
    ]
    rules = {
        field: _selection(selections, field)
        for field in scalar_fields
    }
    element_rule = _selection(selections, "elements", multi=True)
    environment_options = domain_catalog["fields"]["environment"]["options"]
    environment_rule = rules["environment"]
    if environment_rule["mode"] == "fixed":
        _option(domain_catalog, "environment", environment_rule["value"])
        environments = [environment_rule["value"]]
    else:
        environments = list(environment_options)

    for field, rule in rules.items():
        if field == "environment" or rule["mode"] == "random":
            continue
        option = _option(domain_catalog, field, rule["value"])
        allowed = option.get("environments")
        if allowed is not None:
            environments = [
                environment for environment in environments if environment in allowed
            ]
    if element_rule["mode"] == "fixed":
        for value in element_rule["values"]:
            option = _option(domain_catalog, "elements", value)
            allowed = option.get("environments")
            if allowed is not None:
                environments = [
                    environment
                    for environment in environments
                    if environment in allowed
                ]
    if not environments:
        raise ValueError("固定场景字段之间不兼容，没有可用的 environment")

    environment = (
        environments[0]
        if environment_rule["mode"] == "fixed"
        else _weighted_choice(environments, environment_options, rng)
    )
    sampled_fields: dict[str, str] = {"environment": environment}
    for field in scalar_fields:
        if field == "environment":
            continue
        rule = rules[field]
        options = domain_catalog["fields"][field]["options"]
        if rule["mode"] == "fixed":
            sampled_fields[field] = rule["value"]
            continue
        values = [
            value
            for value, option in options.items()
            if "environments" not in option
            or environment in option["environments"]
        ]
        sampled_fields[field] = _weighted_choice(values, options, rng)

    element_options = domain_catalog["fields"]["elements"]["options"]
    if element_rule["mode"] == "fixed":
        elements = element_rule["values"]
    else:
        element_values = [
            value
            for value, option in element_options.items()
            if "environments" not in option
            or environment in option["environments"]
        ]
        count = rng.randint(1, min(3, len(element_values)))
        elements = rng.sample(element_values, count)
    scene = {
        "version": "1.0",
        "scene_id": f"{domain}-configured-{index:04d}",
        "domain": domain,
        **sampled_fields,
        "elements": elements,
        "custom": custom,
    }
    validate_scene(scene, catalog)
    return scene


def prepare_plan(
    request: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request.get("protocol_version") != "1.0":
        raise ValueError("仅支持 1.0 版 Adapter 协议")
    count = int(request.get("sample_count", 0))
    if count < 1:
        raise ValueError("sample_count 必须至少为 1")
    seeds = request.get("seeds") or [request.get("seed", 42)]
    start_seed = int(seeds[0])
    if start_seed < 0:
        raise ValueError("seed 必须是非负整数")

    conditions = request.get("conditions", {})
    scene_conditions = conditions.get("scene", {})
    sensor_conditions = conditions.get("sensor", {})
    raw_domain = scene_conditions.get("domain", "")
    if raw_domain not in DOMAIN_MAP:
        raise ValueError(f"BaseGen 不支持场景域: {raw_domain}")
    domain = DOMAIN_MAP[raw_domain]
    width, height = parse_resolution(sensor_conditions.get("resolution"), domain)

    parameters = request.get("model_parameters", {})
    steps = int(parameters.get("steps", 9))
    guidance_scale = float(parameters.get("guidance_scale", 0.0))
    device_policy = parameters.get("device_policy", "cuda")
    local_files_only = parameters.get("local_files_only", False)
    if steps < 1:
        raise ValueError("steps 必须至少为 1")
    if device_policy not in {"cuda", "cpu-offload"}:
        raise ValueError(f"不支持的 device_policy: {device_policy}")
    if not isinstance(local_files_only, bool):
        raise ValueError("local_files_only 必须是布尔值")

    compile_prompt, _, load_catalog, sample_scene, validate_scene = load_basegen(root)
    catalog = load_catalog()
    output_directory = Path(request["output_directory"]).resolve()
    plan = []
    requested_weather = scene_conditions.get("weather", "晴朗")
    selections = scene_conditions.get("fields")
    if selections is not None and not isinstance(selections, dict):
        raise ValueError("scene.fields 必须是对象")
    custom = scene_conditions.get("custom", "")
    for index in range(count):
        seed = start_seed + index
        rng = random.Random(seed)
        if selections is not None:
            scene = resolve_scene(
                domain,
                catalog,
                rng,
                index + 1,
                selections,
                custom,
                validate_scene,
            )
        else:
            scene = sample_scene(domain, catalog, rng, index + 1)
            if requested_weather == "夜间":
                scene["time_of_day"] = "night"
                scene["weather"] = "clear"
            else:
                if requested_weather not in WEATHER_MAP:
                    raise ValueError(f"BaseGen 不支持天气条件: {requested_weather}")
                scene["weather"] = WEATHER_MAP[requested_weather]
            validate_scene(scene, catalog)
        prompt, template_id = compile_prompt(scene, catalog, rng)
        plan.append(
            {
                "output": str(output_directory / f"sample-{index + 1:04d}.png"),
                "domain": domain,
                "scene_id": scene["scene_id"],
                "scene_file": None,
                "scene": scene,
                "template_id": template_id,
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "selection_rules": selections,
            }
        )
    config = {
        "model_path": parameters.get("model_path", DEFAULT_MODEL),
        "device_policy": device_policy,
        "local_files_only": local_files_only,
    }
    return plan, config


class ProgressStream:
    def __init__(self, target: TextIO, total: int):
        self.target = target
        self.total = total
        self.current = 0
        self.buffer = ""

    def write(self, text: str) -> int:
        self.target.write(text)
        self.target.flush()
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.startswith(("已保存", "已跳过")):
                self.current += 1
                self.target.write(
                    json.dumps(
                        {
                            "type": "progress",
                            "current": self.current,
                            "total": self.total,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self.target.flush()
        return len(text)

    def flush(self) -> None:
        self.target.flush()


def run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    root = basegen_root()
    plan, config = prepare_plan(request, root)
    _, run_plan, _, _, _ = load_basegen(root)
    started = time.perf_counter()
    original_stdout = sys.stdout
    sys.stdout = ProgressStream(original_stdout, len(plan))
    try:
        records = run_plan(plan, **config)
    finally:
        sys.stdout = original_stdout

    output_directory = Path(request["output_directory"]).resolve()
    samples = []
    for index, record in enumerate(records):
        image_path = Path(record["output"]).resolve()
        samples.append(
            {
                "sample_id": f"{request['job_id']}-{index + 1}",
                "image_path": image_path.relative_to(output_directory).as_posix(),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "width": int(record["width"]),
                "height": int(record["height"]),
                "seed": int(record["seed"]),
                "annotation_status": "UNLABELED",
                "metadata_path": image_path.with_suffix(".json")
                .relative_to(output_directory)
                .as_posix(),
            }
        )
    result = {
        "protocol_version": "1.0",
        "job_id": request["job_id"],
        "status": "succeeded",
        "samples": samples,
        "has_candidate_annotations": False,
        "runtime": {
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": "z-image-turbo",
            "model_path": config["model_path"],
            "device_policy": config["device_policy"],
        },
        "warnings": ["BaseGen 为纯文本条件生成，不提供可直接用于评测的真值标注。"],
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
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
