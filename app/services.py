from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image

from app.category_templates import normalize_categories, normalize_category_name
from app.command_protocol import CommandTemplateError, validate_command_arguments
from app.config import settings
from app.db import (
    Database,
    cached_file_sha256,
    db,
    json_dump,
    json_load,
    new_id,
    utc_now,
)


JSON_FIELDS = {
    "parameter_schema",
    "sensor_conditions",
    "payload",
    "result",
    "config",
    "environment_fingerprint",
    "hardware_profile",
    "metrics",
    "curves",
    "categories",
}

BASEGEN_FIELD_LABELS = {
    "region": "中国区域",
    "camera_height": "相机高度",
    "viewpoint": "观察视角",
    "field_of_view": "视场角",
    "environment": "场景环境",
    "time_of_day": "时间",
    "weather": "天气",
    "activity_level": "活动密度",
    "elements": "关键元素",
    "custom": "自定义描述",
}

def decode_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for field in JSON_FIELDS:
        if field in output and isinstance(output[field], str):
            try:
                output[field] = json.loads(output[field])
            except json.JSONDecodeError:
                pass
    for field in ("requires_gpu", "is_demo", "frozen", "cancel_requested", "is_official"):
        if field in output:
            output[field] = bool(output[field])
    return output


def list_adapters(database: Database = db) -> list[dict[str, Any]]:
    return [decode_row(row) for row in database.rows("SELECT * FROM adapters ORDER BY kind,name")]


def list_datasets(database: Database = db) -> list[dict[str, Any]]:
    rows = database.rows("SELECT * FROM datasets ORDER BY created_at DESC")
    for row in rows:
        row = decode_row(row)
        category_row = database.row(
            "SELECT categories FROM dataset_annotation_schemas WHERE dataset_id=?",
            (row["id"],),
        )
        row["categories"] = (
            json_load(category_row["categories"], []) if category_row else []
        )
        relative = row.get("artifact_path")
        row["preview_images"] = []
        if relative:
            directory = settings.artifact_dir / relative
            if directory.exists():
                row["preview_images"] = [
                    f"/artifacts/{path.relative_to(settings.artifact_dir).as_posix()}"
                    for path in sorted(directory.iterdir())
                    if path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"}
                ][:6]
        yield row


class DatasetDeletionError(ValueError):
    pass


class JobDeletionError(ValueError):
    pass


class DatasetArtifactError(ValueError):
    pass


class BaseGenCatalogError(ValueError):
    pass


class DatasetAnnotationError(ValueError):
    pass


class LocalModelRegistrationError(ValueError):
    pass


class ModelDeletionError(ValueError):
    pass


class CategoryCompatibilityError(ValueError):
    pass


def category_compatibility(
    dataset_id: str,
    model_id: str,
    database: Database = db,
) -> dict[str, Any]:
    dataset = database.row("SELECT name FROM datasets WHERE id=?", (dataset_id,))
    model = database.row("SELECT name,categories FROM models WHERE id=?", (model_id,))
    category_row = database.row(
        "SELECT categories FROM dataset_annotation_schemas WHERE dataset_id=?",
        (dataset_id,),
    )
    if not dataset or not model:
        raise CategoryCompatibilityError("数据集或模型不存在")
    dataset_categories = json_load(category_row["categories"], []) if category_row else []
    model_categories = json_load(model["categories"], [])
    if not dataset_categories or not model_categories:
        missing = []
        if not dataset_categories:
            missing.append(f"数据集“{dataset['name']}”")
        if not model_categories:
            missing.append(f"模型“{model['name']}”")
        return {
            "compatible": False,
            "dataset_id": dataset_id,
            "dataset_name": dataset["name"],
            "model_id": model_id,
            "model_name": model["name"],
            "reason": f"{'、'.join(missing)}尚未配置类别",
            "missing_in_model": [],
            "extra_in_model": [],
            "model_to_dataset": {},
        }
    try:
        dataset_categories = normalize_categories(dataset_categories)
        model_categories = normalize_categories(model_categories)
    except ValueError as exc:
        return {
            "compatible": False,
            "dataset_id": dataset_id,
            "dataset_name": dataset["name"],
            "model_id": model_id,
            "model_name": model["name"],
            "reason": str(exc),
            "missing_in_model": [],
            "extra_in_model": [],
            "model_to_dataset": {},
        }
    dataset_by_name = {
        normalize_category_name(item["name"]): item for item in dataset_categories
    }
    model_by_name = {
        normalize_category_name(item["name"]): item for item in model_categories
    }
    missing = sorted(set(dataset_by_name) - set(model_by_name))
    extra = sorted(set(model_by_name) - set(dataset_by_name))
    mapping = {
        str(model_category["id"]): dataset_by_name[name]["id"]
        for name, model_category in model_by_name.items()
        if name in dataset_by_name
    }
    return {
        "compatible": not missing and not extra,
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "model_id": model_id,
        "model_name": model["name"],
        "reason": "" if not missing and not extra else "类别名称集合不一致",
        "missing_in_model": [dataset_by_name[name]["name"] for name in missing],
        "extra_in_model": [model_by_name[name]["name"] for name in extra],
        "model_to_dataset": mapping,
    }


def validate_evaluation_categories(
    dataset_ids: list[str],
    model_ids: list[str],
    database: Database = db,
) -> list[dict[str, Any]]:
    results = [
        category_compatibility(dataset_id, model_id, database)
        for dataset_id in dataset_ids
        for model_id in model_ids
    ]
    incompatible = [item for item in results if not item["compatible"]]
    if incompatible:
        details = []
        for item in incompatible:
            differences = []
            if item["missing_in_model"]:
                differences.append(
                    f"模型缺少 {', '.join(item['missing_in_model'])}"
                )
            if item["extra_in_model"]:
                differences.append(
                    f"模型多出 {', '.join(item['extra_in_model'])}"
                )
            details.append(
                f"{item['dataset_name']} × {item['model_name']}: "
                f"{'；'.join(differences) or item['reason']}"
            )
        raise CategoryCompatibilityError(
            "类别不一致，无法启动评测。" + " | ".join(details)
        )
    return results


LOCAL_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".jit",
    ".onnx",
    ".plan",
    ".pt",
    ".pth",
    ".safetensors",
    ".torchscript",
}


def list_local_model_resources(
    path: str | None,
    scope: str,
    kind: str,
    app_settings=None,
) -> dict[str, Any]:
    app_settings = app_settings or settings
    roots = {
        "model": app_settings.model_library_root,
        "environment": app_settings.model_environment_root,
    }
    root_value = roots.get(scope)
    if root_value is None:
        raise LocalModelRegistrationError(f"不支持的资源范围: {scope}")
    if kind not in {"directory", "entrypoint", "weight"}:
        raise LocalModelRegistrationError(f"不支持的资源类型: {kind}")
    root = root_value.expanduser().resolve()
    if not root.is_dir():
        raise LocalModelRegistrationError(f"资源根目录不存在: {root}")
    current = Path(path).expanduser().resolve() if path else root
    if not current.is_relative_to(root) or not current.is_dir():
        raise LocalModelRegistrationError(f"目录超出允许范围: {current}")

    entries = []
    for child in sorted(
        current.iterdir(),
        key=lambda item: (not item.is_dir(), item.name.casefold()),
    ):
        try:
            resolved = child.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(root):
            continue
        if child.is_dir():
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_directory": True,
                }
            )
            continue
        if kind == "entrypoint" and child.suffix.lower() != ".py":
            continue
        if kind == "weight" and child.suffix.lower() not in LOCAL_WEIGHT_SUFFIXES:
            continue
        if kind != "directory" and child.is_file():
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_directory": False,
                }
            )
    return {
        "scope": scope,
        "kind": kind,
        "root": str(root),
        "current": str(current),
        "parent": str(current.parent) if current != root else None,
        "entries": entries,
    }


def register_local_detector_model(
    values: dict[str, Any],
    database: Database = db,
) -> dict[str, Any]:
    app_settings = database.settings
    model_root = app_settings.model_library_root.expanduser().resolve()
    environment_root = (
        app_settings.model_environment_root.expanduser().resolve()
    )
    project_directory = Path(values["project_directory"]).expanduser().resolve()
    if (
        not project_directory.is_relative_to(model_root)
        or not project_directory.is_dir()
    ):
        raise LocalModelRegistrationError(
            f"模型目录必须位于 {model_root} 内"
        )

    def project_path(value: str) -> Path:
        path = Path(value).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (project_directory / path).resolve()
        )

    working_directory = project_path(values["working_directory"])
    if (
        not working_directory.is_relative_to(project_directory)
        or not working_directory.is_dir()
    ):
        raise LocalModelRegistrationError(
            "命令工作目录必须位于模型项目目录内"
        )

    weight_path = project_path(values["weight_path"])
    if (
        not weight_path.is_relative_to(model_root)
        or not weight_path.is_file()
        or weight_path.suffix.lower() not in LOCAL_WEIGHT_SUFFIXES
    ):
        raise LocalModelRegistrationError(
            "权重必须是模型库根目录内受支持的模型文件"
        )

    runtime_prefix = project_path(values["runtime_prefix"])
    if not (
        runtime_prefix.is_relative_to(project_directory)
        or runtime_prefix.is_relative_to(environment_root)
    ):
        raise LocalModelRegistrationError(
            "Python 环境必须位于模型目录或允许的环境根目录内"
        )
    runtime_python = runtime_prefix / "bin" / "python"
    if not runtime_python.is_file():
        raise LocalModelRegistrationError(
            f"Python 解释器不存在: {runtime_python}"
        )

    executable = runtime_python
    if not os.access(executable, os.X_OK):
        raise LocalModelRegistrationError(f"Python 解释器没有执行权限: {executable}")
    try:
        validate_command_arguments(values["command_arguments"])
    except CommandTemplateError as exc:
        raise LocalModelRegistrationError(str(exc)) from exc
    try:
        categories = normalize_categories(values["categories"])
    except ValueError as exc:
        raise LocalModelRegistrationError(str(exc)) from exc

    adapter_id = new_id("adapter")
    model_id = new_id("model")
    now = utc_now()
    parameter_schema = {
        "type": "object",
        "properties": {
            "catalog_model_id": {
                "type": "string",
                "const": model_id,
            },
            "project_directory": {
                "type": "string",
                "const": str(project_directory),
            },
        },
        "execution": {
            "mode": "command",
            "working_directory": str(working_directory),
            "executable": str(executable),
            "arguments": values["command_arguments"],
            "predictions_filename": "predictions.json",
        },
    }
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO adapters
            (id,name,kind,version,maturity,runtime_kind,runtime_prefix,policy,
             entrypoint,requires_gpu,status,description,parameter_schema,
             created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                adapter_id,
                f"{values['name']} 推理适配器",
                "DETECTOR",
                values["version"],
                "REGISTERED",
                "conda_external",
                str(runtime_prefix),
                "read_only",
                str(executable),
                1,
                "REGISTERED",
                "本地模型库提供的目标检测评测协议入口。",
                json_dump(parameter_schema),
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO models
            (id,name,family,architecture,backbone,detector_head,class_count,categories,
             category_template,
             training_dataset,pretrained_dataset,version,precision,adapter_id,
             weight_path,weight_sha256,is_demo,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                model_id,
                values["name"],
                values["family"],
                values["architecture"],
                values["backbone"],
                values["detector_head"],
                len(categories),
                json_dump(categories),
                values["category_template"],
                values["training_dataset"],
                values["pretrained_dataset"],
                values["version"],
                values["precision"],
                adapter_id,
                str(weight_path),
                cached_file_sha256(str(weight_path)),
                0,
                "REGISTERED",
                now,
            ),
        )
    return next(
        item for item in list_models(database) if item["id"] == model_id
    )


def get_basegen_scene_schema() -> dict[str, Any]:
    catalog_directory = settings.basegen_root / "configs" / "field_options"
    filenames = (
        "autonomous_driving.json",
        "low_altitude_uav.json",
        "offroad_autonomous_driving.json",
    )
    domains = []
    for filename in filenames:
        path = catalog_directory / filename
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BaseGenCatalogError(f"无法读取 BaseGen 场景目录 {path}: {exc}") from exc
        fields = []
        for name, field in catalog.get("fields", {}).items():
            options = field.get("options", {})
            fields.append(
                {
                    "name": name,
                    "label_zh": BASEGEN_FIELD_LABELS.get(name, name),
                    "description_zh": field.get("description_zh", ""),
                    "kind": "text" if name == "custom" else "multi" if name == "elements" else "single",
                    "weighted": any("weight" in option for option in options.values()),
                    "options": [
                        {
                            "value": value,
                            "label_zh": option.get("label_zh", value),
                            **(
                                {"environments": option["environments"]}
                                if "environments" in option
                                else {}
                            ),
                        }
                        for value, option in options.items()
                    ],
                }
            )
        domains.append(
            {
                "value": catalog["domain"],
                "label_zh": catalog.get("label_zh", catalog["domain"]),
                "default_resolution": (
                    "1024×1024"
                    if catalog["domain"] == "low-altitude-uav"
                    else "1024×576"
                ),
                "fields": fields,
            }
        )
    return {"version": "1.0", "domains": domains}


def preview_basegen_plan(payload: dict[str, Any]) -> dict[str, Any]:
    from adapters.basegen_generator import prepare_plan

    request = {
        "protocol_version": "1.0",
        "job_id": "preview",
        "adapter_id": "adapter_basegen",
        "seed": payload.get("seeds", [1001])[0],
        "seeds": payload.get("seeds", [1001]),
        "sample_count": min(int(payload.get("sample_count", 1)), 3),
        "conditions": payload.get("conditions", {}),
        "model_parameters": payload.get("model_parameters", {}),
        "source_type": "GENERATIVE",
        "output_directory": str(settings.task_dir / "preview"),
    }
    plan, config = prepare_plan(request, settings.basegen_root)
    return {
        "model_path": config["model_path"],
        "device_policy": config["device_policy"],
        "images": [
            {
                "seed": item["seed"],
                "scene": item["scene"],
                "template_id": item["template_id"],
                "prompt": item["prompt"],
                "width": item["width"],
                "height": item["height"],
            }
            for item in plan
        ],
    }


def list_dataset_samples(
    dataset_id: str,
    offset: int,
    limit: int,
    database: Database = db,
) -> dict[str, Any] | None:
    dataset = database.row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        return None
    artifact_path = dataset.get("artifact_path")
    files: list[Path] = []
    if artifact_path:
        artifact_root = database.settings.artifact_dir.resolve()
        directory = (artifact_root / artifact_path).resolve()
        if directory == artifact_root or not directory.is_relative_to(artifact_root):
            raise DatasetArtifactError("数据集 Artifact 路径超出受控目录")
        if directory.is_dir():
            files = [
                path
                for path in sorted(directory.iterdir())
                if path.is_file()
                and path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"}
            ]
    page = files[offset : offset + limit]
    visualizations = _sample_visualizations(
        dataset_id,
        directory if artifact_path else None,
        page,
        database,
    )
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "declared_count": dataset["sample_count"],
        "total": len(files),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(files),
        "items": [
            {
                "name": path.name,
                "url": f"/artifacts/{quote(path.resolve().relative_to(database.settings.artifact_dir.resolve()).as_posix(), safe='/')}",
                **visualizations.get(
                    path.name,
                    {
                        "width": 0,
                        "height": 0,
                        "boxes": [],
                        "annotation_source": None,
                    },
                ),
            }
            for path in page
        ],
    }


def _sample_visualizations(
    dataset_id: str,
    directory: Path | None,
    files: list[Path],
    database: Database,
) -> dict[str, dict[str, Any]]:
    if not files:
        return {}
    colors = [
        "#1677FF",
        "#13A8A8",
        "#722ED1",
        "#EB2F96",
        "#52C41A",
        "#FA8C16",
        "#F5222D",
        "#2F54EB",
    ]
    dimensions: dict[str, tuple[int, int]] = {}
    output: dict[str, dict[str, Any]] = {}
    names = {path.name for path in files}
    for path in files:
        try:
            with Image.open(path) as image:
                dimensions[path.name] = (image.width, image.height)
        except OSError:
            dimensions[path.name] = (0, 0)

    categories = {
        int(category["id"]): {
            "name": str(category["name"]),
            "color": str(
                category.get(
                    "color",
                    colors[(int(category["id"]) - 1) % len(colors)],
                )
            ),
        }
        for category in _annotation_categories(dataset_id, database)
    }
    manual_rows = database.rows(
        """
        SELECT sample_name,width,height,boxes FROM sample_annotations
        WHERE dataset_id=?
        """,
        (dataset_id,),
    )
    for row in manual_rows:
        if row["sample_name"] not in names:
            continue
        width = int(row["width"])
        height = int(row["height"])
        boxes = []
        if width > 0 and height > 0:
            for box in json_load(row["boxes"], []):
                category_id = int(box["category_id"])
                category = categories.get(
                    category_id,
                    {
                        "name": f"class {category_id}",
                        "color": colors[category_id % len(colors)],
                    },
                )
                boxes.append(
                    {
                        "label": category["name"],
                        "color": category["color"],
                        "x": float(box["x"]) / width,
                        "y": float(box["y"]) / height,
                        "width": float(box["width"]) / width,
                        "height": float(box["height"]) / height,
                    }
                )
        dimensions[row["sample_name"]] = (width, height)
        output[row["sample_name"]] = {
            "width": width,
            "height": height,
            "boxes": boxes,
            "annotation_source": "MANUAL",
        }

    coco_path = directory / "annotations" / "instances.json" if directory else None
    if coco_path and coco_path.is_file():
        try:
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            coco = {}
        coco_categories = {
            int(category["id"]): {
                "name": str(category["name"]),
                "color": colors[
                    (int(category["id"]) - 1) % len(colors)
                ],
            }
            for category in coco.get("categories", [])
        }
        images_by_id = {
            int(image["id"]): image
            for image in coco.get("images", [])
            if Path(str(image.get("file_name", ""))).name in names
        }
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image_id, image in images_by_id.items():
            name = Path(str(image["file_name"])).name
            if name in output:
                continue
            width = int(image.get("width") or dimensions.get(name, (0, 0))[0])
            height = int(image.get("height") or dimensions.get(name, (0, 0))[1])
            boxes = []
            if width > 0 and height > 0:
                for annotation in annotations_by_image.get(image_id, []):
                    bbox = annotation.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    category_id = int(annotation["category_id"])
                    category = coco_categories.get(
                        category_id,
                        {
                            "name": f"class {category_id}",
                            "color": colors[category_id % len(colors)],
                        },
                    )
                    x, y, box_width, box_height = (
                        float(value) for value in bbox
                    )
                    boxes.append(
                        {
                            "label": category["name"],
                            "color": category["color"],
                            "x": x / width,
                            "y": y / height,
                            "width": box_width / width,
                            "height": box_height / height,
                        }
                    )
            dimensions[name] = (width, height)
            output[name] = {
                "width": width,
                "height": height,
                "boxes": boxes,
                "annotation_source": "COCO",
            }

    label_roots = (
        [
            ("VISDRONE", directory / "annotations" / "visdrone"),
            ("YOLO", directory / "annotations" / "yolo"),
        ]
        if directory
        else []
    )
    label_files: dict[str, tuple[Path, str]] = {}
    for source, root in label_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.txt")):
            label_files.setdefault(path.stem, (path, source))
    if label_files:
        visdrone_categories = {
            0: "ignored region",
            1: "pedestrian",
            2: "people",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            7: "tricycle",
            8: "awning-tricycle",
            9: "bus",
            10: "motor",
            11: "others",
        }
        for path in files:
            if path.name in output:
                continue
            label_entry = label_files.get(path.stem)
            if not label_entry:
                continue
            label_path, configured_source = label_entry
            lines = [
                line.strip()
                for line in label_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            source = configured_source
            if lines:
                fields = lines[0].split(",")
                if len(fields) >= 8:
                    source = "VISDRONE"
            boxes = []
            width, height = dimensions.get(path.name, (0, 0))
            for line in lines:
                if source == "VISDRONE":
                    fields = line.split(",")
                    if len(fields) < 8 or width <= 0 or height <= 0:
                        continue
                    try:
                        x, y, box_width, box_height = (
                            float(value) for value in fields[:4]
                        )
                        category_id = int(float(fields[5]))
                    except ValueError:
                        continue
                    boxes.append(
                        {
                            "label": visdrone_categories.get(
                                category_id, f"class {category_id}"
                            ),
                            "color": (
                                "#8C8C8C"
                                if category_id == 0
                                else colors[
                                    (category_id - 1) % len(colors)
                                ]
                            ),
                            "x": x / width,
                            "y": y / height,
                            "width": box_width / width,
                            "height": box_height / height,
                        }
                    )
                    continue
                fields = line.split()
                if len(fields) < 5:
                    continue
                try:
                    category_id = int(float(fields[0]))
                    center_x, center_y, box_width, box_height = (
                        float(value) for value in fields[1:5]
                    )
                except ValueError:
                    continue
                category = categories.get(
                    category_id,
                    {
                        "name": f"class {category_id}",
                        "color": colors[category_id % len(colors)],
                    },
                )
                boxes.append(
                    {
                        "label": category["name"],
                        "color": category["color"],
                        "x": center_x - box_width / 2,
                        "y": center_y - box_height / 2,
                        "width": box_width,
                        "height": box_height,
                    }
                )
            output[path.name] = {
                "width": width,
                "height": height,
                "boxes": boxes,
                "annotation_source": source,
            }

    for path in files:
        if path.name not in output:
            width, height = dimensions.get(path.name, (0, 0))
            output[path.name] = {
                "width": width,
                "height": height,
                "boxes": [],
                "annotation_source": None,
            }
    return output


def _dataset_images(
    dataset_id: str,
    database: Database,
) -> tuple[dict[str, Any], Path | None, list[Path]] | None:
    dataset = database.row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        return None
    artifact_path = dataset.get("artifact_path")
    if not artifact_path:
        return dataset, None, []
    artifact_root = database.settings.artifact_dir.resolve()
    directory = (artifact_root / artifact_path).resolve()
    if directory == artifact_root or not directory.is_relative_to(artifact_root):
        raise DatasetArtifactError("数据集 Artifact 路径超出受控目录")
    files = []
    if directory.is_dir():
        files = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file()
            and path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"}
        ]
    return dataset, directory, files


def dataset_statistics(
    dataset_id: str,
    database: Database = db,
) -> dict[str, Any] | None:
    resolved = _dataset_images(dataset_id, database)
    if not resolved:
        return None
    dataset, directory, files = resolved
    stored_categories = _annotation_categories(dataset_id, database)
    categories = normalize_categories(stored_categories) if stored_categories else []
    counts = {
        normalize_category_name(category["name"]): {
            "id": category["id"],
            "name": category["name"],
            "count": 0,
        }
        for category in categories
    }
    visualizations = _sample_visualizations(dataset_id, directory, files, database)
    resolution_counts: dict[tuple[int, int], int] = defaultdict(int)
    scale_counts = {"small": 0, "medium": 0, "large": 0, "unknown": 0}
    annotated_images = 0
    object_count = 0
    for item in visualizations.values():
        width = int(item["width"])
        height = int(item["height"])
        if width > 0 and height > 0:
            resolution_counts[(width, height)] += 1
        if item["annotation_source"]:
            annotated_images += 1
        for box in item["boxes"]:
            object_count += 1
            normalized_name = normalize_category_name(box["label"])
            if normalized_name not in counts:
                counts[normalized_name] = {
                    "id": None,
                    "name": box["label"],
                    "count": 0,
                }
            counts[normalized_name]["count"] += 1
            if width <= 0 or height <= 0:
                scale_counts["unknown"] += 1
                continue
            area = max(0.0, float(box["width"]) * width) * max(0.0, float(box["height"]) * height)
            if area < 32 ** 2:
                scale_counts["small"] += 1
            elif area < 96 ** 2:
                scale_counts["medium"] += 1
            else:
                scale_counts["large"] += 1
    category_counts = sorted(
        counts.values(),
        key=lambda item: (
            item["id"] is None,
            item["id"] if item["id"] is not None else item["name"],
        ),
    )
    resolutions = [
        {"width": width, "height": height, "label": f"{width}×{height}", "count": count}
        for (width, height), count in sorted(
            resolution_counts.items(),
            key=lambda item: (-item[1], -(item[0][0] * item[0][1]), item[0]),
        )
    ]
    return {
        "dataset_id": dataset_id,
        "image_count": len(files),
        "annotated_image_count": annotated_images,
        "object_count": object_count,
        "category_counts": category_counts,
        "resolutions": resolutions,
        "scales": [
            {"key": "small", "label": "小目标 < 32²", "count": scale_counts["small"]},
            {"key": "medium", "label": "中目标 32²–96²", "count": scale_counts["medium"]},
            {"key": "large", "label": "大目标 ≥ 96²", "count": scale_counts["large"]},
            {"key": "unknown", "label": "尺寸未知", "count": scale_counts["unknown"]},
        ],
    }


def _annotation_categories(
    dataset_id: str,
    database: Database,
) -> list[dict[str, Any]]:
    row = database.row(
        "SELECT categories FROM dataset_annotation_schemas WHERE dataset_id=?",
        (dataset_id,),
    )
    return json_load(row["categories"]) if row else []


def get_annotation_session(
    dataset_id: str,
    database: Database = db,
) -> dict[str, Any] | None:
    resolved = _dataset_images(dataset_id, database)
    if not resolved:
        return None
    dataset, _, files = resolved
    rows = {
        row["sample_name"]: row
        for row in database.rows(
            """
            SELECT sample_name,completed,boxes,updated_at
            FROM sample_annotations WHERE dataset_id=?
            """,
            (dataset_id,),
        )
    }
    completed = sum(
        bool(rows[path.name]["completed"])
        for path in files
        if path.name in rows
    )
    artifact_root = database.settings.artifact_dir.resolve()
    return {
        "dataset": decode_row(dataset),
        "categories": _annotation_categories(dataset_id, database),
        "progress": {"completed": completed, "total": len(files)},
        "samples": [
            {
                "name": path.name,
                "url": (
                    f"/artifacts/{quote(path.resolve().relative_to(artifact_root).as_posix(), safe='/')}"
                ),
                "completed": bool(rows.get(path.name, {}).get("completed")),
                "box_count": len(json_load(rows.get(path.name, {}).get("boxes"), [])),
            }
            for path in files
        ],
    }


def get_sample_annotation(
    dataset_id: str,
    sample_name: str,
    database: Database = db,
) -> dict[str, Any] | None:
    resolved = _dataset_images(dataset_id, database)
    if not resolved:
        return None
    _, _, files = resolved
    if sample_name not in {path.name for path in files}:
        return None
    row = database.row(
        """
        SELECT width,height,boxes,completed,updated_at
        FROM sample_annotations WHERE dataset_id=? AND sample_name=?
        """,
        (dataset_id, sample_name),
    )
    return {
        "dataset_id": dataset_id,
        "sample_name": sample_name,
        "width": row["width"] if row else 0,
        "height": row["height"] if row else 0,
        "boxes": json_load(row["boxes"], []) if row else [],
        "completed": bool(row["completed"]) if row else False,
        "updated_at": row["updated_at"] if row else None,
    }


def update_annotation_schema(
    dataset_id: str,
    categories: list[dict[str, Any]],
    database: Database = db,
) -> dict[str, Any] | None:
    dataset = database.row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        return None
    if dataset["frozen"]:
        raise DatasetAnnotationError("冻结数据集的标注类别不可修改")
    try:
        normalized = normalize_categories(categories, include_color=True)
    except ValueError as exc:
        raise DatasetAnnotationError(str(exc)) from exc
    used_ids = {
        int(box["category_id"])
        for row in database.rows(
            "SELECT boxes FROM sample_annotations WHERE dataset_id=?",
            (dataset_id,),
        )
        for box in json_load(row["boxes"], [])
    }
    missing = used_ids - {category["id"] for category in normalized}
    if missing:
        raise DatasetAnnotationError("不能删除已被目标框使用的类别")
    database.execute(
        """
        INSERT INTO dataset_annotation_schemas(dataset_id,categories,updated_at)
        VALUES (?,?,?)
        ON CONFLICT(dataset_id) DO UPDATE
        SET categories=excluded.categories,updated_at=excluded.updated_at
        """,
        (dataset_id, json_dump(normalized), utc_now()),
    )
    database.execute(
        "UPDATE datasets SET category_template='custom' WHERE id=?",
        (dataset_id,),
    )
    return {"dataset_id": dataset_id, "categories": normalized}


def save_sample_annotation(
    dataset_id: str,
    sample_name: str,
    payload: dict[str, Any],
    database: Database = db,
) -> dict[str, Any] | None:
    resolved = _dataset_images(dataset_id, database)
    if not resolved:
        return None
    dataset, _, files = resolved
    if dataset["frozen"]:
        raise DatasetAnnotationError("冻结数据集的标注不可修改")
    if sample_name not in {path.name for path in files}:
        return None
    width = int(payload["width"])
    height = int(payload["height"])
    boxes = payload.get("boxes", [])
    categories = {
        category["id"] for category in _annotation_categories(dataset_id, database)
    }
    box_ids = [box["id"] for box in boxes]
    if len(box_ids) != len(set(box_ids)):
        raise DatasetAnnotationError("同一图片中的目标框 ID 必须唯一")
    for box in boxes:
        if box["category_id"] not in categories:
            raise DatasetAnnotationError(f"目标框引用了不存在的类别 {box['category_id']}")
        if box["x"] + box["width"] > width or box["y"] + box["height"] > height:
            raise DatasetAnnotationError("目标框超出图像边界")
    now = utc_now()
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO sample_annotations
            (dataset_id,sample_name,width,height,boxes,completed,updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(dataset_id,sample_name) DO UPDATE SET
              width=excluded.width,height=excluded.height,boxes=excluded.boxes,
              completed=excluded.completed,updated_at=excluded.updated_at
            """,
            (
                dataset_id,
                sample_name,
                width,
                height,
                json_dump(boxes),
                int(bool(payload.get("completed"))),
                now,
            ),
        )
        connection.execute(
            """
            UPDATE datasets SET annotation_status='ANNOTATING'
            WHERE id=? AND annotation_status!='ANNOTATING'
            """,
            (dataset_id,),
        )
    return get_sample_annotation(dataset_id, sample_name, database)


def complete_dataset_annotations(
    dataset_id: str,
    database: Database = db,
) -> dict[str, Any] | None:
    resolved = _dataset_images(dataset_id, database)
    if not resolved:
        return None
    dataset, directory, files = resolved
    if dataset["frozen"]:
        raise DatasetAnnotationError("冻结数据集不能重新提交标注")
    if not directory or not files:
        raise DatasetAnnotationError("数据集没有可标注的图片")
    rows = {
        row["sample_name"]: row
        for row in database.rows(
            """
            SELECT sample_name,width,height,boxes,completed
            FROM sample_annotations WHERE dataset_id=?
            """,
            (dataset_id,),
        )
    }
    incomplete = [
        path.name
        for path in files
        if path.name not in rows or not rows[path.name]["completed"]
    ]
    if incomplete:
        raise DatasetAnnotationError(
            f"还有 {len(incomplete)} 张图片未确认完成"
        )
    categories = _annotation_categories(dataset_id, database)
    if not categories:
        raise DatasetAnnotationError("数据集尚未配置类别")
    images = []
    annotations = []
    annotation_id = 1
    for image_id, path in enumerate(files, start=1):
        row = rows[path.name]
        images.append(
            {
                "id": image_id,
                "file_name": path.name,
                "width": row["width"],
                "height": row["height"],
            }
        )
        for box in json_load(row["boxes"], []):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box["category_id"],
                    "bbox": [box["x"], box["y"], box["width"], box["height"]],
                    "area": box["width"] * box["height"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = {
        "info": {
            "description": dataset["name"],
            "version": dataset["version"],
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category["id"], "name": category["name"]}
            for category in categories
        ],
    }
    annotation_directory = directory / "annotations"
    annotation_directory.mkdir(parents=True, exist_ok=True)
    output = annotation_directory / "instances.json"
    temporary = annotation_directory / "instances.json.tmp"
    temporary.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    database.execute(
        "UPDATE datasets SET annotation_status='CANDIDATE' WHERE id=?",
        (dataset_id,),
    )
    return {
        "dataset_id": dataset_id,
        "annotation_status": "CANDIDATE",
        "images": len(images),
        "annotations": len(annotations),
        "path": output.relative_to(database.settings.artifact_dir).as_posix(),
    }


def delete_dataset(dataset_id: str, database: Database = db) -> dict[str, Any] | None:
    artifact_root = database.settings.artifact_dir.resolve()
    trash_directory = (
        database.settings.data_dir / "trash" / "datasets" / dataset_id
    ).resolve()
    source: Path | None = None
    trash_artifact = trash_directory / "artifact"
    manifest = trash_directory / "dataset.json"
    moved = False
    trash_created = False
    dataset: dict[str, Any] | None = None
    try:
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM datasets WHERE id=?", (dataset_id,)
            ).fetchone()
            if not row:
                return None
            dataset = dict(row)
            if dataset["frozen"]:
                raise DatasetDeletionError("冻结数据集受保护，不能删除")
            run_count = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE dataset_id=?", (dataset_id,)
            ).fetchone()[0]
            if run_count:
                raise DatasetDeletionError(
                    f"数据集已被 {run_count} 个评测运行引用，不能删除"
                )
            referenced_plans = [
                plan["id"]
                for plan in connection.execute(
                    "SELECT id,dataset_ids FROM evaluation_plans"
                ).fetchall()
                if dataset_id in json_load(plan["dataset_ids"], [])
            ]
            if referenced_plans:
                raise DatasetDeletionError(
                    f"数据集已被评测方案 {referenced_plans[0]} 引用，不能删除"
                )

            artifact_path = dataset.get("artifact_path")
            if artifact_path:
                shared_count = connection.execute(
                    "SELECT COUNT(*) FROM datasets WHERE artifact_path=?",
                    (artifact_path,),
                ).fetchone()[0]
                if shared_count > 1:
                    raise DatasetDeletionError("Artifact 目录被多个数据集共享，不能删除")
                source = (artifact_root / artifact_path).resolve()
                if source == artifact_root or not source.is_relative_to(artifact_root):
                    raise DatasetDeletionError("数据集 Artifact 路径超出受控目录")

            if trash_directory.exists():
                raise DatasetDeletionError(f"回收站目标已存在: {trash_directory}")
            trash_directory.mkdir(parents=True)
            trash_created = True
            manifest_payload = {
                **dataset,
                "annotation_schema": connection.execute(
                    """
                    SELECT categories,updated_at FROM dataset_annotation_schemas
                    WHERE dataset_id=?
                    """,
                    (dataset_id,),
                ).fetchone(),
                "sample_annotations": [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT sample_name,width,height,boxes,completed,updated_at
                        FROM sample_annotations WHERE dataset_id=?
                        """,
                        (dataset_id,),
                    ).fetchall()
                ],
            }
            if manifest_payload["annotation_schema"]:
                manifest_payload["annotation_schema"] = dict(
                    manifest_payload["annotation_schema"]
                )
            manifest.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if source and source.exists():
                shutil.move(str(source), str(trash_artifact))
                moved = True
            connection.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    except BaseException:
        if moved and source and trash_artifact.exists() and not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trash_artifact), str(source))
        if trash_created and manifest.exists():
            manifest.unlink()
        if trash_created and trash_directory.exists():
            try:
                trash_directory.rmdir()
            except OSError:
                pass
        raise
    assert dataset is not None
    return {
        "id": dataset_id,
        "name": dataset["name"],
        "deleted": True,
        "artifact_moved": moved,
        "trash_path": str(trash_directory),
    }


def list_models(database: Database = db) -> list[dict[str, Any]]:
    return [decode_row(row) for row in database.rows("SELECT * FROM models ORDER BY created_at DESC")]


def delete_model(
    model_id: str,
    database: Database = db,
) -> dict[str, Any] | None:
    with database.connect() as connection:
        model = connection.execute(
            "SELECT * FROM models WHERE id=?",
            (model_id,),
        ).fetchone()
        if not model:
            return None
        run_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM runs WHERE model_id=?",
                (model_id,),
            ).fetchone()[0]
        )
        plan_references = []
        for plan in connection.execute(
            "SELECT id,name,model_ids FROM evaluation_plans"
        ).fetchall():
            if model_id in json_load(plan["model_ids"], []):
                plan_references.append(plan["id"])
        if run_count or plan_references:
            details = []
            if run_count:
                details.append(f"{run_count} 次历史运行")
            if plan_references:
                details.append(f"{len(plan_references)} 个评测方案")
            raise ModelDeletionError(
                f"模型仍被{'、'.join(details)}引用，不能删除"
            )

        adapter_id = model["adapter_id"]
        connection.execute("DELETE FROM models WHERE id=?", (model_id,))
        adapter = connection.execute(
            "SELECT description FROM adapters WHERE id=?",
            (adapter_id,),
        ).fetchone()
        adapter_deleted = False
        if (
            adapter
            and adapter["description"]
            == "本地模型库提供的目标检测评测协议入口。"
            and not connection.execute(
                "SELECT 1 FROM models WHERE adapter_id=? LIMIT 1",
                (adapter_id,),
            ).fetchone()
        ):
            connection.execute(
                "DELETE FROM adapters WHERE id=?",
                (adapter_id,),
            )
            adapter_deleted = True
    return {
        "id": model_id,
        "name": model["name"],
        "deleted": True,
        "adapter_deleted": adapter_deleted,
        "files_deleted": False,
    }


def list_jobs(database: Database = db, limit: int = 100) -> list[dict[str, Any]]:
    return [
        decode_row(row)
        for row in database.rows(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
        )
    ]


def get_job(job_id: str, database: Database = db) -> dict[str, Any] | None:
    row = database.row("SELECT * FROM jobs WHERE id=?", (job_id,))
    return decode_row(row) if row else None


def delete_job(job_id: str, database: Database = db) -> dict[str, Any] | None:
    task_root = database.settings.task_dir.resolve()
    log_root = database.settings.log_dir.resolve()
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        job = decode_row(dict(row))
        if job["status"] not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            raise JobDeletionError("任务尚未结束，请先取消任务并等待其进入终态")

        paths: list[Path] = []
        workspace = (task_root / job_id).resolve()
        if workspace != task_root and workspace.is_relative_to(task_root):
            paths.append(workspace)
        payload = job.get("payload") or {}
        staged_value = payload.get("staged_upload_root") if isinstance(payload, dict) else None
        if staged_value:
            staged = Path(str(staged_value)).expanduser().resolve()
            staging_root = (task_root / "import_uploads").resolve()
            if staged == staging_root or not staged.is_relative_to(staging_root):
                raise JobDeletionError("任务暂存目录超出受控工作区，已拒绝删除")
            paths.append(staged)
        log_path = (log_root / f"{job_id}.log").resolve()
        if log_path != log_root and log_path.is_relative_to(log_root):
            paths.append(log_path)

        connection.execute("UPDATE runs SET job_id=NULL WHERE job_id=?", (job_id,))
        connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    cleanup_errors: list[str] = []
    deleted_paths: list[str] = []
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted_paths.append(str(path))
        except OSError as exc:
            cleanup_errors.append(f"{path}: {exc}")
    return {
        "id": job_id,
        "deleted": True,
        "deleted_paths": deleted_paths,
        "cleanup_errors": cleanup_errors,
        "results_preserved": True,
    }


def queue_job(job_type: str, payload: dict[str, Any], database: Database = db) -> dict[str, Any]:
    job_id = new_id("job")
    now = utc_now()
    database.execute(
        """
        INSERT INTO jobs(id,type,status,progress,stage,payload,created_at)
        VALUES (?,?, 'QUEUED', 0, '等待执行', ?, ?)
        """,
        (job_id, job_type, json_dump(payload), now),
    )
    return get_job(job_id, database) or {"id": job_id, "status": "QUEUED"}


def overview(database: Database = db) -> dict[str, Any]:
    with database.connect() as connection:
        counts = {
            "datasets": connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "models": connection.execute("SELECT COUNT(*) FROM models").fetchone()[0],
            "running": connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING','VALIDATING')"
            ).fetchone()[0],
            "completed": connection.execute("SELECT COUNT(*) FROM runs WHERE status='SUCCEEDED'").fetchone()[0],
        }
        best = connection.execute(
            """
            SELECT r.map,r.latency_p50,m.name AS model_name,d.name AS dataset_name
            FROM results r JOIN runs x ON x.id=r.run_id
            JOIN models m ON m.id=x.model_id JOIN datasets d ON d.id=x.dataset_id
            ORDER BY r.map DESC LIMIT 1
            """
        ).fetchone()
    datasets = list(list_datasets(database))
    return {
        "counts": counts,
        "best_result": dict(best) if best else None,
        "recent_jobs": list_jobs(database, 6),
        "recent_images": [image for dataset in datasets for image in dataset["preview_images"][:2]][:8],
        "pipeline": [
            {"name": "数据构建", "status": "complete"},
            {"name": "真值校核", "status": "complete"},
            {"name": "批量评测", "status": "active"},
            {"name": "效能展示", "status": "ready"},
        ],
    }


def query_results(
    scene: str | None = None,
    condition: str | None = None,
    resolution: str | None = None,
    model_id: str | None = None,
    database: Database = db,
) -> dict[str, Any]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if scene:
        clauses.append("d.scene_domain = ?")
        parameters.append(scene)
    if condition:
        clauses.append("(d.weather = ? OR d.sensor_conditions LIKE ?)")
        parameters.extend([condition, f"%{condition}%"])
    if resolution:
        clauses.append("d.resolution = ?")
        parameters.append(resolution)
    if model_id:
        clauses.append("m.id = ?")
        parameters.append(model_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = database.rows(
        f"""
        SELECT r.*,x.seed,x.config,x.hardware_profile,x.environment_fingerprint,
               d.id AS dataset_id,d.name AS dataset_name,d.scene_domain,d.weather,d.resolution,
               d.source_type,d.sensor_conditions,d.annotation_status,
               m.id AS model_id,m.name AS model_name,m.family,m.backbone,m.precision AS model_precision,
               m.is_demo
        FROM results r JOIN runs x ON x.id=r.run_id
        JOIN datasets d ON d.id=x.dataset_id JOIN models m ON m.id=x.model_id
        {where}
        ORDER BY r.map DESC
        """,
        tuple(parameters),
    )
    decoded = [decode_row(row) for row in rows]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decoded:
        groups[(row["dataset_id"], row["model_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for values in groups.values():
        map_values = [item["map"] for item in values]
        latency_values = [item["latency_p50"] for item in values]
        first = values[0]
        mean = sum(map_values) / len(map_values)
        variance = sum((value - mean) ** 2 for value in map_values) / len(map_values)
        summaries.append(
            {
                "dataset_id": first["dataset_id"],
                "dataset_name": first["dataset_name"],
                "model_id": first["model_id"],
                "model_name": first["model_name"],
                "family": first["family"],
                "backbone": first["backbone"],
                "scene_domain": first["scene_domain"],
                "weather": first["weather"],
                "resolution": first["resolution"],
                "sensor_conditions": first["sensor_conditions"],
                "source_type": first["source_type"],
                "is_demo": first["is_demo"],
                "is_official": all(item["is_official"] for item in values),
                "map_mean": round(mean, 4),
                "map_std": round(math.sqrt(variance), 4),
                "latency_mean": round(sum(latency_values) / len(latency_values), 2),
                "delta_map_mean": round(sum(item["delta_map"] or 0 for item in values) / len(values), 4),
                "seed_count": len(values),
                "curves": first["curves"],
            }
        )
    summaries.sort(key=lambda item: item["map_mean"], reverse=True)
    dimensions = {
        "scenes": sorted({row["scene_domain"] for row in decoded}),
        "conditions": sorted({row["weather"] for row in decoded}),
        "resolutions": sorted({row["resolution"] for row in decoded}),
        "models": sorted({row["model_name"] for row in decoded}),
    }
    return {"count": len(decoded), "groups": summaries, "runs": decoded, "dimensions": dimensions}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_status() -> dict[str, Any]:
    conda_executable = shutil.which("conda") or shutil.which("mamba")
    conda_data: dict[str, Any] = {"available": bool(conda_executable), "executable": conda_executable, "envs": []}
    if conda_executable:
        try:
            result = subprocess.run(
                [conda_executable, "info", "--json"], capture_output=True, text=True, timeout=12, check=True
            )
            info = json.loads(result.stdout)
            for prefix_value in info.get("envs", []):
                prefix = Path(prefix_value)
                history = prefix / "conda-meta" / "history"
                conda_data["envs"].append(
                    {
                        "name": "base" if prefix == Path(info.get("root_prefix", "")) else prefix.name,
                        "prefix": str(prefix),
                        "policy": "external_read_only",
                        "exists": prefix.exists(),
                        "fingerprint": file_sha256(history)[:16] if history.exists() else None,
                    }
                )
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            conda_data["error"] = str(exc)
    gpu: dict[str, Any] = {"available": False, "devices": []}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=True,
            )
            gpu["devices"] = [
                {"name": fields[0], "driver": fields[1], "memory_mb": int(fields[2])}
                for line in result.stdout.splitlines()
                if len(fields := [value.strip() for value in line.split(",")]) == 3
            ]
            gpu["available"] = bool(gpu["devices"])
        except (subprocess.SubprocessError, ValueError) as exc:
            gpu["error"] = str(exc)
    disk = shutil.disk_usage(settings.root_dir)
    return {
        "isolation": {
            "mode": "workspace",
            "runtime_dir": str(settings.runtime_dir),
            "data_dir": str(settings.data_dir),
            "writes_outside_workspace": False,
            "shell_configuration_modified": False,
        },
        "conda": conda_data,
        "gpu": gpu,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "process": {"pid": os.getpid(), "python": sys.executable},
    }


def adapter_health(adapter_id: str, database: Database = db) -> dict[str, Any] | None:
    adapter = database.row("SELECT * FROM adapters WHERE id=?", (adapter_id,))
    if not adapter:
        return None
    adapter = decode_row(adapter)
    execution = adapter.get("parameter_schema", {}).get("execution", {})
    command_mode = execution.get("mode") == "command"
    checks: list[dict[str, Any]] = []
    if adapter["runtime_kind"] == "platform":
        entrypoint = settings.root_dir / (adapter.get("entrypoint") or "")
        checks.append({"name": "入口脚本", "ok": entrypoint.is_file(), "detail": str(entrypoint)})
    elif adapter["runtime_kind"] in {"conda_external", "conda_clone"}:
        prefix = Path(adapter.get("runtime_prefix") or "")
        history = prefix / "conda-meta" / "history"
        entrypoint_value = adapter.get("entrypoint") or ""
        entrypoint = Path(entrypoint_value)
        if not entrypoint.is_absolute():
            entrypoint = settings.root_dir / entrypoint
        checks.extend(
            [
                {"name": "环境目录", "ok": prefix.is_dir(), "detail": str(prefix)},
                {"name": "Python解释器", "ok": (prefix / "bin" / "python").is_file(), "detail": str(prefix / "bin/python")},
                {"name": "可执行程序" if command_mode else "入口脚本", "ok": entrypoint.is_file(), "detail": str(entrypoint)},
                {"name": "只读策略", "ok": adapter["policy"] == "read_only", "detail": adapter["policy"]},
            ]
        )
        if command_mode:
            working_directory = Path(
                str(execution.get("working_directory", ""))
            )
            checks.extend(
                [
                    {
                        "name": "命令工作目录",
                        "ok": working_directory.is_dir(),
                        "detail": str(working_directory),
                    },
                    {
                        "name": "命令参数",
                        "ok": bool(execution.get("arguments")),
                        "detail": json_dump(execution.get("arguments", [])),
                    },
                ]
            )
        if adapter["id"] == "adapter_basegen":
            checks.append(
                {
                    "name": "BaseGen项目",
                    "ok": (settings.basegen_root / "zimage_gen" / "runner.py").is_file(),
                    "detail": str(settings.basegen_root),
                }
            )
            try:
                result = subprocess.run(
                    [
                        str(prefix / "bin" / "python"),
                        "-c",
                        "import torch,diffusers,transformers; print(torch.__version__,diffusers.__version__,transformers.__version__)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                checks.append(
                    {
                        "name": "生成依赖",
                        "ok": result.returncode == 0,
                        "detail": (result.stdout or result.stderr).strip()[-300:],
                    }
                )
            except (OSError, subprocess.SubprocessError) as exc:
                checks.append(
                    {"name": "生成依赖", "ok": False, "detail": str(exc)}
                )
        if adapter["id"] == "adapter_dronedets_yolov8m":
            dronedets_root = database.settings.dronedets_root
            model = database.row(
                """
                SELECT weight_path FROM models
                WHERE adapter_id='adapter_dronedets_yolov8m'
                ORDER BY created_at DESC LIMIT 1
                """
            )
            weight_path = Path(model["weight_path"]) if model and model["weight_path"] else None
            checks.append(
                {
                    "name": "DroneDets项目",
                    "ok": (
                        dronedets_root / "src" / "aerial_det" / "catalog.py"
                    ).is_file(),
                    "detail": str(dronedets_root),
                }
            )
            checks.append(
                {
                    "name": "模型权重",
                    "ok": bool(weight_path and weight_path.is_file()),
                    "detail": str(weight_path) if weight_path else "未配置",
                }
            )
            try:
                result = subprocess.run(
                    [
                        str(prefix / "bin" / "python"),
                        "-B",
                        str(entrypoint),
                        "health",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={
                        **os.environ,
                        "DRONEDETS_ROOT": str(dronedets_root),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                detail = (result.stdout or result.stderr).strip()[-500:]
                checks.append(
                    {
                        "name": "检测依赖与目录",
                        "ok": result.returncode == 0,
                        "detail": detail,
                    }
                )
            except (OSError, subprocess.SubprocessError) as exc:
                checks.append(
                    {"name": "检测依赖与目录", "ok": False, "detail": str(exc)}
                )
        if history.exists():
            checks.append({"name": "环境指纹", "ok": True, "detail": file_sha256(history)[:16]})
    else:
        checks.append({"name": "远程端点", "ok": bool(adapter.get("runtime_prefix")), "detail": adapter.get("runtime_prefix")})
    healthy = all(check["ok"] for check in checks)
    database.execute(
        "UPDATE adapters SET status=?,updated_at=? WHERE id=?",
        ("HEALTHY" if healthy else "UNAVAILABLE", utc_now(), adapter_id),
    )
    return {"adapter_id": adapter_id, "healthy": healthy, "checks": checks}
