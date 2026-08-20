from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from PIL import Image

from app.category_templates import (
    normalize_categories,
    normalize_category_name,
    template_categories,
)
from app.command_protocol import (
    CommandTemplateError,
    command_placeholders,
    validate_command_arguments,
)
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
        row["dataset_path"] = None
        row["platform_path"] = None
        row["preview_images"] = []
        if relative:
            directory = database.settings.artifact_dir / relative
            platform_path = directory.resolve()
            source_path = row.get("source_path")
            if not source_path and row.get("source_type") == "REAL" and directory.is_dir():
                inferred = _referenced_dataset_source(directory)
                source_path = str(inferred) if inferred else None
            row["source_path"] = source_path
            row["platform_path"] = str(platform_path)
            row["dataset_path"] = source_path or str(platform_path)
            if directory.exists():
                image_files = _dataset_image_files(directory)
                row["preview_images"] = [
                    f"/artifacts/{path.relative_to(database.settings.artifact_dir).as_posix()}"
                    for path in image_files
                ][:6]
                if (
                    not str(row.get("resolution") or "").strip()
                    or row["resolution"] == "原始分辨率"
                ):
                    resolution = summarize_image_resolutions(image_files)
                    if resolution:
                        database.execute(
                            "UPDATE datasets SET resolution=? WHERE id=?",
                            (resolution, row["id"]),
                        )
                        row["resolution"] = resolution
                    else:
                        row["resolution"] = "无法读取"
        if (
            not str(row.get("resolution") or "").strip()
            or row["resolution"] == "原始分辨率"
        ):
            row["resolution"] = "无法读取"
        yield row


class DatasetDeletionError(ValueError):
    pass


class JobDeletionError(ValueError):
    pass


class EvaluationResultDeletionError(ValueError):
    pass


class DatasetArtifactError(ValueError):
    pass


class BaseGenCatalogError(ValueError):
    pass


class DatasetAnnotationError(ValueError):
    pass


class DatasetImportError(ValueError):
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
    common_categories = [
        {
            "name": dataset_by_name[name]["name"],
            "dataset_id": dataset_by_name[name]["id"],
            "model_id": model_by_name[name]["id"],
        }
        for name in dataset_by_name
        if name in model_by_name
    ]
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
        "common_categories": common_categories,
    }


def validate_evaluation_categories(
    dataset_ids: list[str],
    model_ids: list[str],
    database: Database = db,
    evaluation_categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    results = [
        category_compatibility(dataset_id, model_id, database)
        for dataset_id in dataset_ids
        for model_id in model_ids
    ]
    selected_by_name: dict[str, str] | None = None
    if evaluation_categories is not None:
        selected_by_name = {}
        for value in evaluation_categories:
            display_name = str(value).strip()
            normalized_name = normalize_category_name(display_name)
            if not normalized_name:
                raise CategoryCompatibilityError("评测类别名称不能为空")
            if normalized_name in selected_by_name:
                raise CategoryCompatibilityError("评测类别名称不能重复")
            selected_by_name[normalized_name] = display_name
        if not selected_by_name:
            raise CategoryCompatibilityError("至少选择一个评测类别")
        unavailable = []
        for item in results:
            common = {
                normalize_category_name(category["name"]): category
                for category in item.get("common_categories", [])
            }
            missing = [
                selected_by_name[name]
                for name in selected_by_name
                if name not in common
            ]
            if missing:
                unavailable.append(
                    f"{item['dataset_name']} × {item['model_name']}: "
                    f"缺少 {', '.join(missing)}"
                )
            else:
                item["evaluation_categories"] = [
                    common[name]["name"] for name in selected_by_name
                ]
                item["evaluation_category_ids"] = [
                    int(common[name]["dataset_id"])
                    for name in selected_by_name
                ]
        if unavailable:
            raise CategoryCompatibilityError(
                "所选评测类别不是数据集与模型的共同类别。" + " | ".join(unavailable)
            )
        return results

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
    for item in results:
        item["evaluation_categories"] = [
            category["name"] for category in item.get("common_categories", [])
        ]
        item["evaluation_category_ids"] = [
            int(category["dataset_id"])
            for category in item.get("common_categories", [])
        ]
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

LOCAL_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".svg"}


def _dataset_image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in LOCAL_IMAGE_SUFFIXES
        and "annotations" not in path.relative_to(directory).parts
    ]


def summarize_resolutions(dimensions: Iterable[tuple[int, int]]) -> str | None:
    valid = {
        (int(width), int(height))
        for width, height in dimensions
        if int(width) > 0 and int(height) > 0
    }
    if not valid:
        return None
    if len(valid) == 1:
        width, height = next(iter(valid))
        return f"{width}×{height}"
    widths = [width for width, _ in valid]
    heights = [height for _, height in valid]
    return f"{min(widths)}×{min(heights)} ～ {max(widths)}×{max(heights)}"


def summarize_image_resolutions(files: Iterable[Path]) -> str | None:
    dimensions = []
    for path in files:
        try:
            with Image.open(path) as image:
                dimensions.append(image.size)
        except (OSError, ValueError):
            continue
    return summarize_resolutions(dimensions)


def _referenced_dataset_source(directory: Path) -> Path | None:
    roots: set[Path] = set()
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_symlink()
            or path.suffix.lower() not in LOCAL_IMAGE_SUFFIXES
            or "annotations" in path.relative_to(directory).parts
        ):
            continue
        relative = path.relative_to(directory)
        target = path.readlink()
        target = (
            target.resolve()
            if target.is_absolute()
            else (path.parent / target).resolve()
        )
        root = target
        for _ in relative.parts:
            root = root.parent
        if (root / relative).resolve() != target:
            return None
        roots.add(root)
    return next(iter(roots)) if len(roots) == 1 else None


def _dataset_sample_name(directory: Path, path: Path) -> str:
    return path.relative_to(directory).as_posix()


def _match_dataset_sample_name(value: str, names: set[str]) -> str | None:
    normalized = Path(value.replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in names:
        return normalized
    suffix_matches = [
        name for name in names if normalized.endswith(f"/{name}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    basename = Path(normalized).name
    matches = [name for name in names if Path(name).name == basename]
    return matches[0] if len(matches) == 1 else None


def list_local_dataset_resources(
    path: str | None,
    kind: str,
    app_settings=None,
) -> dict[str, Any]:
    app_settings = app_settings or settings
    if kind not in {"directory", "annotation"}:
        raise DatasetImportError(f"不支持的数据资源类型: {kind}")
    root = app_settings.dataset_library_root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetImportError(f"本地数据根目录不存在: {root}")
    current = Path(path).expanduser().resolve() if path else root
    if not current.is_relative_to(root) or not current.is_dir():
        raise DatasetImportError(f"目录超出允许范围: {current}")
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
                {"name": child.name, "path": str(child), "is_directory": True}
            )
        elif kind == "annotation" and child.suffix.lower() == ".json":
            entries.append(
                {"name": child.name, "path": str(child), "is_directory": False}
            )
    return {
        "scope": "dataset",
        "kind": kind,
        "root": str(root),
        "current": str(current),
        "parent": str(current.parent) if current != root else None,
        "entries": entries,
    }


def resolve_local_dataset_import(
    values: dict[str, Any],
    app_settings=None,
) -> tuple[Path, Path | None]:
    app_settings = app_settings or settings
    root = app_settings.dataset_library_root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetImportError(f"本地数据根目录不存在: {root}")
    source = Path(str(values.get("directory") or "")).expanduser().resolve()
    if not source.is_relative_to(root) or not source.is_dir():
        raise DatasetImportError(f"图像目录必须位于 {root} 内")
    if not any(
        path.is_file() and path.suffix.lower() in LOCAL_IMAGE_SUFFIXES
        for path in source.rglob("*")
    ):
        raise DatasetImportError("所选目录中没有支持的图像")
    annotation_value = values.get("annotation_path")
    if not annotation_value:
        return source, None
    annotation = Path(str(annotation_value)).expanduser().resolve()
    if not annotation.is_relative_to(root):
        raise DatasetImportError(f"标注路径必须位于 {root} 内")
    annotation_format = str(values.get("annotation_format") or "COCO").upper()
    if annotation_format == "COCO":
        if not annotation.is_file() or annotation.suffix.lower() != ".json":
            raise DatasetImportError("COCO 标注必须选择 JSON 文件")
    elif annotation_format in {"YOLO", "VISDRONE"}:
        if not annotation.is_dir():
            raise DatasetImportError(f"{annotation_format} 标注必须选择目录")
    else:
        raise DatasetImportError("标注格式不受支持")
    return source, annotation


def _parse_yolo_category_file(annotation_directory: Path) -> tuple[Path, list[dict[str, Any]]]:
    names_files = sorted(annotation_directory.rglob("*.names"))
    if names_files:
        path = names_files[0]
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if names:
            return path, [
                {"id": index, "name": name}
                for index, name in enumerate(names)
            ]

    yaml_files = sorted(
        annotation_directory.rglob("*.yaml")
    ) + sorted(annotation_directory.rglob("*.yml"))
    for path in yaml_files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("names:"):
                continue
            inline = stripped.partition(":")[2].strip()
            if inline:
                try:
                    parsed = ast.literal_eval(inline)
                except (SyntaxError, ValueError):
                    parsed = [item.strip(" '\"") for item in inline.strip("[]").split(",")]
                if isinstance(parsed, dict):
                    categories = [
                        {"id": int(category_id), "name": str(name)}
                        for category_id, name in parsed.items()
                    ]
                elif isinstance(parsed, (list, tuple)):
                    categories = [
                        {"id": category_id, "name": str(name)}
                        for category_id, name in enumerate(parsed)
                    ]
                else:
                    categories = []
                if categories:
                    return path, normalize_categories(categories)

            block: list[dict[str, Any]] = []
            for item in lines[index + 1 :]:
                if item and not item[0].isspace():
                    break
                value = item.strip()
                if not value or value.startswith("#"):
                    continue
                if value.startswith("-"):
                    block.append(
                        {
                            "id": len(block),
                            "name": value[1:].strip().strip("'\""),
                        }
                    )
                    continue
                category_id, separator, name = value.partition(":")
                if separator and category_id.strip().isdigit():
                    block.append(
                        {
                            "id": int(category_id.strip()),
                            "name": name.split("#", 1)[0].strip().strip("'\""),
                        }
                    )
            if block:
                return path, normalize_categories(block)
    raise DatasetImportError(
        "YOLO 标注目录中没有可读取的类别定义；请提供 data.yaml、*.yaml 或 *.names"
    )


def read_dataset_annotation_categories(
    annotation_path: str | Path,
    annotation_format: str,
    app_settings=None,
) -> dict[str, Any]:
    app_settings = app_settings or settings
    root = app_settings.dataset_library_root.expanduser().resolve()
    path = Path(annotation_path).expanduser().resolve()
    if not path.is_relative_to(root):
        raise DatasetImportError(f"标注路径必须位于 {root} 内")
    annotation_format = annotation_format.upper()
    if annotation_format == "COCO":
        if not path.is_file() or path.suffix.lower() != ".json":
            raise DatasetImportError("COCO 标注必须选择 JSON 文件")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetImportError(f"COCO 标注文件无法读取: {exc}") from exc
        try:
            categories = normalize_categories(payload.get("categories", []))
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetImportError(f"COCO 标注没有有效的 categories: {exc}") from exc
        return {
            "categories": categories,
            "category_template": "annotation",
            "source": str(path),
        }
    if annotation_format == "VISDRONE":
        if not path.is_dir():
            raise DatasetImportError("VISDRONE 标注必须选择目录")
        return {
            "categories": template_categories("visdrone", "dataset"),
            "category_template": "visdrone",
            "source": "VisDrone 标准类别模板",
        }
    if annotation_format == "YOLO":
        if not path.is_dir():
            raise DatasetImportError("YOLO 标注必须选择目录")
        category_path, categories = _parse_yolo_category_file(path)
        return {
            "categories": categories,
            "category_template": "annotation",
            "source": str(category_path),
        }
    raise DatasetImportError("标注格式不受支持")


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
        placeholders = command_placeholders(values["command_arguments"])
    except CommandTemplateError as exc:
        raise LocalModelRegistrationError(str(exc)) from exc
    try:
        categories = normalize_categories(values["categories"])
    except ValueError as exc:
        raise LocalModelRegistrationError(str(exc)) from exc

    adapter_id = new_id("adapter")
    model_id = new_id("model")
    now = utc_now()
    defaults = {
        "confidence": 0.25,
        "nms_iou": 0.7,
        "image_size": 1280,
        "input_height": 960,
        "input_width": 1280,
        "max_detections": 300,
        "batch_size": 1,
        "warmup": 0,
        **values.get("inference_defaults", {}),
    }
    inference_properties = {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "nms_iou": {"type": "number", "minimum": 0, "maximum": 1},
        "image_size": {"type": "integer", "minimum": 32, "maximum": 8192},
        "input_height": {"type": "integer", "minimum": 32, "maximum": 8192},
        "input_width": {"type": "integer", "minimum": 32, "maximum": 8192},
        "max_detections": {"type": "integer", "minimum": 1, "maximum": 5000},
        "batch_size": {"type": "integer", "minimum": 1, "maximum": 64},
        "warmup": {"type": "integer", "minimum": 0, "maximum": 200},
    }
    for name, specification in inference_properties.items():
        if name in placeholders:
            specification["default"] = defaults[name]
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
            **{
                name: specification
                for name, specification in inference_properties.items()
                if name in placeholders
            },
            **(
                {
                    "precision": {
                        "type": "string",
                        "enum": ["FP16", "FP32"],
                        "default": values["precision"],
                    }
                }
                if "precision" in placeholders
                else {}
            ),
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


def update_local_detector_model(
    model_id: str,
    values: dict[str, Any],
    database: Database = db,
) -> dict[str, Any] | None:
    model = database.row("SELECT * FROM models WHERE id=?", (model_id,))
    if not model:
        return None
    if model["is_demo"]:
        raise LocalModelRegistrationError("流程样例模型不支持编辑")
    adapter = database.row(
        "SELECT * FROM adapters WHERE id=?", (model["adapter_id"],)
    )
    if not adapter:
        raise LocalModelRegistrationError("模型的推理适配器不存在")

    schema = json_load(adapter["parameter_schema"], {})
    execution = schema.get("execution", {})
    properties = schema.get("properties", {})
    project_directory_value = properties.get("project_directory", {}).get("const")
    if execution.get("mode") != "command" or not project_directory_value:
        raise LocalModelRegistrationError("该模型没有可编辑的本地命令配置")

    app_settings = database.settings
    model_root = app_settings.model_library_root.expanduser().resolve()
    environment_root = app_settings.model_environment_root.expanduser().resolve()
    project_directory = Path(project_directory_value).expanduser().resolve()
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
    runtime_prefix = project_path(values["runtime_prefix"])
    if not (
        runtime_prefix.is_relative_to(project_directory)
        or runtime_prefix.is_relative_to(environment_root)
    ):
        raise LocalModelRegistrationError(
            "Python 环境必须位于模型目录或允许的环境根目录内"
        )
    executable = runtime_prefix / "bin" / "python"
    if not executable.is_file():
        raise LocalModelRegistrationError(
            f"Python 解释器不存在: {executable}"
        )
    if not os.access(executable, os.X_OK):
        raise LocalModelRegistrationError(
            f"Python 解释器没有执行权限: {executable}"
        )

    for name, value in values["inference_defaults"].items():
        specification = properties.get(name)
        if isinstance(specification, dict):
            specification["default"] = value
    precision = model["precision"]
    if isinstance(properties.get("precision"), dict):
        properties["precision"]["default"] = values["precision"]
        precision = values["precision"]
    execution["working_directory"] = str(working_directory)
    execution["executable"] = str(executable)

    with database.connect() as connection:
        connection.execute(
            """
            UPDATE models
            SET name=?,architecture=?,backbone=?,detector_head=?,training_dataset=?,
                pretrained_dataset=?,precision=?
            WHERE id=?
            """,
            (
                values["name"],
                values["architecture"],
                values["backbone"],
                values["detector_head"],
                values["training_dataset"],
                values["pretrained_dataset"],
                precision,
                model_id,
            ),
        )
        connection.execute(
            """
            UPDATE adapters
            SET name=?,runtime_prefix=?,entrypoint=?,status='REGISTERED',
                parameter_schema=?,updated_at=?
            WHERE id=?
            """,
            (
                f"{values['name']} 推理适配器",
                str(runtime_prefix),
                str(executable),
                json_dump(schema),
                utc_now(),
                adapter["id"],
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
                "label_zh": (
                    {
                        "autonomous-driving": "自动驾驶",
                        "low-altitude-uav": "低空无人机",
                    }.get(
                        catalog["domain"],
                        catalog.get("label_zh", catalog["domain"]),
                    )
                ),
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
            files = _dataset_image_files(directory)
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
                "name": _dataset_sample_name(directory, path),
                "url": f"/artifacts/{quote(path.relative_to(database.settings.artifact_dir.resolve()).as_posix(), safe='/')}",
                **visualizations.get(
                    _dataset_sample_name(directory, path),
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
    if not files or not directory:
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
    sample_names = {
        path: _dataset_sample_name(directory, path)
        for path in files
    }
    names = set(sample_names.values())
    all_names = {
        _dataset_sample_name(directory, path)
        for path in _dataset_image_files(directory)
    }
    for path in files:
        name = sample_names[path]
        try:
            with Image.open(path) as image:
                dimensions[name] = (image.width, image.height)
        except OSError:
            dimensions[name] = (0, 0)

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
        images_by_id = {}
        coco_names = {}
        for image in coco.get("images", []):
            name = _match_dataset_sample_name(
                str(image.get("file_name", "")),
                all_names,
            )
            if name in names:
                image_id = int(image["id"])
                images_by_id[image_id] = image
                coco_names[image_id] = name
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image_id, image in images_by_id.items():
            name = coco_names[image_id]
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
            key = path.relative_to(root).with_suffix("").as_posix()
            label_files.setdefault(key, (path, source))
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
            name = sample_names[path]
            if name in output:
                continue
            label_key = Path(name).with_suffix("").as_posix()
            label_entry = label_files.get(label_key)
            if not label_entry:
                path_matches = [
                    entry
                    for key, entry in label_files.items()
                    if key.endswith(f"/{label_key}")
                    or label_key.endswith(f"/{key}")
                ]
                label_entry = path_matches[0] if len(path_matches) == 1 else None
            if not label_entry:
                basename_matches = [
                    entry
                    for key, entry in label_files.items()
                    if Path(key).name == Path(label_key).name
                ]
                label_entry = (
                    basename_matches[0]
                    if len(basename_matches) == 1
                    else None
                )
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
            width, height = dimensions.get(name, (0, 0))
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
            output[name] = {
                "width": width,
                "height": height,
                "boxes": boxes,
                "annotation_source": source,
            }

    for path in files:
        name = sample_names[path]
        if name not in output:
            width, height = dimensions.get(name, (0, 0))
            output[name] = {
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
        files = _dataset_image_files(directory)
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
    relative_scale_counts = {
        "under_0_1": 0,
        "0_1_to_1": 0,
        "1_to_10": 0,
        "over_10": 0,
        "unknown": 0,
    }
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
                relative_scale_counts["unknown"] += 1
                continue
            relative_area = max(0.0, float(box["width"])) * max(
                0.0, float(box["height"])
            )
            area = relative_area * width * height
            if area < 32 ** 2:
                scale_counts["small"] += 1
            elif area < 96 ** 2:
                scale_counts["medium"] += 1
            else:
                scale_counts["large"] += 1
            if relative_area < 0.001:
                relative_scale_counts["under_0_1"] += 1
            elif relative_area < 0.01:
                relative_scale_counts["0_1_to_1"] += 1
            elif relative_area < 0.1:
                relative_scale_counts["1_to_10"] += 1
            else:
                relative_scale_counts["over_10"] += 1
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
        "relative_scales": [
            {"key": "under_0_1", "label": "< 0.1%", "count": relative_scale_counts["under_0_1"]},
            {"key": "0_1_to_1", "label": "0.1%–1%", "count": relative_scale_counts["0_1_to_1"]},
            {"key": "1_to_10", "label": "1%–10%", "count": relative_scale_counts["1_to_10"]},
            {"key": "over_10", "label": "≥ 10%", "count": relative_scale_counts["over_10"]},
            {"key": "unknown", "label": "尺寸未知", "count": relative_scale_counts["unknown"]},
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
    dataset, directory, files = resolved
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
    imported = _sample_visualizations(dataset_id, directory, files, database)
    samples = []
    artifact_root = database.settings.artifact_dir.resolve()
    for path in files:
        name = _dataset_sample_name(directory, path)
        row = rows.get(name)
        visualization = imported.get(name, {})
        completed = (
            bool(row["completed"])
            if row
            else bool(dataset["frozen"] and visualization.get("annotation_source"))
        )
        samples.append(
            {
                "name": name,
                "url": (
                    f"/artifacts/{quote(path.relative_to(artifact_root).as_posix(), safe='/')}"
                ),
                "completed": completed,
                "box_count": (
                    len(json_load(row["boxes"], []))
                    if row
                    else len(visualization.get("boxes", []))
                ),
            }
        )
    return {
        "dataset": decode_row(dataset),
        "categories": _annotation_categories(dataset_id, database),
        "progress": {
            "completed": sum(bool(sample["completed"]) for sample in samples),
            "total": len(samples),
        },
        "samples": samples,
    }


def get_sample_annotation(
    dataset_id: str,
    sample_name: str,
    database: Database = db,
) -> dict[str, Any] | None:
    resolved = _dataset_images(dataset_id, database)
    if not resolved:
        return None
    dataset, directory, files = resolved
    sample_path = next(
        (
            path
            for path in files
            if directory and _dataset_sample_name(directory, path) == sample_name
        ),
        None,
    )
    if not sample_path:
        return None
    row = database.row(
        """
        SELECT width,height,boxes,completed,updated_at
        FROM sample_annotations WHERE dataset_id=? AND sample_name=?
        """,
        (dataset_id, sample_name),
    )
    if row:
        return {
            "dataset_id": dataset_id,
            "sample_name": sample_name,
            "width": row["width"],
            "height": row["height"],
            "boxes": json_load(row["boxes"], []),
            "completed": bool(row["completed"]),
            "updated_at": row["updated_at"],
        }
    visualization = _sample_visualizations(
        dataset_id, directory, [sample_path], database
    ).get(sample_name, {})
    categories_by_name = {
        normalize_category_name(category["name"]): int(category["id"])
        for category in _annotation_categories(dataset_id, database)
    }
    width = int(visualization.get("width", 0))
    height = int(visualization.get("height", 0))
    boxes = []
    for index, box in enumerate(visualization.get("boxes", []), start=1):
        category_id = categories_by_name.get(normalize_category_name(box["label"]))
        if category_id is None:
            continue
        boxes.append(
            {
                "id": f"imported_{index}",
                "category_id": category_id,
                "x": float(box["x"]) * width,
                "y": float(box["y"]) * height,
                "width": float(box["width"]) * width,
                "height": float(box["height"]) * height,
            }
        )
    return {
        "dataset_id": dataset_id,
        "sample_name": sample_name,
        "width": width,
        "height": height,
        "boxes": boxes,
        "completed": bool(dataset["frozen"] and visualization.get("annotation_source")),
        "updated_at": None,
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
    dataset, directory, files = resolved
    if dataset["frozen"]:
        raise DatasetAnnotationError("冻结数据集的标注不可修改")
    if not directory or sample_name not in {
        _dataset_sample_name(directory, path) for path in files
    }:
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
        _dataset_sample_name(directory, path)
        for path in files
        if _dataset_sample_name(directory, path) not in rows
        or not rows[_dataset_sample_name(directory, path)]["completed"]
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
        name = _dataset_sample_name(directory, path)
        row = rows[name]
        images.append(
            {
                "id": image_id,
                "file_name": name,
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
            run_count = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE dataset_id=?", (dataset_id,)
            ).fetchone()[0]
            if run_count:
                raise DatasetDeletionError(
                    f"数据集已被 {run_count} 个评测运行引用，不能删除"
                )
            referenced_plans = [
                dict(plan)
                for plan in connection.execute(
                    "SELECT * FROM evaluation_plans"
                ).fetchall()
                if dataset_id in json_load(plan["dataset_ids"], [])
            ]
            referenced_plan_ids = {plan["id"] for plan in referenced_plans}
            active_plan_jobs = [
                job["id"]
                for job in connection.execute(
                    "SELECT id,status,payload FROM jobs"
                ).fetchall()
                if job["status"] not in {"SUCCEEDED", "FAILED", "CANCELLED"}
                and json_load(job["payload"], {}).get("plan_id")
                in referenced_plan_ids
            ]
            if active_plan_jobs:
                raise DatasetDeletionError(
                    f"评测任务 {active_plan_jobs[0]} 尚未结束，不能删除数据集"
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
                "evaluation_plans": referenced_plans,
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
            for plan in referenced_plans:
                remaining_dataset_ids = [
                    value
                    for value in json_load(plan["dataset_ids"], [])
                    if value != dataset_id
                ]
                plan_run_count = connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE plan_id=?", (plan["id"],)
                ).fetchone()[0]
                if remaining_dataset_ids or plan_run_count:
                    connection.execute(
                        "UPDATE evaluation_plans SET dataset_ids=? WHERE id=?",
                        (json_dump(remaining_dataset_ids), plan["id"]),
                    )
                else:
                    connection.execute(
                        "DELETE FROM evaluation_plans WHERE id=?", (plan["id"],)
                    )
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
        "evaluation_plans_updated": len(referenced_plans),
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


def delete_evaluation_result(
    run_id: str,
    database: Database = db,
) -> dict[str, Any] | None:
    artifact_root = (database.settings.artifact_dir / "evaluations").resolve()
    task_root = database.settings.task_dir.resolve()
    trash_directory = (
        database.settings.data_dir / "trash" / "evaluation_runs" / run_id
    ).resolve()
    manifest = trash_directory / "evaluation.json"
    moved_paths: list[tuple[Path, Path]] = []
    trash_created = False
    run: dict[str, Any] | None = None
    try:
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run_row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            result_row = connection.execute(
                "SELECT * FROM results WHERE run_id=?", (run_id,)
            ).fetchone()
            if not run_row or not result_row:
                return None
            run = decode_row(dict(run_row))
            result = decode_row(dict(result_row))
            if run.get("job_id"):
                job = connection.execute(
                    "SELECT status FROM jobs WHERE id=?", (run["job_id"],)
                ).fetchone()
                if job and job["status"] not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                    raise EvaluationResultDeletionError(
                        "所属评测任务尚未结束，不能删除结果"
                    )

            artifact = (artifact_root / run_id).resolve()
            if artifact == artifact_root or not artifact.is_relative_to(artifact_root):
                raise EvaluationResultDeletionError("评测 Artifact 路径超出受控目录")
            paths = [(artifact, trash_directory / "artifact")]
            if run.get("job_id"):
                workspace = (
                    task_root / str(run["job_id"]) / "runs" / run_id
                ).resolve()
                if workspace == task_root or not workspace.is_relative_to(task_root):
                    raise EvaluationResultDeletionError("评测工作目录超出受控目录")
                paths.append((workspace, trash_directory / "workspace"))

            if trash_directory.exists():
                raise EvaluationResultDeletionError(
                    f"回收站目标已存在: {trash_directory}"
                )
            trash_directory.mkdir(parents=True)
            trash_created = True
            manifest.write_text(
                json.dumps(
                    {"run": run, "result": result},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            for source, target in paths:
                if source.exists():
                    shutil.move(str(source), str(target))
                    moved_paths.append((source, target))

            connection.execute("DELETE FROM results WHERE run_id=?", (run_id,))
            connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
    except BaseException:
        for source, target in reversed(moved_paths):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        if trash_created and trash_directory.exists():
            shutil.rmtree(trash_directory)
        raise
    assert run is not None
    return {
        "id": run_id,
        "deleted": True,
        "artifact_moved": any(target.name == "artifact" for _, target in moved_paths),
        "workspace_moved": any(target.name == "workspace" for _, target in moved_paths),
        "trash_path": str(trash_directory),
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


INFERENCE_COMPARISON_KEYS = (
    "precision",
    "batch_size",
    "warmup",
    "confidence",
    "nms_iou",
    "image_size",
    "input_height",
    "input_width",
    "max_detections",
    "blur_level",
    "metric_protocol",
    "timing",
)


def _comparison_config(
    config: dict[str, Any],
    model_precision: str | None = None,
) -> dict[str, Any]:
    comparable = {
        key: config[key]
        for key in INFERENCE_COMPARISON_KEYS
        if key in config
    }
    if "precision" not in comparable and model_precision:
        comparable["precision"] = model_precision
    if "input_height" in comparable and "input_width" in comparable:
        comparable["input_resolution"] = (
            f"{comparable['input_width']}×{comparable['input_height']}"
        )
    elif "image_size" in comparable:
        comparable["input_resolution"] = (
            f"{comparable['image_size']}×{comparable['image_size']}"
        )
    return comparable


def _condition_metadata(
    weather: str,
    sensor_conditions: dict[str, Any],
) -> dict[str, Any]:
    source_dataset_id = sensor_conditions.get("source_dataset_id")
    if sensor_conditions.get("day_to_night") is True:
        return {
            "condition_type": str(
                sensor_conditions.get("condition_label")
                or (
                    "自动驾驶弱光"
                    if sensor_conditions.get("day_to_night_method") == "unpaired"
                    else "无人机弱光"
                )
            ),
            "condition_strength": None,
            "source_dataset_id": source_dataset_id,
        }
    if "fog_strength" in sensor_conditions:
        return {
            "condition_type": str(
                sensor_conditions.get("condition_label")
                or (
                    "无人机气雾"
                    if sensor_conditions.get("degradation")
                    == "DiffusionDegrade UAV Fog"
                    else "雾"
                )
            ),
            "condition_strength": float(sensor_conditions["fog_strength"]),
            "source_dataset_id": source_dataset_id,
        }
    if "fog_density" in sensor_conditions:
        return {
            "condition_type": "雾",
            "condition_strength": float(sensor_conditions["fog_density"]),
            "source_dataset_id": source_dataset_id,
        }
    if sensor_conditions.get("fog_method") or sensor_conditions.get("fog_model"):
        return {
            "condition_type": str(
                sensor_conditions.get("condition_label")
                or (
                    "自动驾驶气雾"
                    if sensor_conditions.get("fog_method") == "paired"
                    or sensor_conditions.get("degradation")
                    == "WarpI2I Driving Fog"
                    else "雾"
                )
            ),
            "condition_strength": None,
            "source_dataset_id": source_dataset_id,
        }
    if "motion_blur_strength" in sensor_conditions:
        return {
            "condition_type": "无人机运动模糊",
            "condition_strength": float(sensor_conditions["motion_blur_strength"]),
            "source_dataset_id": source_dataset_id,
        }
    motion_blur = sensor_conditions.get("motion_blur")
    if isinstance(motion_blur, (int, float)) and not isinstance(motion_blur, bool):
        return {
            "condition_type": "运动模糊",
            "condition_strength": float(motion_blur),
            "source_dataset_id": source_dataset_id,
        }
    degradation = sensor_conditions.get("degradation")
    if degradation:
        return {
            "condition_type": str(degradation),
            "condition_strength": None,
            "source_dataset_id": source_dataset_id,
        }
    recorded_condition = str(
        sensor_conditions.get("recorded_condition") or ""
    )
    if recorded_condition:
        return {
            "condition_type": (
                "基准" if recorded_condition == "无" else recorded_condition
            ),
            "condition_strength": 0.0 if recorded_condition == "无" else None,
            "source_dataset_id": source_dataset_id,
        }
    if weather not in {"", "晴朗", "未记录"}:
        return {
            "condition_type": weather,
            "condition_strength": None,
            "source_dataset_id": source_dataset_id,
        }
    return {
        "condition_type": "基准",
        "condition_strength": 0.0,
        "source_dataset_id": source_dataset_id,
    }


def evaluation_run_visualization(
    run_id: str,
    offset: int,
    limit: int,
    database: Database = db,
) -> dict[str, Any] | None:
    run = database.row(
        """
        SELECT x.*,d.name AS dataset_name,d.artifact_path,
               m.name AS model_name,m.is_demo
        FROM runs x
        JOIN datasets d ON d.id=x.dataset_id
        JOIN models m ON m.id=x.model_id
        JOIN results r ON r.run_id=x.id
        WHERE x.id=?
        """,
        (run_id,),
    )
    if not run:
        return None
    config = json_load(run.get("config"), {})
    prediction_value = config.get("predictions_path")
    if not prediction_value:
        raise DatasetArtifactError("该运行没有保存预测结果，无法可视化")
    artifact_root = database.settings.artifact_dir.resolve()
    prediction_relative = Path(str(prediction_value))
    predictions_path = (artifact_root / prediction_relative).resolve()
    if (
        prediction_relative.is_absolute()
        or ".." in prediction_relative.parts
        or not predictions_path.is_relative_to(artifact_root)
        or not predictions_path.is_file()
    ):
        raise DatasetArtifactError("该运行的预测结果文件不存在")
    dataset_relative = Path(str(run.get("artifact_path") or ""))
    dataset_directory = (artifact_root / dataset_relative).resolve()
    if (
        not run.get("artifact_path")
        or not dataset_directory.is_relative_to(artifact_root)
        or not dataset_directory.is_dir()
    ):
        raise DatasetArtifactError("该运行关联的数据集目录不存在")

    files = _dataset_image_files(dataset_directory)
    names = {
        _dataset_sample_name(dataset_directory, path): path
        for path in files
    }
    image_paths: dict[int, Path] = {}
    coco_path = dataset_directory / "annotations" / "instances.json"
    if coco_path.is_file():
        try:
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetArtifactError(f"数据集 COCO 标注无法读取: {exc}") from exc
        for image in coco.get("images", []):
            matched = _match_dataset_sample_name(
                str(image.get("file_name", "")), set(names)
            )
            if matched:
                image_paths[int(image["id"])] = names[matched]
    if not image_paths:
        image_paths = {
            image_id: path
            for image_id, path in enumerate(files, start=1)
        }

    try:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetArtifactError(f"预测结果无法读取: {exc}") from exc
    if not isinstance(predictions, list):
        raise DatasetArtifactError("预测结果不是 COCO predictions 数组")

    colors = [
        "#1677FF", "#13A8A8", "#722ED1", "#EB2F96",
        "#52C41A", "#FA8C16", "#F5222D", "#2F54EB",
    ]
    categories = {
        int(category["id"]): {
            "name": str(category["name"]),
            "color": str(category.get("color") or colors[index % len(colors)]),
        }
        for index, category in enumerate(
            _annotation_categories(run["dataset_id"], database)
        )
    }
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        try:
            image_id = int(prediction["image_id"])
            category_id = int(prediction["category_id"])
            score = float(prediction["score"])
            bbox = [float(value) for value in prediction["bbox"]]
        except (KeyError, TypeError, ValueError):
            continue
        if image_id in image_paths and len(bbox) == 4:
            predictions_by_image[image_id].append(
                {"category_id": category_id, "score": score, "bbox": bbox}
            )

    ordered_images = sorted(
        image_paths.items(),
        key=lambda item: _dataset_sample_name(dataset_directory, item[1]),
    )
    page = ordered_images[offset : offset + limit]
    items = []
    for image_id, path in page:
        try:
            with Image.open(path) as image:
                width, height = image.width, image.height
        except OSError:
            width, height = 0, 0
        boxes = []
        if width > 0 and height > 0:
            for prediction in predictions_by_image.get(image_id, []):
                x, y, box_width, box_height = prediction["bbox"]
                left = max(0.0, min(float(width), x))
                top = max(0.0, min(float(height), y))
                right = max(left, min(float(width), x + box_width))
                bottom = max(top, min(float(height), y + box_height))
                if right == left or bottom == top:
                    continue
                category_id = prediction["category_id"]
                category = categories.get(
                    category_id,
                    {
                        "name": f"class {category_id}",
                        "color": colors[category_id % len(colors)],
                    },
                )
                boxes.append(
                    {
                        "category_id": category_id,
                        "label": category["name"],
                        "color": category["color"],
                        "score": prediction["score"],
                        "x": left / width,
                        "y": top / height,
                        "width": (right - left) / width,
                        "height": (bottom - top) / height,
                    }
                )
        items.append(
            {
                "image_id": image_id,
                "name": _dataset_sample_name(dataset_directory, path),
                "url": f"/artifacts/{quote(path.relative_to(artifact_root).as_posix(), safe='/')}",
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
    return {
        "run_id": run_id,
        "dataset_id": run["dataset_id"],
        "dataset_name": run["dataset_name"],
        "model_id": run["model_id"],
        "model_name": run["model_name"],
        "inference_confidence": float(config.get("confidence", 0.0)),
        "total": len(ordered_images),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(ordered_images),
        "items": items,
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
    dataset_categories = {
        dataset_id: [
            str(category["name"])
            for category in _annotation_categories(dataset_id, database)
        ]
        for dataset_id in {row["dataset_id"] for row in decoded}
    }
    for row in decoded:
        configured_categories = (row.get("config") or {}).get(
            "evaluation_categories"
        )
        row["evaluation_categories"] = (
            [str(name) for name in configured_categories]
            if configured_categories
            else dataset_categories.get(row["dataset_id"], [])
        )
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decoded:
        comparison_config = _comparison_config(
            row.get("config") or {},
            row.get("model_precision"),
        )
        comparison_scope = {
            "inference_config": comparison_config,
            "evaluation_categories": row["evaluation_categories"],
        }
        configuration_signature = json.dumps(
            comparison_scope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        hardware_signature = json.dumps(
            {
                "environment_fingerprint": row.get("environment_fingerprint"),
                "hardware_profile": row.get("hardware_profile") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        groups[
            (
                row["dataset_id"],
                row["model_id"],
                configuration_signature,
                hardware_signature,
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for values in groups.values():
        map_values = [item["map"] for item in values]
        first = values[0]
        mean = sum(map_values) / len(map_values)
        variance = sum((value - mean) ** 2 for value in map_values) / len(map_values)
        inference_config = _comparison_config(
            first.get("config") or {},
            first.get("model_precision"),
        )
        evaluation_categories = list(first["evaluation_categories"])
        configuration_id = hashlib.sha256(
            json.dumps(
                {
                    "inference_config": inference_config,
                    "evaluation_categories": evaluation_categories,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        hardware_id = hashlib.sha256(
            json.dumps(
                {
                    "environment_fingerprint": first.get("environment_fingerprint"),
                    "hardware_profile": first.get("hardware_profile") or {},
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        condition = _condition_metadata(
            str(first["weather"]),
            first.get("sensor_conditions") or {},
        )

        def metric_mean(name: str) -> float:
            return sum(float(item[name]) for item in values) / len(values)

        summaries.append(
            {
                "comparison_id": (
                    f"{first['dataset_id']}:{first['model_id']}:{configuration_id}:"
                    f"{hardware_id}"
                ),
                "configuration_id": configuration_id,
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
                "inference_config": inference_config,
                "evaluation_categories": evaluation_categories,
                "hardware_profile": first.get("hardware_profile") or {},
                "environment_fingerprint": first.get("environment_fingerprint"),
                **condition,
                "map_mean": round(mean, 4),
                "map_std": round(math.sqrt(variance), 4),
                "map50_mean": round(metric_mean("map50"), 4),
                "map75_mean": round(metric_mean("map75"), 4),
                "precision_mean": round(metric_mean("precision"), 4),
                "recall_mean": round(metric_mean("recall"), 4),
                "f1_mean": round(metric_mean("f1"), 4),
                "latency_mean": round(metric_mean("latency_p50"), 2),
                "latency_p95_mean": round(metric_mean("latency_p95"), 2),
                "fps_mean": round(metric_mean("fps"), 2),
                "peak_memory_mean": round(metric_mean("peak_memory"), 2),
                "delta_map_mean": round(sum(item["delta_map"] or 0 for item in values) / len(values), 4),
                "seed_count": len(values),
                "seeds": sorted({int(item["seed"]) for item in values}),
                "run_ids": [item["run_id"] for item in values],
                "curves": first["curves"],
            }
        )
    summaries.sort(key=lambda item: item["map_mean"], reverse=True)
    dimensions = {
        "scenes": sorted({row["scene_domain"] for row in decoded}),
        "conditions": sorted({row["weather"] for row in decoded}),
        "resolutions": sorted({row["resolution"] for row in decoded}),
        "models": sorted({row["model_name"] for row in decoded}),
        "model_options": sorted(
            {
                (row["model_id"], row["model_name"])
                for row in decoded
            },
            key=lambda item: item[1],
        ),
        "dataset_options": sorted(
            {
                (row["dataset_id"], row["dataset_name"])
                for row in decoded
            },
            key=lambda item: item[1],
        ),
        "condition_types": sorted(
            {summary["condition_type"] for summary in summaries}
        ),
        "hardware": sorted(
            {
                str(row.get("environment_fingerprint") or "unknown")
                for row in decoded
            }
        ),
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
        if adapter["id"] in {"adapter_condition", "adapter_day_to_night"}:
            adapter_settings = database.settings
            diffusion_root = adapter_settings.diffusion_degrade_root
            is_day_to_night = adapter["id"] == "adapter_day_to_night"
            checkpoint = (
                adapter_settings.diffusion_degrade_day_to_night_checkpoint
                if is_day_to_night
                else adapter_settings.diffusion_degrade_checkpoint
            )
            checkpoint_label = (
                "无人机弱光权重" if is_day_to_night else "无人机气雾权重"
            )
            dependency_label = "弱光依赖" if is_day_to_night else "气雾依赖"
            checkpoint_environment = (
                "DIFFUSION_DEGRADE_UAV_DAY_TO_NIGHT_CHECKPOINT"
                if is_day_to_night
                else "DIFFUSION_DEGRADE_UAV_FOG_CHECKPOINT"
            )
            model_cache = (
                adapter_settings.diffusion_degrade_hf_home
                / "hub"
                / "models--stabilityai--sd-turbo"
            )
            checks.extend(
                [
                    {
                        "name": "DiffusionDegrade项目",
                        "ok": (
                            diffusion_root / "src" / "cyclegan_turbo.py"
                        ).is_file(),
                        "detail": str(diffusion_root),
                    },
                    {
                        "name": checkpoint_label,
                        "ok": checkpoint.is_file(),
                        "detail": str(checkpoint),
                    },
                    {
                        "name": "SD-Turbo本地缓存",
                        "ok": model_cache.is_dir(),
                        "detail": str(model_cache),
                    },
                ]
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
                        "DIFFUSION_DEGRADE_ROOT": str(diffusion_root),
                        checkpoint_environment: str(checkpoint),
                        "HF_HOME": str(
                            adapter_settings.diffusion_degrade_hf_home
                        ),
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                checks.append(
                    {
                        "name": dependency_label,
                        "ok": result.returncode == 0,
                        "detail": (result.stdout or result.stderr).strip()[-500:],
                    }
                )
            except (OSError, subprocess.SubprocessError) as exc:
                checks.append(
                    {"name": dependency_label, "ok": False, "detail": str(exc)}
                )
        if adapter["id"] in {
            "adapter_warpi2i_fog",
            "adapter_warpi2i_day_to_night",
        }:
            adapter_settings = database.settings
            warpi2i_root = adapter_settings.warpi2i_root
            is_fog = adapter["id"] == "adapter_warpi2i_fog"
            checkpoint = (
                adapter_settings.warpi2i_driving_fog_checkpoint
                if is_fog
                else adapter_settings.warpi2i_driving_day_to_night_checkpoint
            )
            effect = "fog" if is_fog else "day_to_night"
            checks.extend(
                [
                    {
                        "name": "WarpI2I项目",
                        "ok": (warpi2i_root / "src" / "model.py").is_file(),
                        "detail": str(warpi2i_root),
                    },
                    {
                        "name": "自动驾驶气雾权重" if is_fog else "自动驾驶弱光权重",
                        "ok": checkpoint.is_file(),
                        "detail": str(checkpoint),
                    },
                    {
                        "name": "SD-Turbo本地缓存",
                        "ok": (
                            adapter_settings.diffusion_degrade_hf_home
                            / "hub"
                            / "models--stabilityai--sd-turbo"
                        ).is_dir(),
                        "detail": str(adapter_settings.diffusion_degrade_hf_home),
                    },
                ]
            )
            try:
                result = subprocess.run(
                    [
                        str(prefix / "bin" / "python"),
                        "-B",
                        str(entrypoint),
                        "health",
                        "--effect",
                        effect,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={
                        **os.environ,
                        "WARPI2I_ROOT": str(warpi2i_root),
                        "WARPI2I_DRIVING_FOG_CHECKPOINT": str(
                            adapter_settings.warpi2i_driving_fog_checkpoint
                        ),
                        "WARPI2I_DRIVING_DAY_TO_NIGHT_CHECKPOINT": str(
                            adapter_settings.warpi2i_driving_day_to_night_checkpoint
                        ),
                        "HF_HOME": str(
                            adapter_settings.diffusion_degrade_hf_home
                        ),
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                checks.append(
                    {
                        "name": "自动驾驶气雾依赖" if is_fog else "自动驾驶弱光依赖",
                        "ok": result.returncode == 0,
                        "detail": (result.stdout or result.stderr).strip()[-500:],
                    }
                )
            except (OSError, subprocess.SubprocessError) as exc:
                checks.append(
                    {"name": "WarpI2I依赖", "ok": False, "detail": str(exc)}
                )
        if adapter["id"] == "adapter_motion_blur":
            adapter_settings = database.settings
            blur_root = adapter_settings.diffusion_blur_root
            checkpoint = adapter_settings.diffusion_blur_checkpoint
            checks.extend(
                [
                    {
                        "name": "DiffusionBlur项目",
                        "ok": (blur_root / "reblur" / "pipeline.py").is_file(),
                        "detail": str(blur_root),
                    },
                    {
                        "name": "ID-Blau权重",
                        "ok": checkpoint.is_file(),
                        "detail": str(checkpoint),
                    },
                ]
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
                        "DIFFUSION_BLUR_ROOT": str(blur_root),
                        "DIFFUSION_BLUR_CHECKPOINT": str(checkpoint),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                checks.append(
                    {
                        "name": "运动模糊依赖",
                        "ok": result.returncode == 0,
                        "detail": (result.stdout or result.stderr).strip()[-500:],
                    }
                )
            except (OSError, subprocess.SubprocessError) as exc:
                checks.append(
                    {"name": "运动模糊依赖", "ok": False, "detail": str(exc)}
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
