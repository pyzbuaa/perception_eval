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

from app.config import settings
from app.db import Database, db, json_dump, json_load, new_id, utc_now


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


class DatasetArtifactError(ValueError):
    pass


class BaseGenCatalogError(ValueError):
    pass


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
            }
            for path in page
        ],
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
            manifest.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
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
                {"name": "入口脚本", "ok": entrypoint.is_file(), "detail": str(entrypoint)},
                {"name": "只读策略", "ok": adapter["policy"] == "read_only", "detail": adapter["policy"]},
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
