from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings, settings


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_load(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    return json.loads(value)


@lru_cache(maxsize=8)
def cached_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Database:
    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.settings.ensure_directories()
        connection = sqlite3.connect(self.settings.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_model_metadata(connection)
        self._seed_demo_data()

    @staticmethod
    def _migrate_model_metadata(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(models)").fetchall()
        }
        columns = {
            "architecture": "TEXT NOT NULL DEFAULT '未记录'",
            "detector_head": "TEXT NOT NULL DEFAULT '未记录'",
            "class_count": "INTEGER NOT NULL DEFAULT 0",
            "training_dataset": "TEXT NOT NULL DEFAULT '未记录'",
            "pretrained_dataset": "TEXT NOT NULL DEFAULT '未记录'",
        }
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(
                    f"ALTER TABLE models ADD COLUMN {name} {definition}"
                )

    def rows(self, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]

    def row(self, query: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
            return dict(row) if row else None

    def execute(self, query: str, parameters: tuple[Any, ...] = ()) -> None:
        with self.connect() as connection:
            connection.execute(query, parameters)

    def _seed_demo_data(self) -> None:
        with self.connect() as connection:
            now = utc_now()
            adapters = [
                (
                    "adapter_basegen",
                    "Z-Image-Turbo 生成器",
                    "GENERATOR",
                    "1.0.0",
                    "EXPERIMENTAL",
                    "conda_external",
                    str(self.settings.basegen_conda_prefix),
                    "read_only",
                    "adapters/basegen_generator.py",
                    1,
                    "REGISTERED",
                    "通过独立 gen 环境调用 BaseGen；输出为未标注的真实生成图像。",
                    json_dump(BASEGEN_SCHEMA),
                ),
                (
                    "adapter_dronedets_yolov8m",
                    "DroneDets · YOLOv8m VisDrone",
                    "DETECTOR",
                    "1.0.0",
                    "EXPERIMENTAL",
                    "conda_external",
                    str(self.settings.dronedets_runtime_prefix),
                    "read_only",
                    "adapters/dronedets_detector.py",
                    1,
                    "REGISTERED",
                    "使用 DroneDets 只读环境执行 YOLOv8m VisDrone 真实目标检测。",
                    json_dump(
                        {
                            "type": "object",
                            "properties": {
                                "catalog_model_id": {
                                    "type": "string",
                                    "const": "yolov8m_visdrone",
                                },
                                "confidence": {
                                    "type": "number",
                                    "default": 0.001,
                                },
                                "nms_iou": {
                                    "type": "number",
                                    "default": 0.7,
                                },
                                "image_size": {
                                    "type": "integer",
                                    "default": 1280,
                                },
                                "max_detections": {
                                    "type": "integer",
                                    "default": 300,
                                },
                            },
                            "license": {
                                "code": "AGPL-3.0",
                                "weights": "see-model-card",
                            },
                        }
                    ),
                ),
                (
                    "adapter_replay",
                    "回放生成器",
                    "GENERATOR",
                    "1.0.0",
                    "CONTRACT_OK",
                    "platform",
                    None,
                    "read_only",
                    "adapters/replay_generator.py",
                    0,
                    "HEALTHY",
                    "固定样例回放，仅用于软件流程验证，不代表生成模型能力。",
                    json_dump(REPLAY_SCHEMA),
                ),
                (
                    "adapter_condition",
                    "可控成像退化",
                    "OPERATOR",
                    "1.0.0",
                    "READY",
                    "platform",
                    None,
                    "read_only",
                    "adapters/condition_operator.py",
                    0,
                    "HEALTHY",
                    "对数据施加运动模糊、雾化、噪声等可重复条件。",
                    json_dump(CONDITION_SCHEMA),
                ),
                (
                    "adapter_reference_detector",
                    "参考检测器",
                    "DETECTOR",
                    "1.0.0",
                    "CONTRACT_OK",
                    "platform",
                    None,
                    "read_only",
                    "adapters/reference_detector.py",
                    0,
                    "HEALTHY",
                    "用于验证评测、时延和图表链路的确定性参考实现。",
                    json_dump({"type": "object", "properties": {}}),
                ),
            ]
            connection.executemany(
                """
                INSERT OR IGNORE INTO adapters
                (id,name,kind,version,maturity,runtime_kind,runtime_prefix,policy,entrypoint,
                 requires_gpu,status,description,parameter_schema,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [item + (now, now) for item in adapters],
            )
            connection.execute(
                """
                UPDATE adapters SET entrypoint='adapters/condition_operator.py',updated_at=?
                WHERE id='adapter_condition' AND entrypoint='adapters/replay_generator.py'
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE adapters SET entrypoint='adapters/replay_generator.py',updated_at=?
                WHERE id='adapter_replay' AND entrypoint!='adapters/replay_generator.py'
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE adapters
                SET runtime_kind='conda_external',runtime_prefix=?,policy='read_only',
                    entrypoint='adapters/basegen_generator.py',requires_gpu=1,updated_at=?
                WHERE id='adapter_basegen'
                """,
                (str(self.settings.basegen_conda_prefix), now),
            )
            connection.execute(
                """
                UPDATE adapters
                SET runtime_kind='conda_external',runtime_prefix=?,policy='read_only',
                    entrypoint='adapters/dronedets_detector.py',requires_gpu=1,updated_at=?
                WHERE id='adapter_dronedets_yolov8m'
                """,
                (str(self.settings.dronedets_runtime_prefix), now),
            )
            weight_override = os.environ.get("DRONEDETS_YOLOV8M_WEIGHT")
            weight_candidates = sorted(
                Path("/mnt/data/cache/huggingface/hub/models--mshamrai--yolov8m-visdrone/snapshots").glob(
                    "*/best.pt"
                )
            )
            weight_path = (
                Path(weight_override).expanduser()
                if weight_override
                else weight_candidates[-1] if weight_candidates else None
            )
            weight_value = str(weight_path) if weight_path and weight_path.is_file() else None
            weight_sha256 = (
                cached_file_sha256(str(weight_path.resolve()))
                if weight_path and weight_path.is_file()
                else None
            )
            models = [
                (
                    "model_dronedets_yolov8m_visdrone",
                    "DroneDets · YOLOv8m VisDrone",
                    "YOLOv8",
                    "YOLOv8m",
                    "DroneDets-33db5f3",
                    "FP16",
                    "adapter_dronedets_yolov8m",
                    weight_value,
                    weight_sha256,
                    0,
                    "EXPERIMENTAL" if weight_value else "UNAVAILABLE",
                ),
                (
                    "model_yolov5s_demo",
                    "YOLOv5s · 流程样例",
                    "YOLOv5",
                    "CSPDarknet",
                    "demo-v1",
                    "FP16",
                    "adapter_reference_detector",
                    None,
                    None,
                    1,
                    "READY",
                ),
                (
                    "model_frcnn_demo",
                    "Faster R-CNN · 流程样例",
                    "Faster R-CNN",
                    "ResNet50",
                    "demo-v1",
                    "FP32",
                    "adapter_reference_detector",
                    None,
                    None,
                    1,
                    "READY",
                ),
                (
                    "model_retinanet_demo",
                    "RetinaNet · 流程样例",
                    "RetinaNet",
                    "ResNet50-FPN",
                    "demo-v1",
                    "FP16",
                    "adapter_reference_detector",
                    None,
                    None,
                    1,
                    "READY",
                ),
            ]
            connection.executemany(
                """
                INSERT OR IGNORE INTO models
                (id,name,family,backbone,version,precision,adapter_id,weight_path,weight_sha256,
                 is_demo,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [item + (now,) for item in models],
            )
            connection.execute(
                """
                UPDATE models SET weight_path=?,weight_sha256=?,status=?
                WHERE id='model_dronedets_yolov8m_visdrone'
                """,
                (
                    weight_value,
                    weight_sha256,
                    "EXPERIMENTAL" if weight_value else "UNAVAILABLE",
                ),
            )
            self._write_demo_artifacts()
            datasets = [
                (
                    "dataset_aerial_clean",
                    "无人机航拍 · 清洁基线",
                    "v1",
                    "REPLAY_FIXTURE",
                    "无人机航拍",
                    "晴朗",
                    json_dump({"motion_blur": 0.0, "fov": 72}),
                    "1920×1080",
                    48,
                    "VERIFIED",
                    1,
                    "demo/aerial-clean",
                ),
                (
                    "dataset_aerial_blur",
                    "无人机航拍 · 运动模糊",
                    "v1",
                    "REAL_TRANSFORMED",
                    "无人机航拍",
                    "晴朗",
                    json_dump({"motion_blur": 0.3, "fov": 72}),
                    "1920×1080",
                    48,
                    "VERIFIED",
                    1,
                    "demo/aerial-blur",
                ),
                (
                    "dataset_urban_fog",
                    "城市驾驶 · 雾天",
                    "v1",
                    "REPLAY_FIXTURE",
                    "城市驾驶",
                    "雾",
                    json_dump({"fog_density": 0.4, "fov": 90}),
                    "1280×720",
                    36,
                    "VERIFIED",
                    1,
                    "demo/urban-fog",
                ),
            ]
            connection.executemany(
                """
                INSERT OR IGNORE INTO datasets
                (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
                 sample_count,annotation_status,frozen,artifact_path,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [item + (now,) for item in datasets],
            )
            if not connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]:
                self._seed_results(connection, now)

    def _write_demo_artifacts(self) -> None:
        styles = {
            "aerial-clean": ("#7cc7d9", "#507d56", "#f7f0d0", 0.0),
            "aerial-blur": ("#7b9fa9", "#607064", "#d8d4bd", 0.3),
            "urban-fog": ("#9aa8b5", "#4c5967", "#cbd2d6", 0.15),
        }
        for folder, palette in styles.items():
            directory = self.settings.artifact_dir / "demo" / folder
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(1, 7):
                path = directory / f"sample-{index}.svg"
                if path.exists():
                    continue
                path.write_text(demo_svg(index, *palette), encoding="utf-8")

    def _seed_results(self, connection: sqlite3.Connection, now: str) -> None:
        datasets = [
            ("dataset_aerial_clean", "无人机航拍", "晴朗", "1920×1080", 0.0),
            ("dataset_aerial_blur", "无人机航拍", "晴朗", "1920×1080", 0.3),
            ("dataset_urban_fog", "城市驾驶", "雾", "1280×720", 0.0),
        ]
        models = [
            ("model_yolov5s_demo", 0.79, 12.8),
            ("model_frcnn_demo", 0.83, 31.4),
            ("model_retinanet_demo", 0.76, 18.7),
        ]
        for dataset_id, domain, weather, resolution, blur in datasets:
            for model_id, base_map, latency in models:
                for seed_index, seed in enumerate((1001, 1002, 1003)):
                    stable = int(hashlib.sha256(f"{dataset_id}{model_id}{seed}".encode()).hexdigest()[:4], 16)
                    noise = ((stable % 21) - 10) / 1000
                    penalty = blur * 0.34 + (0.11 if weather == "雾" else 0)
                    value = max(0.15, min(0.95, base_map - penalty + noise))
                    run_hash = hashlib.sha256(
                        f"{dataset_id}:{model_id}:{seed}".encode()
                    ).hexdigest()[:16]
                    run_id = f"run_seed_{run_hash}"
                    connection.execute(
                        """
                        INSERT INTO runs
                        (id,plan_id,job_id,dataset_id,model_id,seed,status,config,environment_fingerprint,
                         hardware_profile,created_at,finished_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            run_id,
                            None,
                            None,
                            dataset_id,
                            model_id,
                            seed,
                            "SUCCEEDED",
                            json_dump({"batch_size": 1, "warmup": 20, "blur_level": blur}),
                            "reference-demo-environment",
                            json_dump({"device": "流程样例设备", "comparable": True}),
                            now,
                            now,
                        ),
                    )
                    metrics = make_metrics(value, latency + seed_index * 0.15, base_map - value)
                    connection.execute(
                        """
                        INSERT INTO results
                        (id,run_id,map,map50,map75,precision,recall,f1,latency_p50,latency_p95,fps,
                         peak_memory,delta_map,metrics,curves,is_official,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            new_id("result"),
                            run_id,
                            metrics["map"],
                            metrics["map50"],
                            metrics["map75"],
                            metrics["precision"],
                            metrics["recall"],
                            metrics["f1"],
                            metrics["latency_p50"],
                            metrics["latency_p95"],
                            metrics["fps"],
                            metrics["peak_memory"],
                            -max(0.0, base_map - value),
                            json_dump(metrics),
                            json_dump(make_curves(value)),
                            0,
                            now,
                        ),
                    )


def make_metrics(map_value: float, latency: float, degradation: float) -> dict[str, float]:
    precision = min(0.97, map_value + 0.09)
    recall = min(0.95, map_value + 0.04)
    return {
        "map": round(map_value, 4),
        "map50": round(min(0.99, map_value + 0.12), 4),
        "map75": round(max(0.0, map_value - 0.04), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4),
        "latency_p50": round(latency, 2),
        "latency_p95": round(latency * 1.18, 2),
        "fps": round(1000 / latency, 2),
        "peak_memory": round(2100 + latency * 43, 1),
        "degradation": round(degradation, 4),
    }


def make_curves(map_value: float) -> dict[str, list[float]]:
    recall = [round(index / 10, 1) for index in range(11)]
    precision = [round(max(0.05, min(0.99, map_value + 0.18 - index * 0.045)), 3) for index in range(11)]
    return {"recall": recall, "precision": precision}


def demo_svg(index: int, sky: str, ground: str, road: str, blur: float) -> str:
    offset = index * 19
    opacity = max(0.45, 1 - blur)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360">
<defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="{sky}"/><stop offset="1" stop-color="#dfe8e7"/></linearGradient>
<filter id="soft"><feGaussianBlur stdDeviation="{blur * 5:.1f}"/></filter></defs>
<rect width="640" height="360" fill="url(#sky)"/><g filter="url(#soft)" opacity="{opacity}">
<path d="M0 165 L640 130 L640 360 L0 360Z" fill="{ground}"/>
<path d="M{100+offset} 360 L265 150 L365 146 L{510+offset//3} 360Z" fill="{road}"/>
<g fill="#d96c45"><rect x="{210+offset}" y="230" width="55" height="26" rx="5"/><circle cx="{222+offset}" cy="257" r="7" fill="#27313b"/><circle cx="{252+offset}" cy="257" r="7" fill="#27313b"/></g>
<g fill="#e8ca56"><rect x="{415-offset//2}" y="188" width="42" height="22" rx="4"/><circle cx="{425-offset//2}" cy="211" r="6" fill="#27313b"/><circle cx="{448-offset//2}" cy="211" r="6" fill="#27313b"/></g>
<g fill="#e9ecef"><rect x="42" y="142" width="76" height="52"/><rect x="510" y="126" width="88" height="67"/></g></g>
<g fill="none" stroke="#26d9ff" stroke-width="3"><rect x="{202+offset}" y="221" width="72" height="48" rx="3"/><rect x="{406-offset//2}" y="180" width="60" height="42" rx="3"/></g>
<g font-family="sans-serif" font-size="13" fill="#fff"><rect x="{202+offset}" y="201" width="73" height="20" fill="#1689a7"/><text x="{208+offset}" y="216">vehicle 0.91</text></g>
</svg>"""


REPLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_style": {"type": "string", "enum": ["aerial", "satellite", "urban"]},
        "sample_count": {"type": "integer", "minimum": 1, "maximum": 1000},
    },
}

CONDITION_SCHEMA = {
    "type": "object",
    "properties": {
        "motion_blur": {"type": "number", "minimum": 0, "maximum": 1},
        "fog_density": {"type": "number", "minimum": 0, "maximum": 1},
        "noise_level": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

BASEGEN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "integer", "minimum": 1, "default": 9},
        "guidance_scale": {"type": "number", "default": 0},
        "device_policy": {
            "type": "string",
            "enum": ["cuda", "cpu-offload"],
            "default": "cuda",
        },
        "model_path": {
            "type": "string",
            "default": "Tongyi-MAI/Z-Image-Turbo",
        },
        "local_files_only": {"type": "boolean", "default": False},
    },
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS adapters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    version TEXT NOT NULL,
    maturity TEXT NOT NULL,
    runtime_kind TEXT NOT NULL,
    runtime_prefix TEXT,
    policy TEXT NOT NULL DEFAULT 'read_only',
    entrypoint TEXT,
    requires_gpu INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    parameter_schema TEXT NOT NULL DEFAULT '{}',
    environment_fingerprint TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    source_type TEXT NOT NULL,
    scene_domain TEXT NOT NULL,
    weather TEXT NOT NULL,
    sensor_conditions TEXT NOT NULL DEFAULT '{}',
    resolution TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    annotation_status TEXT NOT NULL,
    frozen INTEGER NOT NULL DEFAULT 0,
    artifact_path TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_annotation_schemas (
    dataset_id TEXT PRIMARY KEY REFERENCES datasets(id) ON DELETE CASCADE,
    categories TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_annotations (
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    sample_name TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    boxes TEXT NOT NULL DEFAULT '[]',
    completed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, sample_name)
);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    architecture TEXT NOT NULL DEFAULT '未记录',
    backbone TEXT NOT NULL,
    detector_head TEXT NOT NULL DEFAULT '未记录',
    class_count INTEGER NOT NULL DEFAULT 0,
    training_dataset TEXT NOT NULL DEFAULT '未记录',
    pretrained_dataset TEXT NOT NULL DEFAULT '未记录',
    version TEXT NOT NULL,
    precision TEXT NOT NULL,
    adapter_id TEXT NOT NULL REFERENCES adapters(id),
    weight_path TEXT,
    weight_sha256 TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    result TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS evaluation_plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_ids TEXT NOT NULL,
    model_ids TEXT NOT NULL,
    seeds TEXT NOT NULL,
    blur_levels TEXT NOT NULL,
    protocol TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    plan_id TEXT REFERENCES evaluation_plans(id),
    job_id TEXT REFERENCES jobs(id),
    dataset_id TEXT NOT NULL REFERENCES datasets(id),
    model_id TEXT NOT NULL REFERENCES models(id),
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    config TEXT NOT NULL,
    environment_fingerprint TEXT,
    hardware_profile TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
    map REAL NOT NULL,
    map50 REAL NOT NULL,
    map75 REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    latency_p50 REAL NOT NULL,
    latency_p95 REAL NOT NULL,
    fps REAL NOT NULL,
    peak_memory REAL NOT NULL,
    delta_map REAL,
    metrics TEXT NOT NULL,
    curves TEXT NOT NULL,
    is_official INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_sample_annotations_progress
ON sample_annotations(dataset_id, completed);
CREATE INDEX IF NOT EXISTS idx_runs_dataset_model ON runs(dataset_id, model_id);
CREATE INDEX IF NOT EXISTS idx_results_map ON results(map DESC);
"""


db = Database()
