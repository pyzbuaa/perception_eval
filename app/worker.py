from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from app.annotation_formats import (
    convert_visdrone_to_coco,
    is_visdrone_label_directory,
)
from app.category_templates import normalize_categories, normalize_category_name
from app.command_protocol import CommandTemplateError, render_command
from app.config import Settings, settings
from app.db import Database, db, json_dump, json_load, make_curves, make_metrics, new_id, utc_now
from app.detection_metrics import evaluate_coco_predictions
from app.services import (
    DatasetImportError,
    _dataset_image_files,
    _match_dataset_sample_name,
    _dataset_sample_name,
    category_compatibility,
    resolve_local_dataset_import,
    summarize_image_resolutions,
    validate_evaluation_categories,
)


class JobAgent:
    def __init__(self, database: Database = db, app_settings: Settings = settings):
        self.db = database
        self.settings = app_settings
        self.stop_event = threading.Event()

    def run_forever(self, poll_interval: float = 0.5) -> None:
        while not self.stop_event.is_set():
            if not self.process_one():
                self.stop_event.wait(poll_interval)

    def stop(self) -> None:
        self.stop_event.set()

    def process_one(self) -> bool:
        job = self._claim_next_job()
        if not job:
            return False
        try:
            if job["type"] == "ACQUISITION":
                result = self._run_acquisition(job)
            elif job["type"] == "DATASET_IMPORT":
                result = self._run_import(job)
            elif job["type"] == "EVALUATION":
                result = self._run_evaluation(job)
            elif job["type"] == "AUTO_ANNOTATION":
                result = self._run_auto_annotation(job)
            else:
                raise ValueError(f"未知任务类型: {job['type']}")
            self._finish(job["id"], result)
        except JobCancelled:
            self.db.execute(
                "UPDATE jobs SET status='CANCELLED',stage='已取消',finished_at=? WHERE id=?",
                (utc_now(), job["id"]),
            )
        except Exception as exc:  # Worker must persist adapter failures instead of exiting.
            self.db.execute(
                "UPDATE jobs SET status='FAILED',stage='执行失败',error=?,finished_at=? WHERE id=?",
                (f"{type(exc).__name__}: {exc}", utc_now(), job["id"]),
            )
        return True

    def _claim_next_job(self) -> dict[str, Any] | None:
        with self.db.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE jobs SET status='RUNNING',stage='启动任务',started_at=? WHERE id=?",
                (utc_now(), row["id"]),
            )
            job = dict(row)
            job["payload"] = json_load(job["payload"], {})
            return job

    def _check_cancelled(self, job_id: str) -> None:
        row = self.db.row("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,))
        if row and row["cancel_requested"]:
            raise JobCancelled

    def _progress(self, job_id: str, progress: float, stage: str) -> None:
        self._check_cancelled(job_id)
        self.db.execute(
            "UPDATE jobs SET progress=?,stage=? WHERE id=?",
            (max(0, min(progress, 100)), stage, job_id),
        )

    def _finish(self, job_id: str, result: dict[str, Any]) -> None:
        self.db.execute(
            """
            UPDATE jobs SET status='SUCCEEDED',progress=100,stage='已完成',result=?,finished_at=?
            WHERE id=?
            """,
            (json_dump(result), utc_now(), job_id),
        )

    def _run_acquisition(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        try:
            dataset_categories = normalize_categories(
                payload.get("categories", []), include_color=True
            )
        except ValueError as exc:
            raise ValueError(f"数据集类别无效: {exc}") from exc
        category_template = str(payload.get("category_template") or "custom")
        job_dir = self.settings.task_dir / job["id"]
        artifact_dir = self.settings.artifact_dir / "generated" / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = job_dir / "request.json"
        result_path = job_dir / "result.json"
        request = {
            "protocol_version": "1.0",
            "job_id": job["id"],
            "adapter_id": payload["adapter_id"],
            "seed": payload.get("seeds", [1001])[0],
            "seeds": payload.get("seeds", [1001]),
            "sample_count": payload.get("sample_count", 12),
            "conditions": payload.get("conditions", {}),
            "model_parameters": payload.get("model_parameters", {}),
            "source_type": payload.get("source_type", "GENERATIVE"),
            "output_directory": str(artifact_dir),
        }
        condition_adapter_ids = {
            "adapter_condition",
            "adapter_day_to_night",
            "adapter_motion_blur",
            "adapter_warpi2i_fog",
            "adapter_warpi2i_day_to_night",
        }
        driving_condition_adapter_ids = {
            "adapter_warpi2i_fog",
            "adapter_warpi2i_day_to_night",
        }
        is_condition = payload["adapter_id"] in condition_adapter_ids
        input_dataset: dict[str, Any] | None = None
        source_annotation_directory: Path | None = None
        if is_condition:
            input_dataset_id = payload.get("input_dataset_id")
            if not input_dataset_id:
                raise ValueError("条件退化任务必须选择一个输入数据集")
            input_dataset = self.db.row("SELECT * FROM datasets WHERE id=?", (input_dataset_id,))
            if not input_dataset or not input_dataset.get("artifact_path"):
                raise ValueError("输入数据集不存在或没有 Artifact")
            scene_domain = input_dataset.get("scene_domain")
            if payload["adapter_id"] in driving_condition_adapter_ids:
                if scene_domain not in {
                    "城市驾驶",
                    "自动驾驶",
                    "autonomous-driving",
                }:
                    raise ValueError("WarpI2I 非理想条件生成仅支持自动驾驶场景数据集")
            elif scene_domain not in {
                "无人机航拍",
                "低空无人机",
                "low-altitude-uav",
            }:
                raise ValueError("该非理想条件生成模型仅支持无人机航拍域数据集")
            category_row = self.db.row(
                "SELECT categories FROM dataset_annotation_schemas WHERE dataset_id=?",
                (input_dataset_id,),
            )
            if not category_row:
                raise ValueError("输入数据集尚未配置类别")
            dataset_categories = normalize_categories(
                json_load(category_row["categories"], []), include_color=True
            )
            category_template = str(
                input_dataset.get("category_template") or "custom"
            )
            input_directory = self.settings.artifact_dir / input_dataset["artifact_path"]
            input_images = [
                str(path)
                for path in _dataset_image_files(input_directory)
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if not input_images:
                raise ValueError("输入数据集没有可处理的 PNG/JPEG/WebP 图像；SVG 流程样例不用于像素退化")
            source_annotation_directory = input_directory / "annotations"
            request["input_images"] = input_images
            request["input_directory"] = str(input_directory)
            request["input_dataset_id"] = input_dataset_id
            request["sample_count"] = len(input_images)
            request["has_source_annotations"] = (
                source_annotation_directory / "instances.json"
            ).is_file()
            resolution = summarize_image_resolutions(
                Path(value) for value in input_images
            ) or input_dataset.get("resolution") or "无法读取"
            if payload["adapter_id"] == "adapter_condition":
                try:
                    fog_strength = float(
                        payload.get("model_parameters", {}).get("fog_strength", 1.0)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("气雾强度必须是 0 到 1 之间的数值") from exc
                if not 0 <= fog_strength <= 1:
                    raise ValueError("气雾强度必须位于 0 到 1 之间")
                request["conditions"] = {
                    "scene": {"domain": "无人机航拍", "weather": "雾"},
                    "sensor": {
                        "resolution": resolution,
                        "source_dataset_id": input_dataset_id,
                        "degradation": "DiffusionDegrade UAV Fog",
                        "condition_label": "无人机气雾",
                        "fog_model": "DiffusionDegrade · 无人机气雾",
                        "fog_strength": fog_strength,
                    },
                }
                request["model_parameters"] = {
                    "effect": "fog",
                    "domain": "uav_aerial",
                    "image_prep": "resize_512x512",
                    "precision": "FP16",
                    "fog_strength": fog_strength,
                    "checkpoint": "uav_fog_content15_model_2501",
                }
            elif payload["adapter_id"] == "adapter_day_to_night":
                request["conditions"] = {
                    "scene": {"domain": "无人机航拍", "weather": "弱光"},
                    "sensor": {
                        "resolution": resolution,
                        "source_dataset_id": input_dataset_id,
                        "degradation": "DiffusionDegrade UAV Low-Light",
                        "day_to_night": True,
                        "day_to_night_model": "CycleGAN-Turbo Sichuan",
                        "condition_label": "无人机弱光",
                        "day_to_night_checkpoint": (
                            "uav_daynight_sichuan_3125_model_3125"
                        ),
                        "source_time_of_day": "白天",
                        "target_time_of_day": "夜间",
                        "day_to_night_image_prep": "resize_640x640",
                        "day_to_night_model_size": 640,
                    },
                }
                request["model_parameters"] = {
                    "effect": "day_to_night",
                    "domain": "uav_aerial",
                    "direction": "a2b",
                    "image_prep": "resize_640x640",
                    "model_size": 640,
                    "precision": "FP16",
                    "checkpoint": "uav_daynight_sichuan_3125_model_3125",
                }
            elif payload["adapter_id"] == "adapter_warpi2i_fog":
                request["conditions"] = {
                    "scene": {"domain": "城市驾驶", "weather": "雾"},
                    "sensor": {
                        "resolution": resolution,
                        "source_dataset_id": input_dataset_id,
                        "degradation": "WarpI2I Driving Fog",
                        "condition_label": "自动驾驶气雾",
                        "fog_model": "WarpI2I · 自动驾驶气雾",
                        "fog_method": "paired",
                        "fog_checkpoint": "foggy_1.pkl",
                    },
                }
                request["model_parameters"] = {
                    "effect": "fog",
                    "domain": "autonomous_driving",
                    "method": "paired",
                    "image_prep": "multiple_of_8",
                    "precision": "FP16",
                    "checkpoint": "foggy_1.pkl",
                }
            elif payload["adapter_id"] == "adapter_warpi2i_day_to_night":
                request["conditions"] = {
                    "scene": {"domain": "城市驾驶", "weather": "弱光"},
                    "sensor": {
                        "resolution": resolution,
                        "source_dataset_id": input_dataset_id,
                        "degradation": "WarpI2I Driving Day-to-Night",
                        "day_to_night": True,
                        "condition_label": "自动驾驶弱光",
                        "day_to_night_model": "WarpI2I · 自动驾驶弱光",
                        "day_to_night_method": "unpaired",
                        "day_to_night_checkpoint": "BDD100K_day2night.pkl",
                        "source_time_of_day": "白天",
                        "target_time_of_day": "夜间",
                    },
                }
                request["model_parameters"] = {
                    "effect": "day_to_night",
                    "domain": "autonomous_driving",
                    "method": "unpaired",
                    "direction": "a2b",
                    "image_prep": "resize_512x512",
                    "precision": "FP16",
                    "checkpoint": "BDD100K_day2night.pkl",
                }
            elif payload["adapter_id"] == "adapter_motion_blur":
                motion_parameters = payload.get("model_parameters", {})
                condition_value = str(
                    motion_parameters.get("condition_directory") or ""
                ).strip()
                condition_directory: Path | None = None
                if condition_value:
                    condition_directory = Path(condition_value).expanduser().resolve()
                    dataset_root = self.settings.dataset_library_root.expanduser().resolve()
                    if (
                        not condition_directory.is_relative_to(dataset_root)
                        or not condition_directory.is_dir()
                    ):
                        raise ValueError(
                            f"运动条件目录必须位于 {dataset_root} 内且真实存在"
                        )
                motion = str(motion_parameters.get("motion", "forward"))
                allowed_motions = {
                    "forward", "backward", "fly-left", "fly-right",
                    "ascend", "descend", "yaw-left", "yaw-right",
                    "tilt-up", "tilt-down", "tilt-left", "tilt-right",
                    "vibration",
                }
                if motion not in allowed_motions:
                    raise ValueError(f"不支持的无人机运动类型: {motion}")
                try:
                    motion_strength = float(motion_parameters.get("strength", 0.14))
                except (TypeError, ValueError) as exc:
                    raise ValueError("运动模糊强度必须是数值") from exc
                if not 0.01 <= motion_strength <= 0.35:
                    raise ValueError("运动模糊强度必须位于 0.01 到 0.35 之间")
                request["conditions"] = {
                    "scene": {
                        "domain": "无人机航拍",
                        "weather": input_dataset.get("weather") or "未记录",
                    },
                    "sensor": {
                        "resolution": resolution,
                        "source_dataset_id": input_dataset_id,
                        "degradation": "ID-Blau UAV Motion Blur",
                        "condition_label": "无人机运动模糊",
                        "motion_blur": True,
                        "motion_blur_model": "ID-Blau",
                        "motion": "condition-files" if condition_directory else motion,
                        "motion_blur_strength": motion_strength,
                        "motion_blur_sample_timesteps": 20,
                        **(
                            {
                                "motion_condition_directory": str(condition_directory),
                                "motion_condition_matching": "filename",
                                "motion_condition_fallback": "random-preset",
                            }
                            if condition_directory
                            else {}
                        ),
                    },
                }
                request["model_parameters"] = {
                    "effect": "motion_blur",
                    "domain": "uav_aerial",
                    "motion": motion,
                    "strength": motion_strength,
                    "sample_timesteps": 20,
                    "precision": "FP32",
                    "checkpoint": "ID_Blau.pth",
                    **(
                        {
                            "condition_directory": str(condition_directory),
                            "condition_matching": "filename",
                            "fallback_motion": "random-preset",
                        }
                        if condition_directory
                        else {}
                    ),
                }
        request_path.write_text(json_dump(request), encoding="utf-8")
        self._progress(job["id"], 1, "校验适配器协议")
        adapter = self.db.row("SELECT * FROM adapters WHERE id=?", (payload["adapter_id"],))
        if not adapter:
            raise ValueError("适配器不存在")
        entrypoint_value = adapter.get("entrypoint")
        if not entrypoint_value:
            raise ValueError("适配器没有配置入口脚本")
        entrypoint = Path(entrypoint_value)
        if not entrypoint.is_absolute():
            entrypoint = self.settings.root_dir / entrypoint
        if not entrypoint.is_file():
            raise FileNotFoundError(f"适配器入口不存在: {entrypoint}")
        environment = os.environ.copy()
        environment.update(self._isolated_cache_environment(job_dir))
        environment.update(
            {
                "BASEGEN_ROOT": str(self.settings.basegen_root),
                "DIFFUSION_DEGRADE_ROOT": str(
                    self.settings.diffusion_degrade_root
                ),
                "DIFFUSION_DEGRADE_UAV_FOG_CHECKPOINT": str(
                    self.settings.diffusion_degrade_checkpoint
                ),
                "DIFFUSION_DEGRADE_UAV_DAY_TO_NIGHT_CHECKPOINT": str(
                    self.settings.diffusion_degrade_day_to_night_checkpoint
                ),
                "DIFFUSION_BLUR_ROOT": str(self.settings.diffusion_blur_root),
                "DIFFUSION_BLUR_CHECKPOINT": str(
                    self.settings.diffusion_blur_checkpoint
                ),
                "WARPI2I_ROOT": str(self.settings.warpi2i_root),
                "WARPI2I_DRIVING_FOG_CHECKPOINT": str(
                    self.settings.warpi2i_driving_fog_checkpoint
                ),
                "WARPI2I_DRIVING_DAY_TO_NIGHT_CHECKPOINT": str(
                    self.settings.warpi2i_driving_day_to_night_checkpoint
                ),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if payload["adapter_id"] == "adapter_basegen":
            environment.update(
                {
                    "HF_HOME": str(self.settings.basegen_hf_home),
                    "HF_HUB_CACHE": str(self.settings.basegen_hf_home / "hub"),
                }
            )
        if payload["adapter_id"] in {
            "adapter_condition",
            "adapter_day_to_night",
            "adapter_warpi2i_fog",
            "adapter_warpi2i_day_to_night",
        }:
            environment.update(
                {
                    "HF_HOME": str(self.settings.diffusion_degrade_hf_home),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
        command = self._adapter_command(
            adapter, entrypoint, request_path, result_path
        )
        self._progress(job["id"], 1, "启动生成适配器")
        returncode, log_tail = self._run_adapter_process(
            job["id"], command, job_dir, environment
        )
        if returncode != 0:
            raise RuntimeError(
                f"适配器退出码 {returncode}: {log_tail[-500:]}"
            )
        self._progress(job["id"], 99, "验证输出文件与元数据")
        if not result_path.is_file():
            raise FileNotFoundError("适配器没有生成 result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol_version") != "1.0":
            raise ValueError("适配器结果协议版本不受支持")
        if result.get("job_id") != job["id"]:
            raise ValueError("适配器结果 job_id 与请求不一致")
        if result.get("status") != "succeeded":
            raise ValueError(f"适配器返回失败状态: {result.get('status')}")
        samples = result.get("samples", [])
        if len(samples) != request["sample_count"]:
            raise ValueError("适配器输出样本数与请求不一致")
        for sample in samples:
            image_relative = Path(sample["image_path"])
            if image_relative.is_absolute() or ".." in image_relative.parts:
                raise ValueError(f"生成文件路径必须位于 Artifact 目录内: {image_relative}")
            path = artifact_dir / image_relative
            if not path.is_file():
                raise FileNotFoundError(f"缺少生成文件: {sample['image_path']}")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected != sample["sha256"]:
                raise ValueError(f"文件摘要不一致: {sample['image_path']}")
            if sample.get("metadata_path"):
                metadata_relative = Path(sample["metadata_path"])
                if metadata_relative.is_absolute() or ".." in metadata_relative.parts:
                    raise ValueError(
                        f"元数据路径必须位于 Artifact 目录内: {metadata_relative}"
                    )
                if not (artifact_dir / metadata_relative).is_file():
                    raise FileNotFoundError(
                        f"缺少生成元数据: {sample['metadata_path']}"
                    )
        if is_condition:
            expected_names = {
                _dataset_sample_name(input_directory, Path(value))
                for value in request["input_images"]
            }
            output_names = {str(sample["image_path"]) for sample in samples}
            if output_names != expected_names:
                raise ValueError("退化输出文件名与输入数据集不一致，无法继承标注")
            if request["has_source_annotations"]:
                assert source_annotation_directory is not None
                shutil.copytree(
                    source_annotation_directory,
                    artifact_dir / "annotations",
                )
        dataset_id = new_id("dataset")
        dataset_conditions = (
            request["conditions"]
            if is_condition
            else payload.get("conditions", {})
        )
        sensor = dataset_conditions.get("sensor", {})
        scene = dataset_conditions.get("scene", {})
        if payload["adapter_id"] == "adapter_motion_blur":
            runtime = result.get("runtime", {})
            if "matched_conditions" in runtime:
                sensor["motion_condition_matched"] = int(
                    runtime["matched_conditions"]
                )
            if "fallback_conditions" in runtime:
                sensor["motion_condition_fallback_count"] = int(
                    runtime["fallback_conditions"]
                )
        resolution = summarize_image_resolutions(
            artifact_dir / sample["image_path"] for sample in samples
        )
        if not resolution:
            resolution = sensor.get("resolution", "无法读取")
            if isinstance(resolution, list):
                resolution = "×".join(str(value) for value in resolution)
        annotation_status = "CANDIDATE" if result.get("has_candidate_annotations") else "UNLABELED"
        if is_condition:
            annotation_status = (
                "CANDIDATE" if request["has_source_annotations"] else "UNLABELED"
            )
        relative = artifact_dir.relative_to(self.settings.artifact_dir).as_posix()
        now = utc_now()
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets
                (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
                 sample_count,annotation_status,frozen,artifact_path,category_template,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    dataset_id,
                    payload["name"],
                    "v1",
                    payload.get("source_type", "GENERATIVE"),
                    (
                        "自动驾驶"
                        if scene.get("domain") == "autonomous-driving"
                        else scene.get(
                            "domain_label", scene.get("domain", "无人机航拍")
                        )
                    ),
                    scene.get("weather", "晴朗"),
                    json_dump(sensor),
                    resolution,
                    len(samples),
                    annotation_status,
                    0,
                    relative,
                    category_template,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO dataset_annotation_schemas
                (dataset_id,categories,updated_at) VALUES (?,?,?)
                """,
                (dataset_id, json_dump(dataset_categories), now),
            )
        self._progress(job["id"], 99, "创建数据集草稿")
        return {"dataset_id": dataset_id, "samples": len(samples), "annotation_status": annotation_status}

    def _adapter_command(
        self,
        adapter: dict[str, Any],
        entrypoint: Path,
        request_path: Path,
        result_path: Path,
    ) -> list[str]:
        runtime_kind = adapter["runtime_kind"]
        if runtime_kind == "platform":
            python = Path(sys.executable)
        elif runtime_kind in {"conda_external", "conda_clone"}:
            prefix = Path(adapter.get("runtime_prefix") or "")
            python = prefix / "bin" / "python"
            if not python.is_file():
                raise FileNotFoundError(f"外部环境 Python 不存在: {python}")
        else:
            raise ValueError(f"尚不支持的适配器运行方式: {runtime_kind}")
        return [
            str(python),
            "-B",
            str(entrypoint),
            "run",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]

    def _run_adapter_process(
        self,
        job_id: str,
        command: list[str],
        cwd: Path,
        environment: dict[str, str],
    ) -> tuple[int, str]:
        output_queue: queue.Queue[str | None] = queue.Queue()
        tail: deque[str] = deque(maxlen=80)
        log_path = self.settings.log_dir / f"{job_id}.log"
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        started = time.monotonic()
        output_finished = False
        try:
            with log_path.open("w", encoding="utf-8") as log:
                while process.poll() is None or not output_finished:
                    try:
                        line = output_queue.get(timeout=0.5)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        output_finished = True
                    elif line:
                        log.write(line)
                        log.flush()
                        tail.append(line)
                        self._handle_adapter_progress(job_id, line)
                    self._check_cancelled(job_id)
                    if time.monotonic() - started > self.settings.adapter_timeout_seconds:
                        raise TimeoutError(
                            f"适配器执行超过 {self.settings.adapter_timeout_seconds} 秒"
                        )
        except BaseException:
            self._terminate_process(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            reader.join(timeout=1)
        return process.wait(), "".join(tail)

    def _handle_adapter_progress(self, job_id: str, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if event.get("type") == "stage":
            stage = str(event.get("stage", "")).strip()
            if stage:
                self._progress(
                    job_id,
                    float(event.get("progress", 1)),
                    stage,
                )
            return
        if event.get("type") != "progress":
            return
        current = int(event.get("current", 0))
        total = max(1, int(event.get("total", 1)))
        progress = max(1, min(99, 100 * current / total))
        stage = str(event.get("stage", "生成图像"))
        self._progress(job_id, progress, f"{stage} {current}/{total}")

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def _run_import(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        try:
            dataset_categories = normalize_categories(
                payload.get("categories", []), include_color=True
            )
        except ValueError as exc:
            raise ValueError(f"数据集类别无效: {exc}") from exc
        staged_upload_root = payload.get("staged_upload_root")
        if staged_upload_root:
            source = Path(payload["directory"]).expanduser().resolve()
            resolved_annotation = (
                Path(payload["annotation_path"]).expanduser().resolve()
                if payload.get("annotation_path")
                else None
            )
        else:
            try:
                source, resolved_annotation = resolve_local_dataset_import(
                    payload,
                    self.settings,
                )
            except DatasetImportError as exc:
                raise ValueError(str(exc)) from exc
        candidates = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".svg"}
        )
        if not candidates:
            raise ValueError("目录中没有支持的图像")
        resolution = summarize_image_resolutions(candidates) or "无法读取"
        target = self.settings.artifact_dir / "imports" / job["id"]
        target.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(candidates):
            self._check_cancelled(job["id"])
            relative = Path(path.name) if staged_upload_root else path.relative_to(source)
            image_target = target / relative
            if image_target.exists():
                raise ValueError(
                    f"图像目标路径重复: {relative.as_posix()}"
                )
            image_target.parent.mkdir(parents=True, exist_ok=True)
            if staged_upload_root:
                shutil.copy2(path, image_target)
                action = "保存上传图像"
            else:
                image_target.symlink_to(path.resolve())
                action = "引用图像"
            self._progress(
                job["id"],
                10 + 70 * (index + 1) / len(candidates),
                f"{action} {index + 1}/{len(candidates)}",
            )
        annotation = payload.get("annotation_path")
        annotation_status = "UNLABELED"
        annotation_count = 0
        if annotation:
            assert resolved_annotation is not None
            annotation_path = resolved_annotation
            if annotation_path.is_file():
                normalized_annotation: dict[str, Any] | None = None
                if annotation_path.suffix.lower() == ".json":
                    annotation_payload = json.loads(
                        annotation_path.read_text(encoding="utf-8")
                    )
                    declared = {
                        (item["id"], normalize_category_name(item["name"]))
                        for item in normalize_categories(dataset_categories)
                    }
                    imported = {
                        (item["id"], normalize_category_name(item["name"]))
                        for item in normalize_categories(
                            annotation_payload.get("categories", [])
                        )
                    }
                    if declared != imported:
                        raise ValueError(
                            "COCO 标注中的类别 ID/名称与所选类别模板不一致"
                        )
                    artifact_names = {
                        path.relative_to(target).as_posix()
                        for path in _dataset_image_files(target)
                    }
                    matched_names = set()
                    for image in annotation_payload.get("images", []):
                        matched = _match_dataset_sample_name(
                            str(image.get("file_name", "")),
                            artifact_names,
                        )
                        if not matched:
                            raise ValueError(
                                "COCO 标注图片无法与导入目录唯一匹配；"
                                "存在重名图片时 file_name 必须包含相对路径"
                            )
                        if matched in matched_names:
                            raise ValueError("COCO 标注中存在重复的图片路径")
                        matched_names.add(matched)
                        image["file_name"] = matched
                    normalized_annotation = annotation_payload
                    annotation_target = target / "annotations" / "instances.json"
                    annotation_target.parent.mkdir(parents=True, exist_ok=True)
                else:
                    annotation_target = target / annotation_path.name
                if normalized_annotation is None:
                    shutil.copy2(annotation_path, annotation_target)
                else:
                    annotation_target.write_text(
                        json.dumps(
                            normalized_annotation,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                annotation_status = "CANDIDATE"
                annotation_count = 1
            elif annotation_path.is_dir():
                annotation_format = str(
                    payload.get("annotation_format", "YOLO")
                ).lower()
                annotation_target = (
                    target / "annotations" / annotation_format
                )
                supported = {".txt", ".yaml", ".yml", ".names"}
                annotation_files = sorted(
                    path
                    for path in annotation_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in supported
                )
                for path in annotation_files:
                    relative = path.relative_to(annotation_path)
                    output = annotation_target / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, output)
                if annotation_files:
                    annotation_status = "CANDIDATE"
                    annotation_count = len(annotation_files)
                    if annotation_format == "visdrone":
                        convert_visdrone_to_coco(
                            target,
                            annotation_target,
                            target / "annotations" / "instances.json",
                            payload["name"],
                        )
        dataset_id = new_id("dataset")
        now = utc_now()
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets
                (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
                 sample_count,annotation_status,frozen,artifact_path,source_path,category_template,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    dataset_id,
                    payload["name"],
                    "v1",
                    "REAL",
                    payload.get("scene_domain", "未分类"),
                    "未记录",
                    json_dump(
                        {
                            "recorded_condition": payload.get(
                                "nonideal_condition", "无"
                            )
                        }
                    ),
                    resolution,
                    len(candidates),
                    annotation_status,
                    0,
                    target.relative_to(self.settings.artifact_dir).as_posix(),
                    None if staged_upload_root else str(source),
                    str(payload.get("category_template") or "custom"),
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO dataset_annotation_schemas
                (dataset_id,categories,updated_at) VALUES (?,?,?)
                """,
                (dataset_id, json_dump(dataset_categories), now),
            )
        if staged_upload_root:
            allowed_root = (
                self.settings.task_dir / "import_uploads"
            ).resolve()
            staging = Path(staged_upload_root).resolve()
            if (
                staging != allowed_root
                and staging.is_relative_to(allowed_root)
                and source.is_relative_to(staging)
                and staging.is_dir()
            ):
                shutil.rmtree(staging)
        return {
            "dataset_id": dataset_id,
            "samples": len(candidates),
            "annotation_status": annotation_status,
            "annotation_files": annotation_count,
        }

    def _run_auto_annotation(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        dataset = self.db.row(
            "SELECT * FROM datasets WHERE id=?", (payload["dataset_id"],)
        )
        model = self.db.row(
            "SELECT * FROM models WHERE id=?", (payload["model_id"],)
        )
        if not dataset or not model:
            raise ValueError("数据集或检测模型不存在")
        if dataset["frozen"]:
            raise ValueError("冻结数据集不能执行自动标注")
        if dataset["annotation_status"] not in {"UNLABELED", "ANNOTATING"}:
            raise ValueError("自动标注仅支持未标注或正在标注的数据集")
        if model["is_demo"] or model["status"] == "UNAVAILABLE":
            raise ValueError("自动标注必须使用可用的真实检测模型")

        adapter = self.db.row(
            "SELECT * FROM adapters WHERE id=?", (model["adapter_id"],)
        )
        if not adapter or adapter["kind"] != "DETECTOR":
            raise ValueError(f"模型没有可用的检测适配器: {model['name']}")
        if not model.get("weight_path") or not Path(model["weight_path"]).is_file():
            raise FileNotFoundError(f"模型权重不存在: {model.get('weight_path')}")

        compatibility = category_compatibility(dataset["id"], model["id"], self.db)
        if not compatibility["compatible"]:
            raise ValueError(compatibility["reason"])
        model_to_dataset = {
            int(model_id): int(dataset_id)
            for model_id, dataset_id in compatibility["model_to_dataset"].items()
        }
        model_categories = normalize_categories(json_load(model["categories"], []))

        artifact_root = self.settings.artifact_dir.resolve()
        dataset_directory = (
            artifact_root / str(dataset.get("artifact_path") or "")
        ).resolve()
        if (
            dataset_directory == artifact_root
            or not dataset_directory.is_relative_to(artifact_root)
            or not dataset_directory.is_dir()
        ):
            raise ValueError(f"数据集 Artifact 不可用: {dataset['name']}")
        raster_files = [
            path
            for path in _dataset_image_files(dataset_directory)
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not raster_files:
            raise ValueError("数据集没有可供检测模型处理的栅格图片")
        existing_names = {
            row["sample_name"]
            for row in self.db.rows(
                "SELECT sample_name FROM sample_annotations WHERE dataset_id=?",
                (dataset["id"],),
            )
        }
        eligible_files = [
            path
            for path in raster_files
            if _dataset_sample_name(dataset_directory, path) not in existing_names
        ]
        if not eligible_files:
            raise ValueError("数据集中的图片都已有标注，未执行自动覆盖")

        images: list[dict[str, Any]] = []
        image_dimensions: dict[int, tuple[str, int, int]] = {}
        for image_id, path in enumerate(eligible_files, start=1):
            sample_name = _dataset_sample_name(dataset_directory, path)
            try:
                with Image.open(path) as opened:
                    width, height = opened.size
            except OSError as exc:
                raise ValueError(f"无法读取图片: {sample_name}") from exc
            images.append(
                {
                    "id": image_id,
                    "file_name": sample_name,
                    "width": width,
                    "height": height,
                }
            )
            image_dimensions[image_id] = (sample_name, width, height)

        job_directory = self.settings.task_dir / job["id"]
        output_directory = self.settings.artifact_dir / "auto-annotations" / job["id"]
        job_directory.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = job_directory / "images.json"
        manifest_path.write_text(
            json_dump(
                {
                    "info": {"description": f"{dataset['name']} 自动标注图片清单"},
                    "images": images,
                    "annotations": [],
                    "categories": model_categories,
                }
            ),
            encoding="utf-8",
        )

        schema = json_load(adapter.get("parameter_schema"), {})
        properties = schema.get("properties", {})

        def adapter_parameter(name: str, fallback: Any) -> Any:
            specification = properties.get(name, {})
            return specification.get("const", specification.get("default", fallback))

        def task_parameter(name: str, fallback: Any) -> Any:
            default = adapter_parameter(name, fallback)
            return payload.get(name, default) if name in properties else default

        confidence = float(task_parameter("confidence", 0.25))
        nms_iou = float(task_parameter("nms_iou", 0.7))
        image_size = int(task_parameter("image_size", 1280))
        input_height = int(task_parameter("input_height", image_size))
        input_width = int(task_parameter("input_width", image_size))
        max_detections = int(task_parameter("max_detections", 300))
        batch_size = int(task_parameter("batch_size", 1))
        warmup = int(task_parameter("warmup", 0))
        precision = str(task_parameter("precision", model["precision"]))

        execution = schema.get("execution", {})
        command_mode = execution.get("mode") == "command"
        predictions_filename = str(
            execution.get("predictions_filename", "predictions.json")
        )
        prediction_relative = Path(predictions_filename)
        if prediction_relative.is_absolute() or ".." in prediction_relative.parts:
            raise ValueError("检测结果路径必须位于自动标注输出目录内")
        predictions_path = (output_directory / prediction_relative).resolve()
        request_path = job_directory / "request.json"
        result_path = job_directory / "result.json"
        run_id = new_id("annotation")
        request = {
            "protocol_version": "1.0",
            "job_id": job["id"],
            "run_id": run_id,
            "seed": 0,
            "model": {
                "id": model["id"],
                "catalog_model_id": adapter_parameter("catalog_model_id", model["id"]),
                "project_directory": adapter_parameter("project_directory", None),
                "weight_path": model["weight_path"],
                "weight_sha256": model.get("weight_sha256"),
            },
            "dataset": {
                "id": dataset["id"],
                "image_directory": str(dataset_directory),
                "annotation_path": str(manifest_path),
            },
            "inference": {
                "device": "cuda:0",
                "precision": precision,
                "batch_size": batch_size,
                "warmup": warmup,
                "confidence": confidence,
                "nms_iou": nms_iou,
                "image_size": image_size,
                "input_height": input_height,
                "input_width": input_width,
                "max_detections": max_detections,
            },
            "category_aliases": {"motor": "motorcycle"},
            "output_directory": str(output_directory),
        }
        if command_mode:
            executable = Path(str(execution.get("executable", "")))
            working_directory = Path(str(execution.get("working_directory", "")))
            arguments = execution.get("arguments", [])
            if not executable.is_file():
                raise FileNotFoundError(
                    f"检测命令可执行程序不存在: {executable}"
                )
            if not working_directory.is_dir():
                raise FileNotFoundError(
                    f"检测命令工作目录不存在: {working_directory}"
                )
            if not isinstance(arguments, list) or not all(
                isinstance(value, str) for value in arguments
            ):
                raise ValueError("检测命令参数必须是字符串数组")
            placeholders = {
                "annotation_path": str(manifest_path),
                "batch_size": str(batch_size),
                "confidence": str(confidence),
                "dataset_id": str(dataset["id"]),
                "device": "cuda:0",
                "image_directory": str(dataset_directory),
                "image_size": str(image_size),
                "input_height": str(input_height),
                "input_width": str(input_width),
                "max_detections": str(max_detections),
                "model_id": str(model["id"]),
                "nms_iou": str(nms_iou),
                "output_directory": str(output_directory),
                "precision": precision,
                "predictions_path": str(predictions_path),
                "project_directory": str(request["model"]["project_directory"] or ""),
                "request_path": str(request_path),
                "result_path": str(result_path),
                "warmup": str(warmup),
                "weight_path": str(model["weight_path"]),
            }
            try:
                command = render_command(str(executable), arguments, placeholders)
            except CommandTemplateError as exc:
                raise ValueError(str(exc)) from exc
            process_directory = working_directory
        else:
            entrypoint_value = adapter.get("entrypoint")
            if not entrypoint_value:
                raise ValueError("检测适配器没有配置入口脚本")
            entrypoint = Path(entrypoint_value)
            if not entrypoint.is_absolute():
                entrypoint = self.settings.root_dir / entrypoint
            if not entrypoint.is_file():
                raise FileNotFoundError(f"检测适配器入口不存在: {entrypoint}")
            request_path.write_text(json_dump(request), encoding="utf-8")
            command = self._adapter_command(adapter, entrypoint, request_path, result_path)
            process_directory = job_directory

        environment = os.environ.copy()
        environment.update(self._isolated_cache_environment(job_directory))
        environment.update(
            {
                "DRONEDETS_ROOT": str(self.settings.dronedets_root),
                "PYTHONUNBUFFERED": "1",
            }
        )
        self._progress(job["id"], 15, "启动自动标注模型")
        started = time.monotonic()
        returncode, log_tail = self._run_adapter_process(
            job["id"], command, process_directory, environment
        )
        duration_ms = (time.monotonic() - started) * 1000
        if returncode != 0:
            raise RuntimeError(
                f"检测适配器退出码 {returncode}: {log_tail[-500:]}"
            )
        if not command_mode:
            if not result_path.is_file():
                raise FileNotFoundError("检测适配器没有生成 result.json")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("protocol_version") != "1.0":
                raise ValueError("检测适配器结果协议版本不受支持")
            if result.get("job_id") != job["id"] or result.get("run_id") != run_id:
                raise ValueError("检测适配器结果与当前自动标注任务不匹配")
            if result.get("status") != "succeeded":
                raise ValueError(f"检测适配器返回失败状态: {result.get('status')}")
            if int(result.get("image_count", -1)) != len(images):
                raise ValueError("检测适配器处理图片数与自动标注清单不一致")
            result_relative = Path(str(result.get("predictions_path", "")))
            if result_relative.is_absolute() or ".." in result_relative.parts:
                raise ValueError("检测结果路径必须位于自动标注输出目录内")
            predictions_path = (output_directory / result_relative).resolve()
        if (
            not predictions_path.is_relative_to(output_directory.resolve())
            or not predictions_path.is_file()
        ):
            raise FileNotFoundError("检测模型没有生成有效的 predictions.json")

        self._progress(job["id"], 75, "校验并导入候选框")
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        if not isinstance(predictions, list):
            raise ValueError("检测结果必须是 COCO predictions 数组")
        boxes_by_image: dict[int, list[dict[str, Any]]] = {
            image_id: [] for image_id in image_dimensions
        }
        for index, prediction in enumerate(predictions, start=1):
            if not isinstance(prediction, dict):
                raise ValueError("COCO prediction 必须是对象")
            try:
                image_id = int(prediction["image_id"])
                model_category_id = int(prediction["category_id"])
                score = float(prediction["score"])
                bbox = [float(value) for value in prediction["bbox"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("COCO prediction 缺少有效的 image_id、category_id、score 或 bbox") from exc
            if image_id not in image_dimensions:
                raise ValueError(f"模型输出了清单之外的图片 ID: {image_id}")
            if model_category_id not in model_to_dataset:
                raise ValueError(f"模型输出了未登记类别 {model_category_id}")
            if len(bbox) != 4 or not all(math.isfinite(value) for value in [score, *bbox]):
                raise ValueError("COCO prediction 包含无效数值")
            if not 0 <= score <= 1:
                raise ValueError("COCO prediction 的 score 必须位于 0 到 1 之间")
            if score < confidence:
                continue
            _, image_width, image_height = image_dimensions[image_id]
            x, y, width, height = bbox
            left = max(0.0, min(float(image_width), x))
            top = max(0.0, min(float(image_height), y))
            right = max(left, min(float(image_width), x + width))
            bottom = max(top, min(float(image_height), y + height))
            if right <= left or bottom <= top:
                continue
            boxes_by_image[image_id].append(
                {
                    "id": f"auto_{job['id'].split('_')[-1]}_{index}",
                    "category_id": model_to_dataset[model_category_id],
                    "x": round(left, 2),
                    "y": round(top, 2),
                    "width": round(right - left, 2),
                    "height": round(bottom - top, 2),
                    "confidence": round(score, 6),
                    "source": "AUTO_MODEL",
                }
            )
        for image_boxes in boxes_by_image.values():
            image_boxes.sort(key=lambda item: item["confidence"], reverse=True)
            del image_boxes[max_detections:]
        accepted = sum(len(image_boxes) for image_boxes in boxes_by_image.values())

        now = utc_now()
        inserted_images = 0
        with self.db.connect() as connection:
            current = connection.execute(
                "SELECT frozen FROM datasets WHERE id=?", (dataset["id"],)
            ).fetchone()
            if not current or current["frozen"]:
                raise ValueError("数据集已不存在或已冻结，候选框未写入")
            for image_id, (sample_name, width, height) in image_dimensions.items():
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO sample_annotations
                    (dataset_id,sample_name,width,height,boxes,completed,updated_at)
                    VALUES (?,?,?,?,?,0,?)
                    """,
                    (
                        dataset["id"],
                        sample_name,
                        width,
                        height,
                        json_dump(boxes_by_image[image_id]),
                        now,
                    ),
                )
                inserted_images += int(cursor.rowcount > 0)
            if inserted_images:
                connection.execute(
                    "UPDATE datasets SET annotation_status='ANNOTATING' WHERE id=?",
                    (dataset["id"],),
                )
        return {
            "dataset_id": dataset["id"],
            "model_id": model["id"],
            "model_name": model["name"],
            "annotation_status": "ANNOTATING" if inserted_images else dataset["annotation_status"],
            "processed_images": len(images),
            "annotated_images": inserted_images,
            "skipped_existing_images": (
                len(raster_files) - len(eligible_files) + len(images) - inserted_images
            ),
            "accepted_boxes": accepted,
            "effective_inference": {
                "confidence": confidence,
                "nms_iou": nms_iou,
                "image_size": image_size,
                "input_height": input_height,
                "input_width": input_width,
                "max_detections": max_detections,
                "batch_size": batch_size,
                "warmup": warmup,
                "precision": precision,
                "device": "cuda:0",
            },
            "category_mapping": compatibility["model_to_dataset"],
            "predictions_path": predictions_path.relative_to(
                self.settings.artifact_dir
            ).as_posix(),
            "duration_ms": round(duration_ms, 3),
        }

    def _run_evaluation(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        plan = self.db.row("SELECT * FROM evaluation_plans WHERE id=?", (payload["plan_id"],))
        if not plan:
            raise ValueError("评测方案不存在")
        dataset_ids = json_load(plan["dataset_ids"], [])
        model_ids = json_load(plan["model_ids"], [])
        protocol = json_load(plan.get("protocol"), {})
        compatibility_results = validate_evaluation_categories(
            dataset_ids,
            model_ids,
            self.db,
            evaluation_categories=protocol.get("evaluation_categories"),
        )
        category_scopes = {
            (item["dataset_id"], item["model_id"]): item
            for item in compatibility_results
        }
        seeds = json_load(plan["seeds"], [])
        blur_levels = json_load(plan["blur_levels"], [0])
        combinations = [(dataset, model, seed, blur) for dataset in dataset_ids for model in model_ids for blur in blur_levels for seed in seeds]
        created_runs: list[str] = []
        for index, (dataset_id, model_id, seed, blur) in enumerate(combinations):
            self._check_cancelled(job["id"])
            dataset = self.db.row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
            model = self.db.row("SELECT * FROM models WHERE id=?", (model_id,))
            if not dataset or not model:
                raise ValueError("数据集或模型不存在")
            if not dataset["frozen"]:
                raise ValueError(f"数据集尚未冻结: {dataset['name']}")
            if not model["is_demo"]:
                run_id = self._run_detector_evaluation(
                    job,
                    plan,
                    dataset,
                    model,
                    seed,
                    blur,
                )
                created_runs.append(run_id)
                self._progress(
                    job["id"],
                    8 + 88 * (index + 1) / len(combinations),
                    f"真实检测评测 {index + 1}/{len(combinations)}",
                )
                continue
            run_id = new_id("run")
            stable = int(hashlib.sha256(f"{dataset_id}{model_id}{seed}{blur}".encode()).hexdigest()[:8], 16)
            family_bias = {"YOLOv5": 0.79, "Faster R-CNN": 0.83, "RetinaNet": 0.76}.get(model["family"], 0.72)
            source_penalty = 0.06 if dataset["weather"] == "雾" else 0
            map_value = max(0.08, min(0.95, family_bias - blur * 0.34 - source_penalty + ((stable % 31) - 15) / 1200))
            latency = {"YOLOv5": 12.8, "Faster R-CNN": 31.4, "RetinaNet": 18.7}.get(model["family"], 22.0)
            latency += (stable % 20) / 20
            config = {
                "batch_size": payload.get("batch_size", 1),
                "precision": payload.get("precision", "FP16"),
                "warmup": payload.get("warmup", 20),
                "blur_level": blur,
                "metric_protocol": "reference-demo-v1",
                "evaluation_categories": category_scopes[
                    (dataset_id, model_id)
                ]["evaluation_categories"],
            }
            now = utc_now()
            self.db.execute(
                """
                INSERT INTO runs
                (id,plan_id,job_id,dataset_id,model_id,seed,status,config,environment_fingerprint,
                 hardware_profile,created_at,finished_at)
                VALUES (?,?,?,?,?,?, 'SUCCEEDED', ?,?,?,?,?)
                """,
                (
                    run_id,
                    plan["id"],
                    job["id"],
                    dataset_id,
                    model_id,
                    seed,
                    json_dump(config),
                    "reference-demo-environment",
                    json_dump({"device": "流程样例设备", "precision": config["precision"]}),
                    now,
                    now,
                ),
            )
            metrics = make_metrics(map_value, latency, family_bias - map_value)
            self.db.execute(
                """
                INSERT INTO results
                (id,run_id,map,map50,map75,precision,recall,f1,latency_p50,latency_p95,fps,
                 peak_memory,delta_map,metrics,curves,is_official,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    new_id("result"),
                    run_id,
                    metrics["map"], metrics["map50"], metrics["map75"], metrics["precision"],
                    metrics["recall"], metrics["f1"], metrics["latency_p50"], metrics["latency_p95"],
                    metrics["fps"], metrics["peak_memory"], -max(0.0, family_bias - map_value),
                    json_dump(metrics), json_dump(make_curves(map_value)), 0, now,
                ),
            )
            created_runs.append(run_id)
            self._progress(job["id"], 8 + 88 * (index + 1) / len(combinations), f"参考评测 {index + 1}/{len(combinations)}")
            time.sleep(0.03)
        return {"plan_id": plan["id"], "run_ids": created_runs, "count": len(created_runs), "is_official": False}

    def _run_detector_evaluation(
        self,
        job: dict[str, Any],
        plan: dict[str, Any],
        dataset: dict[str, Any],
        model: dict[str, Any],
        seed: int,
        blur: float,
    ) -> str:
        adapter = self.db.row(
            "SELECT * FROM adapters WHERE id=?", (model["adapter_id"],)
        )
        if not adapter or adapter["kind"] != "DETECTOR":
            raise ValueError(f"模型没有可用的检测适配器: {model['name']}")
        if not model.get("weight_path") or not Path(model["weight_path"]).is_file():
            raise FileNotFoundError(f"模型权重不存在: {model.get('weight_path')}")

        artifact_root = self.settings.artifact_dir.resolve()
        dataset_directory = (
            artifact_root / str(dataset.get("artifact_path") or "")
        ).resolve()
        if (
            dataset_directory == artifact_root
            or not dataset_directory.is_relative_to(artifact_root)
            or not dataset_directory.is_dir()
        ):
            raise ValueError(f"数据集 Artifact 不可用: {dataset['name']}")

        run_id = new_id("run")
        job_directory = self.settings.task_dir / job["id"] / "runs" / run_id
        output_directory = self.settings.artifact_dir / "evaluations" / run_id
        job_directory.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=True, exist_ok=True)

        annotation_path = dataset_directory / "annotations" / "instances.json"
        annotation_conversion = None
        if not annotation_path.is_file():
            annotations_directory = dataset_directory / "annotations"
            visdrone_directory = next(
                (
                    candidate
                    for candidate in (
                        annotations_directory / "visdrone",
                        annotations_directory / "yolo",
                        annotations_directory,
                    )
                    if candidate.is_dir()
                    and is_visdrone_label_directory(candidate)
                ),
                None,
            )
            if not visdrone_directory:
                raise FileNotFoundError(
                    "数据集缺少已提交的 COCO 标注，也未找到可转换的 "
                    f"VisDrone TXT 标注: {annotation_path}"
                )
            annotation_path = job_directory / "annotations" / "instances.json"
            annotation_conversion = convert_visdrone_to_coco(
                dataset_directory,
                visdrone_directory,
                annotation_path,
                dataset["name"],
            )
        ground_truth = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not ground_truth.get("images") or not ground_truth.get("categories"):
            raise ValueError("COCO 标注必须包含图片和类别")
        protocol = json_load(plan.get("protocol"), {})
        selected_categories = protocol.get("evaluation_categories")
        compatibility = validate_evaluation_categories(
            [dataset["id"]],
            [model["id"]],
            self.db,
            evaluation_categories=selected_categories,
        )[0]
        subset_evaluation = selected_categories is not None
        model_to_dataset = {
            int(model_id): int(dataset_id)
            for model_id, dataset_id in compatibility["model_to_dataset"].items()
        }
        dataset_to_model = {
            dataset_id: model_id for model_id, dataset_id in model_to_dataset.items()
        }
        stored_row = self.db.row(
            "SELECT categories FROM dataset_annotation_schemas WHERE dataset_id=?",
            (dataset["id"],),
        )
        stored_categories = normalize_categories(
            json_load(stored_row["categories"], []) if stored_row else []
        )
        actual_categories = normalize_categories(ground_truth["categories"])
        if {
            (item["id"], normalize_category_name(item["name"]))
            for item in stored_categories
        } != {
            (item["id"], normalize_category_name(item["name"]))
            for item in actual_categories
        }:
            raise ValueError("COCO 标注类别与数据集登记类别不一致")
        model_categories = normalize_categories(json_load(model["categories"], []))
        inference_ground_truth = json.loads(json.dumps(ground_truth))
        inference_ground_truth["categories"] = model_categories
        inference_annotations = []
        for annotation in inference_ground_truth.get("annotations", []):
            dataset_category_id = int(annotation["category_id"])
            if dataset_category_id not in dataset_to_model:
                if subset_evaluation:
                    continue
                raise ValueError(f"数据集标注引用了未登记类别 {dataset_category_id}")
            annotation["category_id"] = dataset_to_model[dataset_category_id]
            inference_annotations.append(annotation)
        inference_ground_truth["annotations"] = inference_annotations
        inference_annotation_path = (
            job_directory / "annotations" / "model-category-space.json"
        )
        inference_annotation_path.parent.mkdir(parents=True, exist_ok=True)
        inference_annotation_path.write_text(
            json_dump(inference_ground_truth), encoding="utf-8"
        )

        image_directory = dataset_directory
        if float(blur) > 0:
            image_directory = job_directory / "images"
            self._prepare_blurred_evaluation_images(
                dataset_directory,
                image_directory,
                ground_truth,
                float(blur),
            )

        schema = json_load(adapter.get("parameter_schema"), {})
        properties = schema.get("properties", {})

        def parameter(name: str, fallback: Any) -> Any:
            specification = properties.get(name, {})
            return specification.get("const", specification.get("default", fallback))

        def task_parameter(name: str, fallback: Any) -> Any:
            default = parameter(name, fallback)
            return protocol.get(name, default) if name in properties else default

        config = {
            "batch_size": int(protocol.get("batch_size", 1)),
            "precision": str(protocol.get("precision", model["precision"])),
            "warmup": int(protocol.get("warmup", 20)),
            "blur_level": float(blur),
            "confidence": float(task_parameter("confidence", 0.001)),
            "nms_iou": float(task_parameter("nms_iou", 0.7)),
            "image_size": int(task_parameter("image_size", 1280)),
            "input_height": int(task_parameter("input_height", 1280)),
            "input_width": int(task_parameter("input_width", 1280)),
            "max_detections": int(task_parameter("max_detections", 300)),
            "metric_protocol": "pycocotools-2.0.11",
            "adapter_id": adapter["id"],
            "annotation_conversion": annotation_conversion,
            "evaluation_categories": compatibility["evaluation_categories"],
        }
        request_path = job_directory / "request.json"
        result_path = job_directory / "result.json"
        request = {
            "protocol_version": "1.0",
            "job_id": job["id"],
            "run_id": run_id,
            "seed": int(seed),
            "model": {
                "id": model["id"],
                "catalog_model_id": parameter(
                    "catalog_model_id", "yolov8m_visdrone"
                ),
                "project_directory": parameter(
                    "project_directory", None
                ),
                "weight_path": model["weight_path"],
                "weight_sha256": model.get("weight_sha256"),
            },
            "dataset": {
                "id": dataset["id"],
                "image_directory": str(image_directory),
                "annotation_path": str(inference_annotation_path),
            },
            "inference": {
                "device": "cuda:0",
                "precision": config["precision"],
                "batch_size": config["batch_size"],
                "warmup": config["warmup"],
                "confidence": config["confidence"],
                "nms_iou": config["nms_iou"],
                "image_size": config["image_size"],
                "input_height": config["input_height"],
                "input_width": config["input_width"],
                "max_detections": config["max_detections"],
            },
            "category_aliases": {"motor": "motorcycle"},
            "output_directory": str(output_directory),
        }
        request_path.write_text(json_dump(request), encoding="utf-8")

        execution = schema.get("execution", {})
        command_mode = execution.get("mode") == "command"
        predictions_filename = str(
            execution.get("predictions_filename", "predictions.json")
        )
        prediction_relative = Path(predictions_filename)
        if (
            prediction_relative.is_absolute()
            or ".." in prediction_relative.parts
        ):
            raise ValueError("命令模式预测结果路径必须位于评测输出目录内")
        if command_mode:
            executable = Path(str(execution.get("executable", "")))
            working_directory = Path(
                str(execution.get("working_directory", ""))
            )
            arguments = execution.get("arguments", [])
            if not executable.is_file():
                raise FileNotFoundError(
                    f"检测命令可执行程序不存在: {executable}"
                )
            if not working_directory.is_dir():
                raise FileNotFoundError(
                    f"检测命令工作目录不存在: {working_directory}"
                )
            if not isinstance(arguments, list) or not all(
                isinstance(value, str) for value in arguments
            ):
                raise ValueError("检测命令参数必须是字符串数组")
            placeholders = {
                "annotation_path": str(inference_annotation_path),
                "batch_size": str(config["batch_size"]),
                "confidence": str(config["confidence"]),
                "dataset_id": str(dataset["id"]),
                "device": "cuda:0",
                "image_directory": str(image_directory),
                "image_size": str(config["image_size"]),
                "input_height": str(config["input_height"]),
                "input_width": str(config["input_width"]),
                "max_detections": str(config["max_detections"]),
                "model_id": str(model["id"]),
                "nms_iou": str(config["nms_iou"]),
                "output_directory": str(output_directory),
                "precision": str(config["precision"]),
                "predictions_path": str(
                    output_directory / prediction_relative
                ),
                "project_directory": str(
                    request["model"].get("project_directory") or ""
                ),
                "request_path": str(request_path),
                "result_path": str(result_path),
                "warmup": str(config["warmup"]),
                "weight_path": str(model["weight_path"]),
            }
            try:
                command = render_command(
                    str(executable),
                    arguments,
                    placeholders,
                )
            except CommandTemplateError as exc:
                raise ValueError(str(exc)) from exc
            process_directory = working_directory
        else:
            entrypoint_value = adapter.get("entrypoint")
            if not entrypoint_value:
                raise ValueError("检测适配器没有配置入口脚本")
            entrypoint = Path(entrypoint_value)
            if not entrypoint.is_absolute():
                entrypoint = self.settings.root_dir / entrypoint
            if not entrypoint.is_file():
                raise FileNotFoundError(
                    f"检测适配器入口不存在: {entrypoint}"
                )
            command = self._adapter_command(
                adapter, entrypoint, request_path, result_path
            )
            process_directory = job_directory
        environment = os.environ.copy()
        environment.update(self._isolated_cache_environment(job_directory))
        environment.update(
            {
                "DRONEDETS_ROOT": str(self.settings.dronedets_root),
                "PYTHONUNBUFFERED": "1",
            }
        )
        self._progress(
            job["id"],
            20,
            "启动目标检测命令" if command_mode else "启动目标检测适配器",
        )
        command_started = time.monotonic()
        returncode, log_tail = self._run_adapter_process(
            job["id"], command, process_directory, environment
        )
        command_duration_ms = (
            time.monotonic() - command_started
        ) * 1000
        if returncode != 0:
            raise RuntimeError(
                f"检测适配器退出码 {returncode}: {log_tail[-500:]}"
            )
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        elif command_mode:
            result = {
                "protocol_version": "1.0",
                "job_id": job["id"],
                "run_id": run_id,
                "status": "succeeded",
                "predictions_path": predictions_filename,
                "image_count": len(ground_truth["images"]),
                "runtime": {
                    "duration_ms": command_duration_ms,
                    "peak_memory_mb": 0.0,
                },
                "environment": {
                    "execution_mode": "command",
                    "executable": str(command[0]),
                },
                "warnings": [
                    "模型命令未生成 result.json；时延仅包含进程总耗时，"
                    "不提供分阶段耗时和峰值显存。"
                ],
            }
        else:
            raise FileNotFoundError("检测适配器没有生成 result.json")
        if result.get("protocol_version") != "1.0":
            raise ValueError("检测适配器结果协议版本不受支持")
        if result.get("job_id") != job["id"] or result.get("run_id") != run_id:
            raise ValueError("检测适配器结果与当前运行不匹配")
        if result.get("status") != "succeeded":
            raise ValueError(f"检测适配器返回失败状态: {result.get('status')}")
        if int(result.get("image_count", -1)) != len(ground_truth["images"]):
            raise ValueError("检测适配器处理图片数与 COCO 真值不一致")

        prediction_relative = Path(str(result.get("predictions_path", "")))
        if prediction_relative.is_absolute() or ".." in prediction_relative.parts:
            raise ValueError("检测结果路径必须位于评测 Artifact 目录内")
        predictions_path = (output_directory / prediction_relative).resolve()
        if (
            not predictions_path.is_relative_to(output_directory.resolve())
            or not predictions_path.is_file()
        ):
            raise FileNotFoundError("检测适配器没有生成有效的 predictions.json")
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        if not isinstance(predictions, list):
            raise ValueError("检测结果必须是 COCO predictions 数组")
        mapped_predictions = []
        for prediction in predictions:
            model_category_id = int(prediction["category_id"])
            if model_category_id not in model_to_dataset:
                if subset_evaluation:
                    continue
                raise ValueError(f"模型输出了未登记类别 {model_category_id}")
            prediction["category_id"] = model_to_dataset[model_category_id]
            mapped_predictions.append(prediction)
        predictions = mapped_predictions
        predictions_path.write_text(json_dump(predictions), encoding="utf-8")
        self._progress(job["id"], 78, "计算 COCO 检测指标")
        metrics, curves = evaluate_coco_predictions(
            annotation_path,
            predictions_path,
            result.get("runtime", {}),
            compatibility["evaluation_category_ids"],
        )
        config.update(
            {
                "predictions_path": predictions_path.relative_to(
                    self.settings.artifact_dir
                ).as_posix(),
                "unmatched_labels": result.get("unmatched_labels", {}),
                "warnings": result.get("warnings", []),
                "category_mapping": compatibility["model_to_dataset"],
            }
        )
        environment_details = result.get("environment", {})
        fingerprint = hashlib.sha256(
            json_dump(
                {
                    "environment": environment_details,
                    "weight_sha256": model.get("weight_sha256"),
                }
            ).encode()
        ).hexdigest()
        is_official = (
            adapter["maturity"] == "BENCHMARK_READY"
            and model["status"] == "BENCHMARK_READY"
        )
        now = utc_now()
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs
                (id,plan_id,job_id,dataset_id,model_id,seed,status,config,
                 environment_fingerprint,hardware_profile,created_at,finished_at)
                VALUES (?,?,?,?,?,?, 'SUCCEEDED', ?,?,?,?,?)
                """,
                (
                    run_id,
                    plan["id"],
                    job["id"],
                    dataset["id"],
                    model["id"],
                    int(seed),
                    json_dump(config),
                    fingerprint,
                    json_dump(environment_details),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO results
                (id,run_id,map,map50,map75,precision,recall,f1,latency_p50,
                 latency_p95,fps,peak_memory,delta_map,metrics,curves,is_official,
                 created_at)
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
                    None,
                    json_dump(metrics),
                    json_dump(curves),
                    int(is_official),
                    now,
                ),
            )
        return run_id

    @staticmethod
    def _prepare_blurred_evaluation_images(
        source_directory: Path,
        target_directory: Path,
        ground_truth: dict[str, Any],
        blur: float,
    ) -> None:
        target_directory.mkdir(parents=True, exist_ok=True)
        radius = blur * 8
        for image in ground_truth["images"]:
            relative = Path(str(image["file_name"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"COCO 图片路径不安全: {relative}")
            source = (source_directory / relative).resolve()
            if (
                not source.is_relative_to(source_directory.resolve())
                or not source.is_file()
            ):
                raise FileNotFoundError(f"缺少评测图片: {source}")
            target = target_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as opened:
                opened.convert("RGB").filter(
                    ImageFilter.GaussianBlur(radius=radius)
                ).save(target)

    def _isolated_cache_environment(self, job_dir: Path) -> dict[str, str]:
        cache = self.settings.runtime_dir / "cache"
        values = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "XDG_CACHE_HOME": cache / "xdg",
            "PIP_CACHE_DIR": cache / "pip",
            "HF_HOME": cache / "huggingface",
            "TORCH_HOME": cache / "torch",
            "CUDA_CACHE_PATH": cache / "cuda",
            "MPLCONFIGDIR": cache / "matplotlib",
            "TMPDIR": job_dir / "tmp",
        }
        for path in values.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
        return {key: str(value) for key, value in values.items()}


class JobCancelled(Exception):
    pass


def main() -> None:
    db.initialize()
    agent = JobAgent()
    try:
        agent.run_forever()
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    main()
