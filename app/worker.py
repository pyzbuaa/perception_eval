from __future__ import annotations

import hashlib
import json
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

from app.config import Settings, settings
from app.db import Database, db, json_dump, json_load, make_curves, make_metrics, new_id, utc_now


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
            "source_type": payload.get("source_type", "REPLAY_FIXTURE"),
            "output_directory": str(artifact_dir),
        }
        if payload["adapter_id"] == "adapter_condition":
            input_dataset_id = payload.get("input_dataset_id")
            if not input_dataset_id:
                raise ValueError("条件退化任务必须选择一个输入数据集")
            input_dataset = self.db.row("SELECT * FROM datasets WHERE id=?", (input_dataset_id,))
            if not input_dataset or not input_dataset.get("artifact_path"):
                raise ValueError("输入数据集不存在或没有 Artifact")
            input_directory = self.settings.artifact_dir / input_dataset["artifact_path"]
            input_images = [
                str(path)
                for path in sorted(input_directory.iterdir())
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if not input_images:
                raise ValueError("输入数据集没有可处理的 PNG/JPEG/WebP 图像；SVG 流程样例不用于像素退化")
            request["input_images"] = input_images
            request["input_dataset_id"] = input_dataset_id
            request["sample_count"] = min(request["sample_count"], len(input_images))
        request_path.write_text(json_dump(request), encoding="utf-8")
        self._progress(job["id"], 12, "校验适配器协议")
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
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = self._adapter_command(
            adapter, entrypoint, request_path, result_path
        )
        self._progress(job["id"], 25, "启动生成适配器")
        returncode, log_tail = self._run_adapter_process(
            job["id"], command, job_dir, environment
        )
        if returncode != 0:
            raise RuntimeError(
                f"适配器退出码 {returncode}: {log_tail[-500:]}"
            )
        self._progress(job["id"], 75, "验证输出文件与元数据")
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
        dataset_id = new_id("dataset")
        sensor = payload.get("conditions", {}).get("sensor", {})
        scene = payload.get("conditions", {}).get("scene", {})
        resolution = sensor.get("resolution", "1920×1080")
        if isinstance(resolution, list):
            resolution = "×".join(str(value) for value in resolution)
        annotation_status = "CANDIDATE" if result.get("has_candidate_annotations") else "UNLABELED"
        relative = artifact_dir.relative_to(self.settings.artifact_dir).as_posix()
        self.db.execute(
            """
            INSERT INTO datasets
            (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
             sample_count,annotation_status,frozen,artifact_path,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                dataset_id,
                payload["name"],
                "v1",
                payload.get("source_type", "REPLAY_FIXTURE"),
                scene.get("domain_label", scene.get("domain", "无人机航拍")),
                scene.get("weather", "晴朗"),
                json_dump(sensor),
                resolution,
                len(samples),
                annotation_status,
                0,
                relative,
                utc_now(),
            ),
        )
        self._progress(job["id"], 95, "创建数据集草稿")
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
        if event.get("type") != "progress":
            return
        current = int(event.get("current", 0))
        total = max(1, int(event.get("total", 1)))
        progress = 25 + 48 * current / total
        self._progress(job_id, progress, f"生成图像 {current}/{total}")

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
        source = Path(payload["directory"]).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"导入目录不存在: {source}")
        candidates = [
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".svg"}
        ]
        if not candidates:
            raise ValueError("目录中没有支持的图像")
        target = self.settings.artifact_dir / "imports" / job["id"]
        target.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(candidates):
            self._check_cancelled(job["id"])
            shutil.copy2(path, target / f"{index:06d}{path.suffix.lower()}")
            self._progress(job["id"], 10 + 70 * (index + 1) / len(candidates), f"复制图像 {index + 1}/{len(candidates)}")
        annotation = payload.get("annotation_path")
        annotation_status = "UNLABELED"
        if annotation:
            annotation_path = Path(annotation).expanduser().resolve()
            if annotation_path.is_file():
                shutil.copy2(annotation_path, target / annotation_path.name)
                annotation_status = "CANDIDATE"
        dataset_id = new_id("dataset")
        self.db.execute(
            """
            INSERT INTO datasets
            (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
             sample_count,annotation_status,frozen,artifact_path,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                dataset_id,
                payload["name"],
                "v1",
                "REAL",
                payload.get("scene_domain", "未分类"),
                "未记录",
                "{}",
                "原始分辨率",
                len(candidates),
                annotation_status,
                0,
                target.relative_to(self.settings.artifact_dir).as_posix(),
                utc_now(),
            ),
        )
        return {"dataset_id": dataset_id, "samples": len(candidates), "annotation_status": annotation_status}

    def _run_evaluation(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        plan = self.db.row("SELECT * FROM evaluation_plans WHERE id=?", (payload["plan_id"],))
        if not plan:
            raise ValueError("评测方案不存在")
        dataset_ids = json_load(plan["dataset_ids"], [])
        model_ids = json_load(plan["model_ids"], [])
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
