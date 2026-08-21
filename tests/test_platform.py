from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

import app.main as main_module
from adapters import diffusionblur_motion, diffusiondegrade_day_to_night
from adapters.basegen_generator import emit_stage, prepare_plan, resolve_cached_model_path
from adapters.diffusiondegrade_fog import blend_fog
from adapters.dronedets_detector import category_mapping
from app.command_protocol import (
    CommandTemplateError,
    command_placeholders,
    render_command,
    validate_command_arguments,
)
from app.category_templates import list_category_templates, template_categories
from app.config import ROOT_DIR, Settings
from app.db import Database, json_dump, utc_now
from app.detection_metrics import evaluate_coco_predictions
from app.services import (
    DatasetAnnotationError,
    DatasetDeletionError,
    DatasetImportError,
    EvaluationResultDeletionError,
    CategoryCompatibilityError,
    JobDeletionError,
    LocalModelRegistrationError,
    ModelDeletionError,
    complete_dataset_annotations,
    category_compatibility,
    delete_dataset,
    delete_evaluation_result,
    delete_job,
    delete_model,
    dataset_statistics,
    evaluation_run_visualization,
    get_annotation_session,
    get_basegen_scene_schema,
    get_sample_annotation,
    list_datasets,
    list_dataset_samples,
    list_local_dataset_resources,
    list_local_model_resources,
    preview_basegen_plan,
    query_results,
    queue_job,
    read_dataset_annotation_categories,
    register_local_detector_model,
    resolve_local_dataset_import,
    save_sample_annotation,
    summarize_resolutions,
    update_annotation_schema,
    update_local_detector_model,
    validate_evaluation_categories,
)
from app.schemas import DatasetImportRequest
from app.worker import JobAgent


def test_motion_blur_condition_files_match_by_name_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "inputs"
    (input_directory / "nested").mkdir(parents=True)
    Image.new("RGB", (8, 8), (40, 80, 120)).save(
        input_directory / "matched.png"
    )
    Image.new("RGB", (8, 8), (80, 120, 160)).save(
        input_directory / "nested" / "missing.png"
    )
    condition_directory = tmp_path / "conditions"
    condition_directory.mkdir()
    file_condition = np.stack(
        (
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32),
            np.full((8, 8), 0.25, dtype=np.float32),
        )
    )
    np.save(condition_directory / "matched_condition.npy", file_condition)

    generated_motions: list[str] = []
    received_conditions: list[np.ndarray] = []

    class FakeConfig:
        def __init__(self, motion: str, mean_strength: float):
            self.motion = motion
            self.mean_strength = mean_strength

    class FakeGenerator:
        def __init__(self, config: FakeConfig):
            self.config = config

        def generate(self, height: int, width: int) -> np.ndarray:
            generated_motions.append(self.config.motion)
            condition = np.zeros((3, height, width), dtype=np.float32)
            condition[0] = 1
            condition[2] = self.config.mean_strength
            return condition

    class FakePipeline:
        def generate(
            self,
            image: np.ndarray,
            condition: np.ndarray,
            sample_timesteps: int,
            seed: int,
        ) -> np.ndarray:
            assert sample_timesteps == 20
            received_conditions.append(condition.copy())
            return image.copy()

    monkeypatch.setattr(
        diffusionblur_motion,
        "validate_installation",
        lambda: (tmp_path / "DiffusionBlur", tmp_path / "ID_Blau.pth"),
    )
    monkeypatch.setattr(
        diffusionblur_motion,
        "load_components",
        lambda root, checkpoint: (
            FakeGenerator,
            FakeConfig,
            lambda image: (image, (0, 0)),
            FakePipeline(),
        ),
    )
    output_directory = tmp_path / "outputs"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "protocol_version": "1.0",
                "job_id": "job-motion-condition",
                "input_directory": str(input_directory),
                "input_images": [
                    str(input_directory / "matched.png"),
                    str(input_directory / "nested" / "missing.png"),
                ],
                "output_directory": str(output_directory),
                "sample_count": 2,
                "seeds": [100],
                "has_source_annotations": False,
                "model_parameters": {
                    "effect": "motion_blur",
                    "domain": "uav_aerial",
                    "motion": "forward",
                    "strength": 0.14,
                    "sample_timesteps": 20,
                    "condition_directory": str(condition_directory),
                },
            }
        ),
        encoding="utf-8",
    )

    diffusionblur_motion.run(request_path, result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["runtime"]["matched_conditions"] == 1
    assert result["runtime"]["fallback_conditions"] == 1
    assert result["samples"][0]["condition_source"] == "file"
    assert result["samples"][0]["condition_file"] == "matched_condition.npy"
    assert result["samples"][1]["condition_source"] == "random-preset-fallback"
    expected_fallback = random.Random(101).choice(
        sorted(diffusionblur_motion.MOTIONS)
    )
    assert result["samples"][1]["motion"] == expected_fallback
    assert generated_motions == [expected_fallback]
    np.testing.assert_array_equal(received_conditions[0], file_condition)
    assert (output_directory / "matched.png").is_file()
    assert (output_directory / "nested" / "missing.png").is_file()


def make_database(tmp_path: Path) -> tuple[Database, Settings]:
    app_settings = Settings(
        root_dir=ROOT_DIR,
        data_dir=tmp_path / "data",
        dataset_library_root=tmp_path,
    )
    database = Database(app_settings)
    database.initialize()
    return database, app_settings


def test_database_seeds_traceable_demo_data(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    assert database.row("SELECT COUNT(*) AS n FROM datasets")["n"] == 3
    assert database.row("SELECT COUNT(*) AS n FROM models")["n"] == 4
    assert database.row("SELECT COUNT(*) AS n FROM results")["n"] == 27
    assert database.row("SELECT COUNT(*) AS n FROM results WHERE is_official=1")["n"] == 0
    datasets = list(list_datasets(database))
    assert all(Path(item["dataset_path"]).is_absolute() for item in datasets)
    adapter = database.row("SELECT * FROM adapters WHERE id='adapter_basegen'")
    assert adapter
    assert adapter["runtime_kind"] == "conda_external"
    assert adapter["requires_gpu"] == 1
    assert database.row("SELECT * FROM adapters WHERE id='adapter_replay'") is None
    detector = database.row(
        "SELECT * FROM adapters WHERE id='adapter_dronedets_yolov8m'"
    )
    assert detector and detector["runtime_prefix"].endswith("DroneDets/.venv")
    condition = database.row(
        "SELECT * FROM adapters WHERE id='adapter_condition'"
    )
    assert condition and condition["runtime_kind"] == "conda_external"
    assert condition["runtime_prefix"].endswith("DiffusionDegrade/.venv")
    assert condition["entrypoint"] == "adapters/diffusiondegrade_fog.py"
    assert condition["requires_gpu"] == 1
    motion_blur = database.row(
        "SELECT * FROM adapters WHERE id='adapter_motion_blur'"
    )
    assert motion_blur and motion_blur["runtime_kind"] == "conda_external"
    assert motion_blur["runtime_prefix"].endswith("envs/blau")
    assert motion_blur["entrypoint"] == "adapters/diffusionblur_motion.py"
    assert motion_blur["requires_gpu"] == 1
    day_to_night = database.row(
        "SELECT * FROM adapters WHERE id='adapter_day_to_night'"
    )
    assert day_to_night and day_to_night["runtime_kind"] == "conda_external"
    assert day_to_night["runtime_prefix"].endswith("DiffusionDegrade/.venv")
    assert day_to_night["entrypoint"] == (
        "adapters/diffusiondegrade_day_to_night.py"
    )
    assert day_to_night["requires_gpu"] == 1
    day_to_night_properties = json.loads(day_to_night["parameter_schema"])[
        "properties"
    ]
    assert day_to_night_properties["image_prep"]["const"] == "resize_640x640"
    assert day_to_night_properties["model_size"]["const"] == 640
    assert "tile_size" not in day_to_night_properties
    assert "overlap" not in day_to_night_properties
    warpi2i_fog = database.row(
        "SELECT * FROM adapters WHERE id='adapter_warpi2i_fog'"
    )
    assert warpi2i_fog
    assert warpi2i_fog["runtime_prefix"].endswith("DiffusionDegrade/.venv")
    assert warpi2i_fog["entrypoint"] == "adapters/warpi2i_driving.py"
    warpi2i_day_to_night = database.row(
        "SELECT * FROM adapters WHERE id='adapter_warpi2i_day_to_night'"
    )
    assert warpi2i_day_to_night
    assert warpi2i_day_to_night["runtime_prefix"].endswith(
        "DiffusionDegrade/.venv"
    )
    assert warpi2i_day_to_night["entrypoint"] == "adapters/warpi2i_driving.py"
    detector_model = database.row(
        "SELECT weight_path FROM models "
        "WHERE id='model_dronedets_yolov8m_visdrone'"
    )
    if detector_model["weight_path"]:
        assert Path(detector_model["weight_path"]).suffix == ".pt"


def test_database_does_not_restore_deleted_demo_data(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    with database.connect() as connection:
        connection.execute("DELETE FROM results")
        connection.execute("DELETE FROM runs")
        connection.execute("DELETE FROM evaluation_plans")
        connection.execute("DELETE FROM jobs")
        connection.execute("DELETE FROM datasets")
        connection.execute("DELETE FROM models")

    database.initialize()

    for table in ("datasets", "models", "evaluation_plans", "runs", "results", "jobs"):
        assert database.row(f"SELECT COUNT(*) AS n FROM {table}")["n"] == 0
    assert database.row("SELECT COUNT(*) AS n FROM adapters")["n"] == 8


def test_diffusiondegrade_fog_strength_blends_source_and_model_output() -> None:
    source = Image.new("RGB", (2, 2), (0, 0, 0))
    fogged = Image.new("RGB", (2, 2), (200, 200, 200))

    assert blend_fog(source, fogged, 0).getpixel((0, 0)) == (0, 0, 0)
    assert blend_fog(source, fogged, 0.5).getpixel((0, 0)) == (100, 100, 100)
    assert blend_fog(source, fogged, 1).getpixel((0, 0)) == (200, 200, 200)
    with pytest.raises(ValueError, match="0 到 1"):
        blend_fog(source, fogged, 1.1)


def test_uav_low_light_processes_one_full_frame_without_tiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    input_path = input_directory / "aerial.png"
    Image.new("RGB", (1400, 900), (40, 80, 120)).save(input_path)
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"checkpoint")
    translated_sizes: list[tuple[int, int]] = []

    class FakeCuda:
        @staticmethod
        def manual_seed_all(seed: int) -> None:
            return None

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

    def translate(image: Image.Image) -> Image.Image:
        translated_sizes.append(image.size)
        return image.resize((640, 640)).resize(image.size)

    monkeypatch.setattr(
        diffusiondegrade_day_to_night,
        "validate_installation",
        lambda: (tmp_path, checkpoint),
    )
    monkeypatch.setattr(
        diffusiondegrade_day_to_night,
        "load_translator",
        lambda root, model: (FakeTorch(), translate),
    )
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(
        json_dump(
            {
                "protocol_version": "1.0",
                "job_id": "job_low_light",
                "sample_count": 1,
                "input_images": [str(input_path)],
                "input_directory": str(input_directory),
                "output_directory": str(output_directory),
                "model_parameters": {
                    "effect": "day_to_night",
                    "domain": "uav_aerial",
                },
            }
        ),
        encoding="utf-8",
    )

    diffusiondegrade_day_to_night.run(request_path, result_path)

    assert translated_sizes == [(1400, 900)]
    with Image.open(output_directory / "aerial.png") as output:
        assert output.size == (1400, 900)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["runtime"]["image_prep"] == "resize_640x640"
    assert result["runtime"]["output_restore"] == "original_size"
    assert "tile_size" not in result["runtime"]
    assert "overlap" not in result["runtime"]


def test_resolution_summary_uses_value_or_range() -> None:
    assert summarize_resolutions([(1920, 1080)]) == "1920×1080"
    assert summarize_resolutions([(1920, 1080), (640, 480)]) == (
        "640×480 ～ 1920×1080"
    )


def test_basegen_loading_stage_updates_job(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, app_settings = make_database(tmp_path)
    job = queue_job("ACQUISITION", {}, database)
    emit_stage("加载生成模型（可能需要数分钟）")

    JobAgent(database, app_settings)._handle_adapter_progress(
        job["id"], capsys.readouterr().out.strip()
    )

    updated = database.row("SELECT progress,stage FROM jobs WHERE id=?", (job["id"],))
    assert updated["progress"] == 1
    assert updated["stage"] == "加载生成模型（可能需要数分钟）"

    agent = JobAgent(database, app_settings)
    agent._handle_adapter_progress(
        job["id"], json.dumps({"type": "progress", "current": 17, "total": 100})
    )
    updated = database.row("SELECT progress,stage FROM jobs WHERE id=?", (job["id"],))
    assert updated["progress"] == 17
    assert updated["stage"] == "生成图像 17/100"

    agent._handle_adapter_progress(
        job["id"], json.dumps({"type": "progress", "current": 100, "total": 100})
    )
    updated = database.row("SELECT progress,stage FROM jobs WHERE id=?", (job["id"],))
    assert updated["progress"] == 99
    assert updated["stage"] == "生成图像 100/100"


def test_dataset_migration_marks_existing_id_blau_motion_blur(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,
         resolution,sample_count,annotation_status,frozen,artifact_path,
         category_template,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_legacy_motion_blur",
            "旧版运动模糊数据",
            "v1",
            "REAL_TRANSFORMED",
            "无人机航拍",
            "晴朗",
            json_dump(
                {
                    "degradation": "ID-Blau UAV Motion Blur",
                    "motion": "forward",
                    "motion_blur_strength": 0.14,
                }
            ),
            "原始分辨率",
            1,
            "UNLABELED",
            0,
            None,
            "custom",
            utc_now(),
        ),
    )

    database.initialize()

    row = database.row(
        "SELECT sensor_conditions FROM datasets WHERE id='dataset_legacy_motion_blur'"
    )
    conditions = json.loads(row["sensor_conditions"])
    assert conditions["motion_blur"] is True
    assert conditions["motion_blur_model"] == "ID-Blau"
    assert conditions["motion_blur_sample_timesteps"] == 20


def test_dataset_migration_renames_autonomous_driving_domain(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    dataset_id = database.row("SELECT id FROM datasets LIMIT 1")["id"]
    database.execute(
        "UPDATE datasets SET scene_domain='城市自动驾驶感知',"
        "name='城市自动驾驶感知 · 基础图像生成' WHERE id=?",
        (dataset_id,),
    )

    database.initialize()

    dataset = database.row(
        "SELECT name,scene_domain FROM datasets WHERE id=?", (dataset_id,)
    )
    assert dataset["scene_domain"] == "自动驾驶"
    assert dataset["name"] == "自动驾驶 · 基础图像生成"


def test_delete_job_rejects_active_task_and_preserves_evaluation_results(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    job = queue_job("EVALUATION", {"plan_id": "test"}, database)
    workspace = app_settings.task_dir / job["id"]
    workspace.mkdir(parents=True)
    (workspace / "result.json").write_text("{}", encoding="utf-8")
    staged = app_settings.task_dir / "import_uploads" / "upload_test"
    staged.mkdir(parents=True)
    (staged / "annotation.json").write_text("{}", encoding="utf-8")
    log = app_settings.log_dir / f"{job['id']}.log"
    log.write_text("completed", encoding="utf-8")
    database.execute(
        "UPDATE jobs SET payload=? WHERE id=?",
        (json_dump({"staged_upload_root": str(staged)}), job["id"]),
    )
    with pytest.raises(JobDeletionError, match="尚未结束"):
        delete_job(job["id"], database)

    run = database.row("SELECT id FROM runs LIMIT 1")
    database.execute("UPDATE runs SET job_id=? WHERE id=?", (job["id"], run["id"]))
    database.execute(
        "UPDATE jobs SET status='SUCCEEDED',progress=100 WHERE id=?", (job["id"],)
    )
    result_count = database.row("SELECT COUNT(*) AS n FROM results")["n"]

    deleted = delete_job(job["id"], database)

    assert deleted and deleted["deleted"]
    assert deleted["results_preserved"]
    assert deleted["cleanup_errors"] == []
    assert database.row("SELECT id FROM jobs WHERE id=?", (job["id"],)) is None
    assert database.row("SELECT job_id FROM runs WHERE id=?", (run["id"],))["job_id"] is None
    assert database.row("SELECT COUNT(*) AS n FROM results")["n"] == result_count
    assert not workspace.exists()
    assert not staged.exists()
    assert not log.exists()


def test_delete_evaluation_result_moves_run_files_to_trash(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    run = database.row("SELECT * FROM runs LIMIT 1")
    job = queue_job("EVALUATION", {"plan_id": run["plan_id"]}, database)
    database.execute("UPDATE runs SET job_id=? WHERE id=?", (job["id"], run["id"]))

    with pytest.raises(EvaluationResultDeletionError, match="尚未结束"):
        delete_evaluation_result(run["id"], database)

    database.execute(
        "UPDATE jobs SET status='SUCCEEDED',progress=100 WHERE id=?", (job["id"],)
    )
    artifact = app_settings.artifact_dir / "evaluations" / run["id"]
    artifact.mkdir(parents=True)
    (artifact / "predictions.json").write_text("[]", encoding="utf-8")
    workspace = app_settings.task_dir / job["id"] / "runs" / run["id"]
    workspace.mkdir(parents=True)
    (workspace / "request.json").write_text("{}", encoding="utf-8")
    dataset_id = run["dataset_id"]
    model_id = run["model_id"]

    deleted = delete_evaluation_result(run["id"], database)

    assert deleted and deleted["deleted"]
    assert deleted["artifact_moved"]
    assert deleted["workspace_moved"]
    assert database.row("SELECT id FROM results WHERE run_id=?", (run["id"],)) is None
    assert database.row("SELECT id FROM runs WHERE id=?", (run["id"],)) is None
    assert database.row("SELECT id FROM datasets WHERE id=?", (dataset_id,))
    assert database.row("SELECT id FROM models WHERE id=?", (model_id,))
    assert database.row("SELECT id FROM jobs WHERE id=?", (job["id"],))
    trash = app_settings.data_dir / "trash" / "evaluation_runs" / run["id"]
    assert (trash / "artifact" / "predictions.json").read_text(encoding="utf-8") == "[]"
    assert (trash / "workspace" / "request.json").read_text(encoding="utf-8") == "{}"
    assert (trash / "evaluation.json").is_file()


def test_model_metadata_migration_preserves_legacy_rows() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE models (id TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO models (id) VALUES ('legacy_model')")

    Database._migrate_model_metadata(connection)

    row = connection.execute(
        "SELECT * FROM models WHERE id='legacy_model'"
    ).fetchone()
    assert row["architecture"] == "未记录"
    assert row["detector_head"] == "未记录"
    assert row["class_count"] == 0
    assert row["training_dataset"] == "未记录"
    assert row["pretrained_dataset"] == "未记录"
    assert row["categories"] == "[]"
    assert row["category_template"] == "unconfigured"
    connection.close()


def test_category_templates_keep_dataset_and_model_id_spaces() -> None:
    templates = {item["id"]: item for item in list_category_templates()}
    assert set(templates) == {"coco2017", "visdrone", "voc"}
    assert len(templates["coco2017"]["categories"]) == 80
    coco_car = next(
        item for item in templates["coco2017"]["categories"]
        if item["name"] == "car"
    )
    assert coco_car == {"name": "car", "dataset_id": 3, "model_id": 2}
    visdrone_car = next(
        item for item in templates["visdrone"]["categories"]
        if item["name"] == "car"
    )
    assert visdrone_car == {"name": "car", "dataset_id": 3, "model_id": 3}
    assert templates["voc"]["categories"][0] == {
        "name": "aeroplane", "dataset_id": 0, "model_id": 0,
    }
    assert templates["voc"]["categories"][-1] == {
        "name": "tvmonitor", "dataset_id": 19, "model_id": 19,
    }


def test_category_compatibility_maps_ids_and_rejects_name_mismatch(
    tmp_path: Path,
) -> None:
    database, _ = make_database(tmp_path)
    compatible = category_compatibility(
        "dataset_aerial_clean", "model_yolov5s_demo", database
    )
    assert compatible["compatible"]
    assert compatible["model_to_dataset"] == {
        "0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    }
    database.execute(
        "UPDATE models SET categories=? WHERE id='model_yolov5s_demo'",
        (json_dump([{"id": 0, "name": "person"}]),),
    )
    with pytest.raises(CategoryCompatibilityError, match="模型缺少"):
        validate_evaluation_categories(
            ["dataset_aerial_clean"], ["model_yolov5s_demo"], database
        )
    database.execute(
        "UPDATE models SET categories=? WHERE id='model_yolov5s_demo'",
        (json_dump([{"id": 9, "name": "car"}]),),
    )
    subset = validate_evaluation_categories(
        ["dataset_aerial_clean"],
        ["model_yolov5s_demo"],
        database,
        evaluation_categories=["car"],
    )[0]
    assert subset["model_to_dataset"] == {"9": 1}
    assert subset["evaluation_categories"] == ["car"]
    assert subset["evaluation_category_ids"] == [1]


def test_auto_annotation_runs_registered_detector_and_preserves_existing_boxes(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    detector_script = tmp_path / "closed_set_detector.py"
    detector_script.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--annotations", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
manifest = json.load(open(args.annotations, encoding="utf-8"))
predictions = []
for image in manifest["images"]:
    predictions.extend([
        {"image_id": image["id"], "category_id": 9, "bbox": [-2, 3, 20, 14], "score": 0.91},
        {"image_id": image["id"], "category_id": 9, "bbox": [2, 2, 5, 5], "score": 0.1},
    ])
json.dump(predictions, open(args.output, "w", encoding="utf-8"))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    weight_path = tmp_path / "detector.pt"
    weight_path.write_bytes(b"closed-set-weight")
    artifact_directory = app_settings.artifact_dir / "custom" / "auto-label"
    artifact_directory.mkdir(parents=True)
    Image.new("RGB", (32, 24), (10, 20, 30)).save(
        artifact_directory / "new.jpg"
    )
    Image.new("RGB", (32, 24), (30, 20, 10)).save(
        artifact_directory / "manual.jpg"
    )
    now = utc_now()
    adapter_schema = {
        "type": "object",
        "properties": {},
        "execution": {
            "mode": "command",
            "working_directory": str(tmp_path),
            "executable": sys.executable,
            "arguments": [
                str(detector_script),
                "--annotations",
                "{annotation_path}",
                "--output",
                "{predictions_path}",
            ],
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
                "adapter_auto_test",
                "测试闭集检测模型",
                "DETECTOR",
                "v1",
                "REGISTERED",
                "platform",
                None,
                "read_only",
                sys.executable,
                0,
                "REGISTERED",
                "测试自动标注",
                json_dump(adapter_schema),
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO models
            (id,name,family,architecture,backbone,detector_head,class_count,
             categories,category_template,training_dataset,pretrained_dataset,
             version,precision,adapter_id,weight_path,weight_sha256,is_demo,
             status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "model_auto_test",
                "测试闭集检测模型",
                "TestDetector",
                "One-stage",
                "TestNet",
                "TestHead",
                1,
                json_dump([{"id": 9, "name": "car"}]),
                "custom",
                "test",
                "test",
                "v1",
                "FP32",
                "adapter_auto_test",
                str(weight_path),
                "test-sha",
                0,
                "REGISTERED",
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO datasets
            (id,name,version,source_type,scene_domain,weather,sensor_conditions,
             resolution,sample_count,annotation_status,frozen,artifact_path,
             category_template,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "dataset_auto_test",
                "自动标注测试集",
                "draft",
                "REAL",
                "无人机航拍",
                "晴朗",
                "{}",
                "32×24",
                2,
                "ANNOTATING",
                0,
                "custom/auto-label",
                "custom",
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO dataset_annotation_schemas(dataset_id,categories,updated_at)
            VALUES (?,?,?)
            """,
            (
                "dataset_auto_test",
                json_dump([{"id": 3, "name": "car", "color": "#1677FF"}]),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO sample_annotations
            (dataset_id,sample_name,width,height,boxes,completed,updated_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                "dataset_auto_test",
                "manual.jpg",
                32,
                24,
                json_dump(
                    [
                        {
                            "id": "manual_1",
                            "category_id": 3,
                            "x": 1,
                            "y": 1,
                            "width": 4,
                            "height": 4,
                        }
                    ]
                ),
                0,
                now,
            ),
        )

    job = queue_job(
        "AUTO_ANNOTATION",
        {
            "dataset_id": "dataset_auto_test",
            "model_id": "model_auto_test",
            "confidence": 0.25,
            "nms_iou": 0.7,
            "image_size": 1280,
            "max_detections": 300,
            "precision": "FP32",
        },
        database,
    )
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    result = json.loads(completed["result"])
    assert result["processed_images"] == 1
    assert result["annotated_images"] == 1
    assert result["accepted_boxes"] == 1
    generated = get_sample_annotation(
        "dataset_auto_test", "new.jpg", database
    )
    assert generated["completed"] is False
    assert generated["boxes"] == [
        {
            "id": generated["boxes"][0]["id"],
            "category_id": 3,
            "x": 0.0,
            "y": 3.0,
            "width": 18.0,
            "height": 14.0,
            "confidence": 0.91,
            "source": "AUTO_MODEL",
        }
    ]
    existing = get_sample_annotation(
        "dataset_auto_test", "manual.jpg", database
    )
    assert existing["boxes"][0]["id"] == "manual_1"


def test_local_detector_model_registration_uses_bounded_resources(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    environment_root = tmp_path / "envs"
    project = model_root / "custom-detector"
    runtime = environment_root / "custom"
    project.mkdir(parents=True)
    (runtime / "bin").mkdir(parents=True)
    (project / "platform_adapter.py").write_text(
        "print('adapter')\n",
        encoding="utf-8",
    )
    (project / "best.pt").write_bytes(b"test-model-weight")
    (runtime / "bin" / "python").write_text("", encoding="utf-8")
    (runtime / "bin" / "python").chmod(0o755)
    app_settings = Settings(
        root_dir=ROOT_DIR,
        data_dir=tmp_path / "data",
        model_library_root=model_root,
        model_environment_root=environment_root,
    )
    database = Database(app_settings)
    database.initialize()

    listing = list_local_model_resources(
        str(project),
        "model",
        "entrypoint",
        app_settings,
    )
    assert listing["current"] == str(project)
    assert [item["name"] for item in listing["entries"]] == [
        "platform_adapter.py"
    ]
    with pytest.raises(LocalModelRegistrationError, match="允许范围"):
        list_local_model_resources(
            str(tmp_path),
            "model",
            "directory",
            app_settings,
        )
    model = register_local_detector_model(
        {
            "name": "自定义检测模型",
            "family": "Custom",
            "architecture": "One-stage",
            "backbone": "CustomNet",
            "detector_head": "CustomHead",
            "category_template": "custom",
            "categories": [
                {"id": index, "name": name}
                for index, name in enumerate(
                    ("car", "truck", "bus", "pedestrian", "bicycle", "motorcycle")
                )
            ],
            "training_dataset": "Custom Detection v1",
            "pretrained_dataset": "ImageNet-1K",
            "version": "v1",
            "precision": "FP16",
            "project_directory": str(project),
            "working_directory": str(project),
            "runtime_prefix": str(runtime),
            "command_arguments": [
                "platform_adapter.py",
                "--weights",
                "{weight_path}",
                "--confidence",
                "{confidence}",
                "--input-height",
                "{input_height}",
                "--input-width",
                "{input_width}",
                "--output",
                "{predictions_path}",
            ],
            "inference_defaults": {
                "confidence": 0.3,
                "nms_iou": 0.7,
                "image_size": 1280,
                "input_height": 960,
                "input_width": 1280,
                "max_detections": 300,
                "batch_size": 1,
                "warmup": 0,
            },
            "weight_path": str(project / "best.pt"),
        },
        database,
    )
    assert model["weight_path"] == str(project / "best.pt")
    assert model["weight_sha256"]
    assert model["is_demo"] is False
    assert model["architecture"] == "One-stage"
    assert model["backbone"] == "CustomNet"
    assert model["detector_head"] == "CustomHead"
    assert model["class_count"] == 6
    assert model["training_dataset"] == "Custom Detection v1"
    assert model["pretrained_dataset"] == "ImageNet-1K"
    adapter = database.row(
        "SELECT * FROM adapters WHERE id=?", (model["adapter_id"],)
    )
    assert adapter["runtime_prefix"] == str(runtime)
    assert adapter["entrypoint"] == str(runtime / "bin" / "python")
    schema = json.loads(adapter["parameter_schema"])
    assert schema["execution"]["mode"] == "command"
    assert schema["execution"]["arguments"][-1] == "{predictions_path}"
    assert schema["properties"]["confidence"]["default"] == 0.3
    assert schema["properties"]["input_height"]["default"] == 960
    assert schema["properties"]["input_width"]["default"] == 1280
    assert "nms_iou" not in schema["properties"]
    alternate_runtime = environment_root / "alternate"
    (alternate_runtime / "bin").mkdir(parents=True)
    (alternate_runtime / "bin" / "python").write_text("", encoding="utf-8")
    (alternate_runtime / "bin" / "python").chmod(0o755)
    working_directory = project / "runtime"
    working_directory.mkdir()
    updated = update_local_detector_model(
        model["id"],
        {
            "name": "自定义检测模型（修订）",
            "architecture": "One-stage revised",
            "backbone": "CustomNet-v2",
            "detector_head": "CustomHead-v2",
            "training_dataset": "Custom Detection v2",
            "pretrained_dataset": "COCO",
            "precision": "FP32",
            "working_directory": str(working_directory),
            "runtime_prefix": str(alternate_runtime),
            "inference_defaults": {
                "confidence": 0.42,
                "nms_iou": 0.65,
                "image_size": 1024,
                "input_height": 1024,
                "input_width": 1024,
                "max_detections": 500,
                "batch_size": 2,
                "warmup": 3,
            },
        },
        database,
    )
    assert updated["id"] == model["id"]
    assert updated["name"] == "自定义检测模型（修订）"
    assert updated["family"] == "Custom"
    assert updated["version"] == "v1"
    assert updated["categories"] == model["categories"]
    assert updated["weight_path"] == model["weight_path"]
    adapter = database.row(
        "SELECT * FROM adapters WHERE id=?", (model["adapter_id"],)
    )
    assert adapter["runtime_prefix"] == str(alternate_runtime)
    assert adapter["entrypoint"] == str(alternate_runtime / "bin" / "python")
    schema = json.loads(adapter["parameter_schema"])
    assert schema["execution"]["working_directory"] == str(working_directory)
    assert schema["execution"]["arguments"][-1] == "{predictions_path}"
    assert schema["properties"]["confidence"]["default"] == 0.42
    assert schema["properties"]["input_height"]["default"] == 1024
    assert "nms_iou" not in schema["properties"]
    with pytest.raises(LocalModelRegistrationError, match="允许的环境根目录"):
        update_local_detector_model(
            model["id"],
            {
                "name": updated["name"],
                "architecture": updated["architecture"],
                "backbone": updated["backbone"],
                "detector_head": updated["detector_head"],
                "training_dataset": updated["training_dataset"],
                "pretrained_dataset": updated["pretrained_dataset"],
                "precision": "FP32",
                "working_directory": str(working_directory),
                "runtime_prefix": str(tmp_path),
                "inference_defaults": {
                    "confidence": 0.4,
                    "nms_iou": 0.7,
                    "image_size": 1024,
                    "input_height": 1024,
                    "input_width": 1024,
                    "max_detections": 300,
                    "batch_size": 1,
                    "warmup": 0,
                },
            },
            database,
        )
    with pytest.raises(CommandTemplateError, match="不支持的命令占位符"):
        validate_command_arguments(["--output", "{unknown_path}"])
    assert command_placeholders(["--confidence", "{confidence}"]) == {
        "confidence"
    }
    assert render_command(
        "/usr/bin/example",
        ["--confidence", "{confidence}"],
        {"confidence": 0.4},
    ) == ["/usr/bin/example", "--confidence", "0.4"]
    database.execute(
        """
        INSERT INTO model_profiles
        (id,model_id,weight_sha256,input_shape,scope,unsupported_ops,created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            "profile_delete_test",
            model["id"],
            model["weight_sha256"],
            "[1,3,960,1280]",
            "forward_only",
            "[]",
            utc_now(),
        ),
    )
    deleted = delete_model(model["id"], database)
    assert deleted["deleted"] is True
    assert deleted["adapter_deleted"] is True
    assert deleted["files_deleted"] is False
    assert (project / "best.pt").is_file()
    assert database.row(
        "SELECT id FROM models WHERE id=?", (model["id"],)
    ) is None
    assert database.row(
        "SELECT id FROM adapters WHERE id=?", (model["adapter_id"],)
    ) is None
    assert database.row(
        "SELECT id FROM model_profiles WHERE model_id=?", (model["id"],)
    ) is None


def test_known_detector_commands_are_upgraded_to_benchmark_protocol(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    arguments = JobAgent(database, app_settings)._benchmark_command_arguments(
        [
            "tools/inference/platform_coco.py",
            "--weights",
            "{weight_path}",
            "--output",
            "{predictions_path}",
        ],
        Path("/models/ultralytics-visdrone"),
    )
    assert arguments[0] == str(
        app_settings.root_dir / "adapters" / "detector_benchmark.py"
    )
    assert arguments[arguments.index("--backend") + 1] == "ultralytics"
    assert "{request_path}" in arguments
    assert "{result_path}" in arguments
    assert "{warmup}" in arguments


def test_local_dataset_resources_are_bounded_and_validate_import_paths(
    tmp_path: Path,
) -> None:
    _, app_settings = make_database(tmp_path)
    image_directory = tmp_path / "library" / "images"
    annotation_file = tmp_path / "library" / "instances.json"
    image_directory.mkdir(parents=True)
    Image.new("RGB", (32, 24), (30, 90, 150)).save(
        image_directory / "sample.jpg"
    )
    annotation_file.write_text("{}", encoding="utf-8")
    (tmp_path / "library" / "notes.txt").write_text("ignored", encoding="utf-8")

    listing = list_local_dataset_resources(
        str(tmp_path / "library"),
        "annotation",
        app_settings,
    )
    assert [item["name"] for item in listing["entries"]] == [
        "images",
        "instances.json",
    ]
    source, annotation = resolve_local_dataset_import(
        {
            "directory": str(image_directory),
            "annotation_path": str(annotation_file),
            "annotation_format": "COCO",
        },
        app_settings,
    )
    assert source == image_directory
    assert annotation == annotation_file

    with pytest.raises(DatasetImportError, match="允许范围"):
        list_local_dataset_resources(
            str(tmp_path.parent),
            "directory",
            app_settings,
        )
    with pytest.raises(DatasetImportError, match="必须位于"):
        resolve_local_dataset_import(
            {"directory": str(tmp_path.parent)},
            app_settings,
        )


def test_annotated_dataset_categories_are_read_from_annotation_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, app_settings = make_database(tmp_path)
    image_directory = tmp_path / "annotated-images"
    image_directory.mkdir()
    Image.new("RGB", (32, 24), (30, 90, 150)).save(
        image_directory / "sample.png"
    )
    coco_path = tmp_path / "instances.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.png"}],
                "annotations": [],
                "categories": [
                    {"id": 3, "name": "car"},
                    {"id": 8, "name": "truck"},
                ],
            }
        ),
        encoding="utf-8",
    )
    detected = read_dataset_annotation_categories(
        coco_path,
        "COCO",
        app_settings,
    )
    assert detected["categories"] == [
        {"id": 3, "name": "car"},
        {"id": 8, "name": "truck"},
    ]

    yolo_directory = tmp_path / "yolo-labels"
    yolo_directory.mkdir()
    (yolo_directory / "data.yaml").write_text(
        "names:\n  0: pedestrian\n  1: car\n",
        encoding="utf-8",
    )
    detected = read_dataset_annotation_categories(
        yolo_directory,
        "YOLO",
        app_settings,
    )
    assert detected["categories"] == [
        {"id": 0, "name": "pedestrian"},
        {"id": 1, "name": "car"},
    ]

    captured: dict[str, object] = {}

    def capture_job(job_type: str, payload: dict[str, object]) -> dict[str, str]:
        captured["job_type"] = job_type
        captured["payload"] = payload
        return {"id": "job_annotation_categories", "status": "QUEUED"}

    monkeypatch.setattr(main_module, "settings", app_settings)
    monkeypatch.setattr(main_module, "queue_job", capture_job)
    main_module.import_dataset(
        DatasetImportRequest(
            name="类别自动读取",
            directory=str(image_directory),
            annotation_path=str(coco_path),
            annotation_format="COCO",
            scene_domain="无人机航拍",
            category_template="custom",
            categories=[{"id": 0, "name": "wrong"}],
        )
    )
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["categories"] == [
        {"id": 3, "name": "car"},
        {"id": 8, "name": "truck"},
    ]
    assert payload["category_template"] == "annotation"
    assert payload["nonideal_condition"] == "无"


def test_local_dataset_import_references_images_without_copying_source(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    source_directory = tmp_path / "reference-source"
    source_directory.mkdir()
    source_image = source_directory / "sample.png"
    Image.new("RGB", (32, 24), (30, 90, 150)).save(source_image)

    job = queue_job(
        "DATASET_IMPORT",
        {
            "name": "引用模式数据集",
            "directory": str(source_directory),
            "annotation_path": None,
            "scene_domain": "无人机航拍",
            "nonideal_condition": "无人机气雾",
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    dataset = database.row("SELECT * FROM datasets WHERE name='引用模式数据集'")
    assert dataset["source_path"] == str(source_directory)
    assert json.loads(dataset["sensor_conditions"]) == {
        "recorded_condition": "无人机气雾"
    }
    linked_image = (
        app_settings.artifact_dir / dataset["artifact_path"] / "sample.png"
    )
    assert linked_image.is_symlink()
    assert linked_image.resolve() == source_image
    database.execute(
        "UPDATE datasets SET source_path=NULL WHERE id=?",
        (dataset["id"],),
    )
    listed = next(item for item in list_datasets(database) if item["id"] == dataset["id"])
    assert listed["dataset_path"] == str(source_directory)
    assert listed["platform_path"] == str(linked_image.parent)

    result = delete_dataset(dataset["id"], database)
    assert result and result["deleted"] is True
    assert source_image.is_file()


def test_local_dataset_import_preserves_duplicate_names_in_subdirectories(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    source_directory = tmp_path / "nested-source"
    for folder, color in (("A", (180, 40, 40)), ("B", (40, 80, 180))):
        directory = source_directory / folder
        directory.mkdir(parents=True)
        Image.new("RGB", (32, 24), color).save(directory / "sample.png")
    annotation_path = tmp_path / "nested-instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "A/sample.png", "width": 32, "height": 24},
                    {"id": 2, "file_name": "B/sample.png", "width": 32, "height": 24},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 8, 6]},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [4, 5, 10, 7]},
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    job = queue_job(
        "DATASET_IMPORT",
        {
            "name": "子目录同名图像",
            "directory": str(source_directory),
            "annotation_path": str(annotation_path),
            "annotation_format": "COCO",
            "scene_domain": "无人机航拍",
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )

    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    dataset = database.row("SELECT * FROM datasets WHERE name='子目录同名图像'")
    artifact = app_settings.artifact_dir / dataset["artifact_path"]
    assert (artifact / "A/sample.png").is_symlink()
    assert (artifact / "B/sample.png").is_symlink()
    page = list_dataset_samples(dataset["id"], 0, 50, database)
    assert [item["name"] for item in page["items"]] == [
        "A/sample.png",
        "B/sample.png",
    ]
    assert [item["boxes"][0]["x"] for item in page["items"]] == [
        pytest.approx(2 / 32),
        pytest.approx(4 / 32),
    ]
    first = get_sample_annotation(dataset["id"], "A/sample.png", database)
    second = get_sample_annotation(dataset["id"], "B/sample.png", database)
    assert first and first["boxes"][0]["x"] == pytest.approx(2)
    assert second and second["boxes"][0]["x"] == pytest.approx(4)


def test_model_deletion_rejects_historical_references(
    tmp_path: Path,
) -> None:
    database, _ = make_database(tmp_path)
    with pytest.raises(ModelDeletionError, match="历史运行"):
        delete_model("model_yolov5s_demo", database)
    assert database.row(
        "SELECT id FROM models WHERE id='model_yolov5s_demo'"
    )


def test_structured_detector_command_only_requires_coco_predictions(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    project = model_root / "command-detector"
    project.mkdir(parents=True)
    script = project / "evaluate.py"
    script.write_text(
        "\n".join(
            [
                "import argparse, json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--annotation', required=True)",
                "parser.add_argument('--output', required=True)",
                "parser.add_argument('--confidence', type=float, required=True)",
                "args = parser.parse_args()",
                "ground_truth = json.loads(Path(args.annotation).read_text())",
                "annotation = ground_truth['annotations'][0]",
                "prediction = {",
                "    'image_id': annotation['image_id'],",
                "    'category_id': annotation['category_id'],",
                "    'bbox': annotation['bbox'],",
                "    'score': 0.99,",
                "}",
                "Path(args.output).write_text(json.dumps([prediction]))",
            ]
        ),
        encoding="utf-8",
    )
    weight = project / "model.pt"
    weight.write_bytes(b"command-model-weight")
    runtime_prefix = Path(sys.executable).parent.parent
    app_settings = Settings(
        root_dir=ROOT_DIR,
        data_dir=tmp_path / "data",
        model_library_root=model_root,
        model_environment_root=runtime_prefix.parent,
    )
    database = Database(app_settings)
    database.initialize()
    model = register_local_detector_model(
        {
            "name": "命令检测模型",
            "family": "Command",
            "architecture": "One-stage",
            "backbone": "Test",
            "detector_head": "TestHead",
            "category_template": "custom",
            "categories": [{"id": 0, "name": "car"}],
            "training_dataset": "Test Detection",
            "pretrained_dataset": "无",
            "version": "v1",
            "precision": "FP16",
            "project_directory": str(project),
            "working_directory": str(project),
            "runtime_prefix": str(runtime_prefix),
            "command_arguments": [
                "evaluate.py",
                "--annotation",
                "{annotation_path}",
                "--output",
                "{predictions_path}",
                "--confidence",
                "{confidence}",
            ],
            "weight_path": str(weight),
        },
        database,
    )

    dataset_directory = app_settings.artifact_dir / "command-dataset"
    annotation_directory = dataset_directory / "annotations"
    annotation_directory.mkdir(parents=True)
    Image.new("RGB", (100, 80), (80, 120, 160)).save(
        dataset_directory / "sample.png"
    )
    (annotation_directory / "instances.json").write_text(
        json.dumps(
            {
                "info": {"description": "command detector"},
                "images": [
                    {
                        "id": 1,
                        "file_name": "sample.png",
                        "width": 100,
                        "height": 80,
                    }
                ],
                "categories": [{"id": 1, "name": "car"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 30, 20],
                        "area": 600,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,
         resolution,sample_count,annotation_status,frozen,artifact_path,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_command_detector",
            "命令检测数据集",
            "v1",
            "REAL",
            "无人机航拍",
            "晴朗",
            "{}",
            "100×80",
            1,
            "VERIFIED",
            0,
            "command-dataset",
            utc_now(),
        ),
    )
    update_annotation_schema(
        "dataset_command_detector",
        [{"id": 1, "name": "car", "color": "#1677FF"}],
        database,
    )
    database.execute(
        "UPDATE datasets SET frozen=1 WHERE id='dataset_command_detector'"
    )
    database.execute(
        """
        INSERT INTO evaluation_plans
        (id,name,dataset_ids,model_ids,seeds,blur_levels,protocol,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "plan_command_detector",
            "命令检测评测",
            json_dump(["dataset_command_detector"]),
            json_dump([model["id"]]),
            json_dump([1001]),
            json_dump([0]),
            json_dump(
                {
                    "batch_size": 1,
                    "precision": "FP16",
                    "warmup": 0,
                    "confidence": 0.42,
                }
            ),
            utc_now(),
        ),
    )
    job = queue_job(
        "EVALUATION",
        {"plan_id": "plan_command_detector"},
        database,
    )
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    run = database.row(
        """
        SELECT runs.config,results.map,results.latency_p50,
               results.performance_status
        FROM runs JOIN results ON results.run_id=runs.id
        WHERE runs.job_id=?
        """,
        (job["id"],),
    )
    assert run["map"] == pytest.approx(1)
    assert run["latency_p50"] == 0
    assert run["performance_status"] == "UNAVAILABLE"
    config = json.loads(run["config"])
    assert config["category_mapping"] == {"0": 1}
    assert config["confidence"] == pytest.approx(0.42)
    assert "未生成 result.json" in config["warnings"][0]


def test_dronedets_category_mapping_and_coco_metrics(tmp_path: Path) -> None:
    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "motorcycle"},
    ]
    assert category_mapping(categories) == {
        "car": 1,
        "motorcycle": 2,
        "motor": 2,
    }
    annotation_path = tmp_path / "instances.json"
    predictions_path = tmp_path / "predictions.json"
    annotation_path.write_text(
        json.dumps(
            {
                "info": {"description": "metric test"},
                "images": [
                    {"id": 1, "file_name": "sample.png", "width": 100, "height": 80}
                ],
                "categories": categories,
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 30, 20],
                        "area": 600,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [55, 45, 20, 15],
                        "area": 300,
                        "iscrowd": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 30, 20],
                    "score": 0.99,
                }
            ]
        ),
        encoding="utf-8",
    )
    metrics, curves = evaluate_coco_predictions(
        annotation_path,
        predictions_path,
        {
            "preprocess_ms": [1],
            "inference_ms": [8],
            "postprocess_ms": [1],
            "peak_memory_mb": 1024,
        },
    )
    assert metrics["map"] == pytest.approx(0.5)
    assert metrics["map50"] == pytest.approx(0.5)
    assert metrics["latency_p50"] == 10
    assert metrics["fps"] == 100
    assert metrics["performance_status"] == "MEASURED"
    assert len(curves["recall"]) == len(curves["precision"]) == 101
    batched_metrics, _ = evaluate_coco_predictions(
        annotation_path,
        predictions_path,
        {
            "inference_ms": [5, 5],
            "end_to_end_ms": [10, 10],
            "batch_duration_ms": [15],
            "batch_image_counts": [2],
            "torch_peak_allocated_mb": 900,
            "torch_peak_reserved_mb": 1100,
            "nvml_process_peak_mb": 1200,
        },
    )
    assert batched_metrics["throughput_fps"] == pytest.approx(133.333)
    assert batched_metrics["peak_memory"] == 1200
    selected_metrics, _ = evaluate_coco_predictions(
        annotation_path,
        predictions_path,
        {},
        [1],
    )
    assert selected_metrics["map"] == pytest.approx(1)
    assert selected_metrics["map50"] == pytest.approx(1)
    assert selected_metrics["performance_status"] == "UNAVAILABLE"


def test_result_query_keeps_resolution_as_group_dimension(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    result = query_results(scene="无人机航拍", database=database)
    assert result["count"] == 18
    assert len(result["groups"]) == 6
    assert {group["resolution"] for group in result["groups"]} == {"1920×1080"}
    assert all(group["seed_count"] == 3 for group in result["groups"])
    assert {
        run["condition_type"]
        for run in result["runs"]
        if run["dataset_id"] == "dataset_aerial_clean"
    } == {"无"}
    assert {
        run["condition_type"]
        for run in result["runs"]
        if run["dataset_id"] == "dataset_aerial_blur"
    } == {"运动模糊"}


def test_result_query_does_not_merge_different_evaluation_categories(
    tmp_path: Path,
) -> None:
    database, _ = make_database(tmp_path)
    run = database.row(
        """
        SELECT id,config FROM runs
        WHERE dataset_id='dataset_aerial_clean'
          AND model_id='model_yolov5s_demo'
        LIMIT 1
        """
    )
    config = json.loads(run["config"])
    config["evaluation_categories"] = ["car"]
    database.execute(
        "UPDATE runs SET config=? WHERE id=?",
        (json_dump(config), run["id"]),
    )
    result = query_results(scene="无人机航拍", database=database)
    groups = [
        group for group in result["groups"]
        if group["dataset_id"] == "dataset_aerial_clean"
        and group["model_id"] == "model_yolov5s_demo"
    ]
    assert len(groups) == 2
    assert sorted(group["seed_count"] for group in groups) == [1, 2]
    assert sorted(len(group["evaluation_categories"]) for group in groups) == [1, 6]
    assert any(group["evaluation_categories"] == ["car"] for group in groups)


def test_result_query_does_not_merge_different_inference_configs(
    tmp_path: Path,
) -> None:
    database, _ = make_database(tmp_path)
    now = utc_now()
    database.execute(
        """
        INSERT INTO runs
        (id,plan_id,job_id,dataset_id,model_id,seed,status,config,
         environment_fingerprint,hardware_profile,created_at,finished_at)
        VALUES (?,?,?,?,?,?, 'SUCCEEDED', ?,?,?,?,?)
        """,
        (
            "run_different_resolution",
            None,
            None,
            "dataset_aerial_clean",
            "model_yolov5s_demo",
            2001,
            json_dump(
                {
                    "batch_size": 1,
                    "warmup": 20,
                    "blur_level": 0,
                    "image_size": 640,
                }
            ),
            "reference-demo-environment",
            json_dump({"device": "流程样例设备", "comparable": True}),
            now,
            now,
        ),
    )
    database.execute(
        """
        INSERT INTO results
        (id,run_id,map,map50,map75,precision,recall,f1,latency_p50,
         latency_p95,fps,peak_memory,delta_map,metrics,curves,is_official,
         created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "result_different_resolution",
            "run_different_resolution",
            0.75,
            0.85,
            0.70,
            0.80,
            0.78,
            0.79,
            9.0,
            11.0,
            111.11,
            1800.0,
            -0.04,
            json_dump({}),
            json_dump({"recall": [0, 1], "precision": [1, 0]}),
            0,
            now,
        ),
    )

    result = query_results(scene="无人机航拍", database=database)
    comparable = [
        group
        for group in result["groups"]
        if group["dataset_id"] == "dataset_aerial_clean"
        and group["model_id"] == "model_yolov5s_demo"
    ]
    assert len(comparable) == 2
    assert sorted(group["seed_count"] for group in comparable) == [1, 3]
    resized = next(
        group
        for group in comparable
        if group["inference_config"].get("image_size") == 640
    )
    assert resized["inference_config"]["input_resolution"] == "640×640"
    assert resized["map50_mean"] == pytest.approx(0.85)
    assert resized["latency_p95_mean"] == pytest.approx(11)


def test_external_conda_adapter_uses_registered_python(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    runtime_prefix = Path(sys.executable).parent.parent
    adapter_script = tmp_path / "external_generator.py"
    adapter_script.write_text(
        """
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request.read_text(encoding="utf-8"))
output = Path(request["output_directory"])
output.mkdir(parents=True, exist_ok=True)
samples = []
for index in range(int(request["sample_count"])):
    name = f"sample-{index + 1}.png"
    path = output / name
    path.write_bytes(b"test-image")
    samples.append({
        "image_path": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
args.result.write_text(json.dumps({
    "protocol_version": "1.0",
    "job_id": request["job_id"],
    "status": "succeeded",
    "samples": samples,
}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    now = utc_now()
    database.execute(
        """
        INSERT INTO adapters
        (id,name,kind,version,maturity,runtime_kind,runtime_prefix,policy,entrypoint,
         requires_gpu,status,description,parameter_schema,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "adapter_external_test",
            "外部环境测试",
            "GENERATOR",
            "1.0",
            "CONTRACT_OK",
            "conda_external",
            str(runtime_prefix),
            "read_only",
            str(adapter_script),
            0,
            "HEALTHY",
            "",
            json_dump({}),
            now,
            now,
        ),
    )
    job = queue_job(
        "ACQUISITION",
        {
            "name": "外部环境生成",
            "adapter_id": "adapter_external_test",
            "source_type": "GENERATIVE",
            "sample_count": 2,
            "seeds": [7],
            "conditions": {
                "scene": {"domain": "城市驾驶", "weather": "晴朗"},
                "sensor": {"resolution": "960×540"},
            },
            "model_parameters": {},
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"


def test_basegen_adapter_prepares_reproducible_batch(tmp_path: Path) -> None:
    request = {
        "protocol_version": "1.0",
        "sample_count": 3,
        "seeds": [1001],
        "output_directory": str(tmp_path),
        "conditions": {
            "scene": {"domain": "无人机航拍", "weather": "雾"},
            "sensor": {"resolution": "1024×1024"},
        },
        "model_parameters": {"steps": 9, "device_policy": "cuda"},
    }
    plan, config = prepare_plan(request, ROOT_DIR.parent / "BaseGen")
    assert [item["seed"] for item in plan] == [1001, 1002, 1003]
    assert {item["domain"] for item in plan} == {"low-altitude-uav"}
    assert {item["scene"]["weather"] for item in plan} == {"fog"}
    assert {(item["width"], item["height"]) for item in plan} == {(1024, 1024)}
    assert config["model_path"] == "Tongyi-MAI/Z-Image-Turbo"


def test_basegen_resolves_cached_snapshot_without_hub_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = tmp_path / "hub"
    repository = hub / "models--Tongyi-MAI--Z-Image-Turbo"
    snapshot = repository / "snapshots" / "cached-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text(
        "cached-revision\n", encoding="utf-8"
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    assert resolve_cached_model_path("Tongyi-MAI/Z-Image-Turbo") == str(
        snapshot.resolve()
    )


def test_basegen_keeps_repo_id_when_cached_snapshot_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "hub" / "models--Tongyi-MAI--Z-Image-Turbo"
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text(
        "missing-revision\n", encoding="utf-8"
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    assert resolve_cached_model_path("Tongyi-MAI/Z-Image-Turbo") == (
        "Tongyi-MAI/Z-Image-Turbo"
    )


def test_basegen_scene_schema_exposes_domain_specific_ui_fields() -> None:
    schema = get_basegen_scene_schema()
    assert [domain["value"] for domain in schema["domains"]] == [
        "autonomous-driving",
        "low-altitude-uav",
        "offroad-autonomous-driving",
    ]
    driving = next(
        domain
        for domain in schema["domains"]
        if domain["value"] == "autonomous-driving"
    )
    assert driving["label_zh"] == "自动驾驶"
    uav = next(
        domain
        for domain in schema["domains"]
        if domain["value"] == "low-altitude-uav"
    )
    assert uav["label_zh"] == "低空无人机"
    assert {field["name"] for field in uav["fields"]} >= {
        "region",
        "camera_height",
        "viewpoint",
        "field_of_view",
        "environment",
        "weather",
        "elements",
    }
    assert "prompt_en" not in json_dump(schema)


def test_basegen_mixed_fixed_and_random_fields_obey_constraints(
    tmp_path: Path,
) -> None:
    request = {
        "protocol_version": "1.0",
        "sample_count": 4,
        "seeds": [2201],
        "output_directory": str(tmp_path),
        "conditions": {
            "scene": {
                "domain": "low-altitude-uav",
                "fields": {
                    "region": {"mode": "fixed", "value": "north_china"},
                    "camera_height": {
                        "mode": "fixed",
                        "value": "ultra_low",
                    },
                    "environment": {"mode": "random"},
                    "time_of_day": {"mode": "fixed", "value": "night"},
                    "elements": {
                        "mode": "fixed",
                        "values": ["warehouses"],
                    },
                },
                "custom": "Blue delivery trucks are visible",
            },
            "sensor": {"resolution": "1024×1024"},
        },
        "model_parameters": {"steps": 9, "device_policy": "cuda"},
    }
    first, _ = prepare_plan(request, ROOT_DIR.parent / "BaseGen")
    second, _ = prepare_plan(request, ROOT_DIR.parent / "BaseGen")
    assert [item["scene"] for item in first] == [
        item["scene"] for item in second
    ]
    assert {item["scene"]["environment"] for item in first} == {
        "suburban_district"
    }
    assert {item["scene"]["camera_height"] for item in first} == {
        "ultra_low"
    }
    assert {item["scene"]["region"] for item in first} == {"north_china"}
    assert all(item["scene"]["elements"] == ["warehouses"] for item in first)
    assert all(
        item["scene"]["custom"] == "Blue delivery trucks are visible"
        for item in first
    )


def test_basegen_rejects_incompatible_fixed_scene_fields(tmp_path: Path) -> None:
    request = {
        "protocol_version": "1.0",
        "sample_count": 1,
        "seeds": [42],
        "output_directory": str(tmp_path),
        "conditions": {
            "scene": {
                "domain": "low-altitude-uav",
                "fields": {
                    "environment": {
                        "mode": "fixed",
                        "value": "farmland",
                    },
                    "elements": {
                        "mode": "fixed",
                        "values": ["warehouses"],
                    },
                },
            },
            "sensor": {"resolution": "1024×1024"},
        },
        "model_parameters": {},
    }
    with pytest.raises(ValueError, match="不兼容"):
        prepare_plan(request, ROOT_DIR.parent / "BaseGen")


def test_basegen_random_fields_work_for_every_domain(tmp_path: Path) -> None:
    for domain in get_basegen_scene_schema()["domains"]:
        fields = {
            field["name"]: {"mode": "random"}
            for field in domain["fields"]
            if field["kind"] != "text"
        }
        plan, _ = prepare_plan(
            {
                "protocol_version": "1.0",
                "sample_count": 6,
                "seeds": [2901],
                "output_directory": str(tmp_path),
                "conditions": {
                    "scene": {
                        "domain": domain["value"],
                        "fields": fields,
                    },
                    "sensor": {
                        "resolution": domain["default_resolution"],
                    },
                },
                "model_parameters": {},
            },
            ROOT_DIR.parent / "BaseGen",
        )
        assert len(plan) == 6
        assert {item["scene"]["domain"] for item in plan} == {
            domain["value"]
        }


def test_basegen_preview_resolves_three_scenes_without_generation() -> None:
    preview = preview_basegen_plan(
        {
            "sample_count": 8,
            "seeds": [3101],
            "conditions": {
                "scene": {
                    "domain": "autonomous-driving",
                    "fields": {
                        "viewpoint": {
                            "mode": "fixed",
                            "value": "front_vehicle",
                        },
                        "weather": {"mode": "random"},
                        "elements": {"mode": "random"},
                    },
                },
                "sensor": {"resolution": "1024×576"},
            },
            "model_parameters": {"steps": 9, "device_policy": "cuda"},
        }
    )
    assert len(preview["images"]) == 3
    assert [item["seed"] for item in preview["images"]] == [
        3101,
        3102,
        3103,
    ]
    assert {
        item["scene"]["viewpoint"] for item in preview["images"]
    } == {"front_vehicle"}
    assert all(item["prompt"] for item in preview["images"])


def test_delete_frozen_dataset_moves_artifacts_to_trash(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    artifact_directory = app_settings.artifact_dir / "imports" / "delete-test"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "sample.jpg").write_bytes(b"sample")
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
         sample_count,annotation_status,frozen,artifact_path,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_delete_test",
            "待删除数据集",
            "v1",
            "REAL",
            "无人机航拍",
            "晴朗",
            "{}",
            "原始分辨率",
            1,
            "UNLABELED",
            1,
            "imports/delete-test",
            utc_now(),
        ),
    )
    result = delete_dataset("dataset_delete_test", database)
    assert result and result["deleted"]
    assert database.row(
        "SELECT id FROM datasets WHERE id='dataset_delete_test'"
    ) is None
    assert not artifact_directory.exists()
    trash = app_settings.data_dir / "trash" / "datasets" / "dataset_delete_test"
    assert (trash / "artifact" / "sample.jpg").read_bytes() == b"sample"
    assert (trash / "dataset.json").is_file()


def test_dataset_samples_are_paginated_from_artifact_directory(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    artifact_directory = app_settings.artifact_dir / "imports" / "browse-test"
    artifact_directory.mkdir(parents=True)
    for index in range(5):
        (artifact_directory / f"sample-{index}.jpg").write_bytes(b"sample")
    (artifact_directory / "metadata.json").write_text("{}", encoding="utf-8")
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
         sample_count,annotation_status,frozen,artifact_path,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_browse_test",
            "分页浏览数据集",
            "v1",
            "REAL",
            "无人机航拍",
            "晴朗",
            "{}",
            "原始分辨率",
            5,
            "UNLABELED",
            0,
            "imports/browse-test",
            utc_now(),
        ),
    )
    page = list_dataset_samples("dataset_browse_test", 1, 2, database)
    assert page
    assert page["total"] == 5
    assert page["declared_count"] == 5
    assert page["has_more"]
    assert [item["name"] for item in page["items"]] == [
        "sample-1.jpg",
        "sample-2.jpg",
    ]
    assert page["items"][0]["url"].endswith(
        "/imports/browse-test/sample-1.jpg"
    )
    assert list_dataset_samples("missing", 0, 48, database) is None


def test_dataset_browser_auto_detects_visdrone_annotations(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    artifact = app_settings.artifact_dir / "imports" / "visdrone-browser"
    label_directory = artifact / "annotations" / "yolo"
    label_directory.mkdir(parents=True)
    Image.new("RGB", (100, 80), (38, 120, 180)).save(
        artifact / "sample.jpg"
    )
    (label_directory / "sample.txt").write_text(
        "10,20,30,40,1,4,0,0\n",
        encoding="utf-8",
    )
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,
         resolution,sample_count,annotation_status,frozen,artifact_path,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_visdrone_browser",
            "VisDrone浏览测试",
            "v1",
            "REAL",
            "无人机航拍",
            "晴朗",
            "{}",
            "100×80",
            1,
            "CANDIDATE",
            0,
            "imports/visdrone-browser",
            utc_now(),
        ),
    )
    page = list_dataset_samples(
        "dataset_visdrone_browser", 0, 48, database
    )
    assert page
    item = page["items"][0]
    assert item["annotation_source"] == "VISDRONE"
    assert item["boxes"][0] == {
        "label": "car",
        "color": "#EB2F96",
        "x": pytest.approx(0.1),
        "y": pytest.approx(0.25),
        "width": pytest.approx(0.3),
        "height": pytest.approx(0.5),
    }


def test_detection_annotations_persist_and_export_coco(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    artifact_directory = app_settings.artifact_dir / "imports" / "annotation-test"
    artifact_directory.mkdir(parents=True)
    for name in ("first.png", "second.png"):
        Image.new("RGB", (64, 40), (38, 120, 180)).save(
            artifact_directory / name
        )
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,resolution,
         sample_count,annotation_status,frozen,artifact_path,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_annotation_test",
            "目标检测标注数据集",
            "v1",
            "GENERATIVE",
            "低空无人机",
            "随机",
            "{}",
            "64×40",
            2,
            "UNLABELED",
            0,
            "imports/annotation-test",
            utc_now(),
        ),
    )
    session = get_annotation_session("dataset_annotation_test", database)
    assert session
    assert session["progress"] == {"completed": 0, "total": 2}
    assert session["categories"] == []

    categories = [
        {"id": 1, "name": "vehicle", "color": "#1677FF"},
        {"id": 2, "name": "person", "color": "#EB2F96"},
    ]
    update_annotation_schema("dataset_annotation_test", categories, database)
    first = save_sample_annotation(
        "dataset_annotation_test",
        "first.png",
        {
            "width": 64,
            "height": 40,
            "boxes": [
                {
                    "id": "box_1",
                    "category_id": 1,
                    "x": 10,
                    "y": 5,
                    "width": 20,
                    "height": 15,
                }
            ],
            "completed": True,
        },
        database,
    )
    assert first and first["completed"]
    assert get_sample_annotation(
        "dataset_annotation_test", "first.png", database
    )["boxes"][0]["category_id"] == 1
    browse_page = list_dataset_samples(
        "dataset_annotation_test", 0, 48, database
    )
    assert browse_page
    first_visualization = next(
        item
        for item in browse_page["items"]
        if item["name"] == "first.png"
    )
    assert first_visualization["annotation_source"] == "MANUAL"
    assert first_visualization["boxes"][0] == {
        "label": "vehicle",
        "color": "#1677FF",
        "x": pytest.approx(10 / 64),
        "y": pytest.approx(5 / 40),
        "width": pytest.approx(20 / 64),
        "height": pytest.approx(15 / 40),
    }
    statistics = dataset_statistics("dataset_annotation_test", database)
    assert statistics
    assert statistics["image_count"] == 2
    assert statistics["annotated_image_count"] == 1
    assert statistics["object_count"] == 1
    assert statistics["category_counts"] == [
        {"id": 1, "name": "vehicle", "count": 1},
        {"id": 2, "name": "person", "count": 0},
    ]
    assert statistics["resolutions"] == [
        {"width": 64, "height": 40, "label": "64×40", "count": 2}
    ]
    assert [item["count"] for item in statistics["scales"]] == [1, 0, 0, 0]
    assert [item["count"] for item in statistics["relative_scales"]] == [
        0,
        0,
        0,
        1,
        0,
    ]
    assert database.row(
        "SELECT annotation_status FROM datasets WHERE id='dataset_annotation_test'"
    )["annotation_status"] == "ANNOTATING"

    with pytest.raises(DatasetAnnotationError, match="已被目标框使用"):
        update_annotation_schema(
            "dataset_annotation_test",
            [categories[1]],
            database,
        )
    with pytest.raises(DatasetAnnotationError, match="1 张"):
        complete_dataset_annotations("dataset_annotation_test", database)

    save_sample_annotation(
        "dataset_annotation_test",
        "second.png",
        {
            "width": 64,
            "height": 40,
            "boxes": [],
            "completed": True,
        },
        database,
    )
    result = complete_dataset_annotations("dataset_annotation_test", database)
    assert result
    assert result["images"] == 2
    assert result["annotations"] == 1
    coco = json.loads(
        (artifact_directory / "annotations" / "instances.json").read_text(
            encoding="utf-8"
        )
    )
    assert [image["file_name"] for image in coco["images"]] == [
        "first.png",
        "second.png",
    ]
    assert coco["annotations"][0]["bbox"] == [10, 5, 20, 15]
    assert coco["categories"] == [
        {"id": 1, "name": "vehicle"},
        {"id": 2, "name": "person"},
    ]
    assert database.row(
        "SELECT annotation_status FROM datasets WHERE id='dataset_annotation_test'"
    )["annotation_status"] == "CANDIDATE"

    database.execute(
        "UPDATE datasets SET frozen=1 WHERE id='dataset_annotation_test'"
    )
    with pytest.raises(DatasetAnnotationError, match="冻结"):
        save_sample_annotation(
            "dataset_annotation_test",
            "second.png",
            {"width": 64, "height": 40, "boxes": [], "completed": True},
            database,
        )


def test_delete_dataset_rejects_run_referenced_data(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    with pytest.raises(DatasetDeletionError, match="评测运行"):
        delete_dataset("dataset_aerial_clean", database)
    assert database.row(
        "SELECT id FROM datasets WHERE id='dataset_aerial_clean'"
    )


def test_delete_dataset_cleans_evaluation_plan_references(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    database.execute("DELETE FROM results")
    database.execute("DELETE FROM runs")
    database.execute(
        """
        INSERT INTO evaluation_plans
        (id,name,dataset_ids,model_ids,seeds,blur_levels,protocol,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "plan_delete_guard",
            "删除保护",
            json_dump(["dataset_aerial_clean"]),
            json_dump(["model_yolov5s_demo"]),
            json_dump([1001]),
            json_dump([0]),
            json_dump({}),
            utc_now(),
        ),
    )
    database.execute(
        """
        INSERT INTO evaluation_plans
        (id,name,dataset_ids,model_ids,seeds,blur_levels,protocol,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "plan_keep_other_dataset",
            "保留其他数据集",
            json_dump(["dataset_aerial_clean", "dataset_aerial_blur"]),
            json_dump(["model_yolov5s_demo"]),
            json_dump([1001]),
            json_dump([0]),
            json_dump({}),
            utc_now(),
        ),
    )
    job = queue_job("EVALUATION", {"plan_id": "plan_delete_guard"}, database)
    with pytest.raises(DatasetDeletionError, match="评测任务"):
        delete_dataset("dataset_aerial_clean", database)
    database.execute(
        "UPDATE jobs SET status='CANCELLED' WHERE id=?", (job["id"],)
    )

    deleted = delete_dataset("dataset_aerial_clean", database)

    assert deleted and deleted["deleted"]
    assert deleted["evaluation_plans_updated"] == 2
    assert database.row(
        "SELECT id FROM evaluation_plans WHERE id='plan_delete_guard'"
    ) is None
    retained_plan = database.row(
        "SELECT dataset_ids FROM evaluation_plans WHERE id='plan_keep_other_dataset'"
    )
    assert json.loads(retained_plan["dataset_ids"]) == ["dataset_aerial_blur"]


def test_evaluation_rejects_unfrozen_dataset(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    database.execute("UPDATE datasets SET frozen=0 WHERE id='dataset_aerial_clean'")
    database.execute(
        """
        INSERT INTO evaluation_plans(id,name,dataset_ids,model_ids,seeds,blur_levels,protocol,created_at)
        VALUES ('plan_test','test','[\"dataset_aerial_clean\"]','[\"model_yolov5s_demo\"]','[1001]','[0]','{}','now')
        """
    )
    job = queue_job("EVALUATION", {"plan_id": "plan_test"}, database)
    JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "FAILED"
    assert "尚未冻结" in completed["error"]


def test_real_detector_evaluation_converts_visdrone_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, app_settings = make_database(tmp_path)
    dataset_directory = app_settings.artifact_dir / "real-detector"
    annotation_directory = dataset_directory / "annotations" / "yolo"
    annotation_directory.mkdir(parents=True)
    Image.new("RGB", (100, 80), (80, 120, 160)).save(
        dataset_directory / "sample.png"
    )
    (annotation_directory / "sample.txt").write_text(
        "10,10,30,20,1,4,0,0\n",
        encoding="utf-8",
    )
    database.execute(
        """
        INSERT INTO datasets
        (id,name,version,source_type,scene_domain,weather,sensor_conditions,
         resolution,sample_count,annotation_status,frozen,artifact_path,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "dataset_real_detector",
            "真实检测评测集",
            "v1",
            "REAL",
            "无人机航拍",
            "晴朗",
            "{}",
            "100×80",
            1,
            "CANDIDATE",
            0,
            "real-detector",
            utc_now(),
        ),
    )
    update_annotation_schema(
        "dataset_real_detector",
        template_categories("visdrone", "dataset"),
        database,
    )
    database.execute(
        "UPDATE datasets SET frozen=1 WHERE id='dataset_real_detector'"
    )
    fake_weight = tmp_path / "best.pt"
    fake_weight.write_bytes(b"test-only-weight")
    database.execute(
        """
        UPDATE adapters SET runtime_kind='platform',runtime_prefix=NULL
        WHERE id='adapter_dronedets_yolov8m'
        """
    )
    database.execute(
        """
        UPDATE models SET weight_path=?,weight_sha256='test-sha',status='EXPERIMENTAL',
                          categories=?,class_count=1
        WHERE id='model_dronedets_yolov8m_visdrone'
        """,
        (str(fake_weight), json_dump([{"id": 3, "name": "car"}])),
    )
    database.execute(
        """
        INSERT INTO evaluation_plans
        (id,name,dataset_ids,model_ids,seeds,blur_levels,protocol,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "plan_real_detector",
            "真实检测链路",
            json_dump(["dataset_real_detector"]),
            json_dump(["model_dronedets_yolov8m_visdrone"]),
            json_dump([1001]),
            json_dump([0]),
            json_dump(
                {
                    "batch_size": 1,
                    "precision": "FP16",
                    "warmup": 0,
                    "evaluation_categories": ["car"],
                }
            ),
            utc_now(),
        ),
    )
    job = queue_job("EVALUATION", {"plan_id": "plan_real_detector"}, database)
    agent = JobAgent(database, app_settings)

    def fake_adapter_process(
        job_id: str,
        command: list[str],
        job_directory: Path,
        environment: dict[str, str],
    ) -> tuple[int, str]:
        assert job_id == job["id"]
        assert environment["DRONEDETS_ROOT"].endswith("DroneDets")
        request_path = Path(command[command.index("--request") + 1])
        result_path = Path(command[command.index("--result") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        converted_annotation = Path(request["dataset"]["annotation_path"])
        assert converted_annotation != (
            dataset_directory / "annotations" / "instances.json"
        )
        converted = json.loads(
            converted_annotation.read_text(encoding="utf-8")
        )
        assert converted["info"]["source_format"] == "VisDrone"
        assert converted["annotations"][0]["category_id"] == 3
        output_directory = Path(request["output_directory"])
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "predictions.json").write_text(
            json.dumps(
                [
                    {
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [10, 10, 30, 20],
                        "score": 0.99,
                    }
                ]
            ),
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(
                {
                    "protocol_version": "1.0",
                    "job_id": request["job_id"],
                    "run_id": request["run_id"],
                    "status": "succeeded",
                    "predictions_path": "predictions.json",
                    "image_count": 1,
                    "runtime": {
                        "preprocess_ms": [1],
                        "inference_ms": [8],
                        "postprocess_ms": [1],
                        "peak_memory_mb": 512,
                    },
                    "complexity": {
                        "parameters_total": 1000,
                        "parameters_trainable": 900,
                        "input_shape": [1, 3, 960, 1280],
                        "macs": 2000,
                        "flops": 4000,
                        "scope": "forward_only",
                        "profiler": "test-profiler",
                        "unsupported_ops": [],
                    },
                    "environment": {"device": "test-gpu"},
                }
            ),
            encoding="utf-8",
        )
        return 0, ""

    monkeypatch.setattr(agent, "_run_adapter_process", fake_adapter_process)
    assert agent.process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    run = database.row(
        """
        SELECT runs.*,results.map,results.latency_p50,results.is_official,
               results.performance_status,results.inference_latency_p50
        FROM runs JOIN results ON results.run_id=runs.id
        WHERE runs.job_id=?
        """,
        (job["id"],),
    )
    assert run["model_id"] == "model_dronedets_yolov8m_visdrone"
    assert run["map"] == pytest.approx(1)
    assert run["latency_p50"] == 10
    assert run["inference_latency_p50"] == 8
    assert run["performance_status"] == "MEASURED"
    assert run["is_official"] == 0
    profile = database.row(
        "SELECT * FROM model_profiles WHERE model_id=?", (run["model_id"],)
    )
    assert profile["parameters_total"] == 1000
    assert profile["flops"] == 4000
    run_config = json.loads(run["config"])
    assert run_config["annotation_conversion"]["images"] == 1
    assert run_config["evaluation_categories"] == ["car"]
    assert not (dataset_directory / "annotations" / "instances.json").exists()
    visualization = evaluation_run_visualization(run["id"], 0, 12, database)
    assert visualization
    assert visualization["inference_confidence"] == pytest.approx(0.001)
    assert visualization["total"] == 1
    assert visualization["items"][0]["name"] == "sample.png"
    box = visualization["items"][0]["boxes"][0]
    assert box["label"] == "car"
    assert box["score"] == pytest.approx(0.99)
    assert box["x"] == pytest.approx(0.1)
    assert box["width"] == pytest.approx(0.3)


def test_local_import_can_feed_nonideal_condition_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, app_settings = make_database(tmp_path)
    source_directory = tmp_path / "source-images"
    source_directory.mkdir()
    Image.new("RGB", (64, 40), (38, 120, 180)).save(source_directory / "aerial.png")
    import_job = queue_job(
        "DATASET_IMPORT",
        {
            "name": "真实航拍导入",
            "directory": str(source_directory),
            "annotation_path": None,
            "scene_domain": "无人机航拍",
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    agent = JobAgent(database, app_settings)
    assert agent.process_one()
    imported = database.row("SELECT * FROM datasets WHERE name='真实航拍导入'")
    assert imported and imported["source_type"] == "REAL"
    assert imported["resolution"] == "64×40"
    imported_directory = app_settings.artifact_dir / imported["artifact_path"]
    imported_image = imported_directory / "aerial.png"
    assert imported_image.is_symlink()
    assert imported_image.resolve() == source_directory / "aerial.png"
    database.execute(
        "UPDATE datasets SET resolution='原始分辨率' WHERE id=?",
        (imported["id"],),
    )
    listed = next(
        item for item in list_datasets(database) if item["id"] == imported["id"]
    )
    assert listed["resolution"] == "64×40"
    assert database.row(
        "SELECT resolution FROM datasets WHERE id=?", (imported["id"],)
    )["resolution"] == "64×40"
    annotation_directory = imported_directory / "annotations"
    annotation_directory.mkdir()
    (annotation_directory / "instances.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "aerial.png", "width": 64, "height": 40}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 8, 20, 12],
                        "area": 240,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    database.execute(
        """
        UPDATE datasets
        SET annotation_status='CANDIDATE',source_type='GENERATIVE',scene_domain='低空无人机'
        WHERE id=?
        """,
        (imported["id"],),
    )

    captured: dict[str, object] = {}

    def fake_adapter_process(job_id, command, cwd, environment):
        request = json.loads((cwd / "request.json").read_text(encoding="utf-8"))
        captured.update({"command": command, "environment": environment, "request": request})
        output_directory = Path(request["output_directory"])
        output = output_directory / "aerial.png"
        with Image.open(request["input_images"][0]) as source:
            source.save(output)
        (cwd / "result.json").write_text(
            json_dump(
                {
                    "protocol_version": "1.0",
                    "job_id": job_id,
                    "status": "succeeded",
                    "samples": [
                        {
                            "sample_id": f"{job_id}-1",
                            "image_path": "aerial.png",
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "width": 64,
                            "height": 40,
                            "seed": 1001,
                            "annotation_status": "CANDIDATE",
                        }
                    ],
                    "has_candidate_annotations": True,
                    "runtime": (
                        {"matched_conditions": 1, "fallback_conditions": 0}
                        if request.get("model_parameters", {}).get(
                            "condition_directory"
                        )
                        else {}
                    ),
                }
            ),
            encoding="utf-8",
        )
        return 0, ""

    monkeypatch.setattr(agent, "_run_adapter_process", fake_adapter_process)
    condition_job = queue_job(
        "ACQUISITION",
        {
            "name": "真实航拍加雾",
            "adapter_id": "adapter_condition",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [1001],
            "input_dataset_id": imported["id"],
            "conditions": {
                "scene": {"domain": "无人机航拍", "weather": "雾"},
                "sensor": {"resolution": "原始分辨率"},
            },
            "model_parameters": {"fog_strength": 0.6},
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert agent.process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (condition_job["id"],))
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row("SELECT * FROM datasets WHERE name='真实航拍加雾'")
    assert transformed["source_type"] == "REAL_TRANSFORMED"
    assert transformed["scene_domain"] == "无人机航拍"
    assert transformed["weather"] == "雾"
    assert transformed["resolution"] == "64×40"
    assert transformed["annotation_status"] == "CANDIDATE"
    sensor = json.loads(transformed["sensor_conditions"])
    assert sensor["fog_strength"] == pytest.approx(0.6)
    assert sensor["condition_label"] == "无人机气雾"
    assert sensor["fog_model"] == "DiffusionDegrade · 无人机气雾"
    outputs = list((app_settings.artifact_dir / transformed["artifact_path"]).glob("*.png"))
    assert len(outputs) == 1
    assert (
        app_settings.artifact_dir
        / transformed["artifact_path"]
        / "annotations"
        / "instances.json"
    ).is_file()
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["sample_count"] == 1
    assert request["has_source_annotations"] is True
    assert request["model_parameters"]["checkpoint"] == "uav_fog_content15_model_2501"
    assert request["model_parameters"]["fog_strength"] == pytest.approx(0.6)
    assert str(captured["command"][0]).endswith("DiffusionDegrade/.venv/bin/python")
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"

    blur_job = queue_job(
        "ACQUISITION",
        {
            "name": "真实航拍运动模糊",
            "adapter_id": "adapter_motion_blur",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [2023],
            "input_dataset_id": imported["id"],
            "conditions": {},
            "model_parameters": {"motion": "yaw-left", "strength": 0.18},
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert agent.process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (blur_job["id"],))
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row("SELECT * FROM datasets WHERE name='真实航拍运动模糊'")
    assert transformed["scene_domain"] == "无人机航拍"
    assert transformed["weather"] == "未记录"
    sensor = json.loads(transformed["sensor_conditions"])
    assert sensor["motion_blur"] is True
    assert sensor["condition_label"] == "无人机运动模糊"
    assert sensor["motion_blur_model"] == "ID-Blau"
    assert sensor["motion"] == "yaw-left"
    assert sensor["motion_blur_strength"] == pytest.approx(0.18)
    assert sensor["motion_blur_sample_timesteps"] == 20
    request = captured["request"]
    assert request["model_parameters"] == {
        "effect": "motion_blur",
        "domain": "uav_aerial",
        "motion": "yaw-left",
        "strength": 0.18,
        "sample_timesteps": 20,
        "precision": "FP32",
        "checkpoint": "ID_Blau.pth",
    }
    assert str(captured["command"][0]).endswith("envs/blau/bin/python")
    assert captured["environment"]["DIFFUSION_BLUR_ROOT"].endswith(
        "DiffusionBlur"
    )

    condition_directory = tmp_path / "motion-conditions"
    condition_directory.mkdir()
    np.save(
        condition_directory / "aerial_condition.npy",
        np.stack(
            (
                np.ones((40, 64), dtype=np.float32),
                np.zeros((40, 64), dtype=np.float32),
                np.full((40, 64), 0.2, dtype=np.float32),
            )
        ),
    )
    condition_file_job = queue_job(
        "ACQUISITION",
        {
            "name": "真实航拍条件文件运动模糊",
            "adapter_id": "adapter_motion_blur",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [2023],
            "input_dataset_id": imported["id"],
            "conditions": {},
            "model_parameters": {
                "motion": "forward",
                "strength": 0.14,
                "condition_directory": str(condition_directory),
            },
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert agent.process_one()
    completed = database.row(
        "SELECT * FROM jobs WHERE id=?", (condition_file_job["id"],)
    )
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row(
        "SELECT * FROM datasets WHERE name='真实航拍条件文件运动模糊'"
    )
    sensor = json.loads(transformed["sensor_conditions"])
    assert sensor["motion"] == "condition-files"
    assert sensor["motion_condition_directory"] == str(condition_directory)
    assert sensor["motion_condition_matching"] == "filename"
    assert sensor["motion_condition_fallback"] == "random-preset"
    assert sensor["motion_condition_matched"] == 1
    assert sensor["motion_condition_fallback_count"] == 0
    request = captured["request"]
    assert request["model_parameters"]["condition_directory"] == str(
        condition_directory
    )
    assert request["model_parameters"]["condition_matching"] == "filename"
    assert request["model_parameters"]["fallback_motion"] == "random-preset"

    day_to_night_job = queue_job(
        "ACQUISITION",
        {
            "name": "真实航拍无人机弱光",
            "adapter_id": "adapter_day_to_night",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [42],
            "input_dataset_id": imported["id"],
            "conditions": {},
            "model_parameters": {"effect": "day_to_night"},
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert agent.process_one()
    completed = database.row(
        "SELECT * FROM jobs WHERE id=?", (day_to_night_job["id"],)
    )
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row(
        "SELECT * FROM datasets WHERE name='真实航拍无人机弱光'"
    )
    assert transformed["scene_domain"] == "无人机航拍"
    assert transformed["weather"] == "弱光"
    assert transformed["annotation_status"] == "CANDIDATE"
    sensor = json.loads(transformed["sensor_conditions"])
    assert sensor["day_to_night"] is True
    assert sensor["condition_label"] == "无人机弱光"
    assert sensor["source_time_of_day"] == "白天"
    assert sensor["target_time_of_day"] == "夜间"
    assert sensor["day_to_night_checkpoint"] == (
        "uav_daynight_sichuan_3125_model_3125"
    )
    assert sensor["day_to_night_image_prep"] == "resize_640x640"
    assert sensor["day_to_night_model_size"] == 640
    request = captured["request"]
    assert request["model_parameters"] == {
        "effect": "day_to_night",
        "domain": "uav_aerial",
        "direction": "a2b",
        "image_prep": "resize_640x640",
        "model_size": 640,
        "precision": "FP16",
        "checkpoint": "uav_daynight_sichuan_3125_model_3125",
    }
    assert str(captured["command"][0]).endswith("DiffusionDegrade/.venv/bin/python")
    assert captured["environment"][
        "DIFFUSION_DEGRADE_UAV_DAY_TO_NIGHT_CHECKPOINT"
    ].endswith("outputs/uav_daynight_sichuan_3125/checkpoints/model_3125.pkl")
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"

    database.execute(
        "UPDATE datasets SET scene_domain='城市驾驶' WHERE id=?",
        (imported["id"],),
    )
    driving_fog_job = queue_job(
        "ACQUISITION",
        {
            "name": "城市驾驶自动驾驶气雾",
            "adapter_id": "adapter_warpi2i_fog",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [42],
            "input_dataset_id": imported["id"],
            "conditions": {},
            "model_parameters": {"effect": "fog", "method": "paired"},
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert agent.process_one()
    completed = database.row(
        "SELECT * FROM jobs WHERE id=?", (driving_fog_job["id"],)
    )
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row(
        "SELECT * FROM datasets WHERE name='城市驾驶自动驾驶气雾'"
    )
    assert transformed["scene_domain"] == "城市驾驶"
    assert transformed["weather"] == "雾"
    sensor = json.loads(transformed["sensor_conditions"])
    assert sensor["condition_label"] == "自动驾驶气雾"
    assert sensor["fog_model"] == "WarpI2I · 自动驾驶气雾"
    assert sensor["fog_method"] == "paired"
    assert sensor["fog_checkpoint"] == "foggy_1.pkl"
    assert captured["request"]["model_parameters"] == {
        "effect": "fog",
        "domain": "autonomous_driving",
        "method": "paired",
        "image_prep": "multiple_of_8",
        "precision": "FP16",
        "checkpoint": "foggy_1.pkl",
    }
    assert str(captured["command"][0]).endswith(
        "DiffusionDegrade/.venv/bin/python"
    )
    assert captured["environment"]["WARPI2I_DRIVING_FOG_CHECKPOINT"].endswith(
        "pix2pix_turbo/2_24_drive_v2_warped_128/foggy_1.pkl"
    )

    driving_night_job = queue_job(
        "ACQUISITION",
        {
            "name": "城市驾驶自动驾驶弱光",
            "adapter_id": "adapter_warpi2i_day_to_night",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [42],
            "input_dataset_id": imported["id"],
            "conditions": {},
            "model_parameters": {"effect": "day_to_night"},
            "category_template": "custom",
            "categories": [{"id": 1, "name": "car"}],
        },
        database,
    )
    assert agent.process_one()
    completed = database.row(
        "SELECT * FROM jobs WHERE id=?", (driving_night_job["id"],)
    )
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row(
        "SELECT * FROM datasets WHERE name='城市驾驶自动驾驶弱光'"
    )
    assert transformed["scene_domain"] == "城市驾驶"
    assert transformed["weather"] == "弱光"
    sensor = json.loads(transformed["sensor_conditions"])
    assert sensor["day_to_night"] is True
    assert sensor["condition_label"] == "自动驾驶弱光"
    assert sensor["day_to_night_model"] == "WarpI2I · 自动驾驶弱光"
    assert sensor["day_to_night_method"] == "unpaired"
    assert sensor["day_to_night_checkpoint"] == "BDD100K_day2night.pkl"
    assert captured["request"]["model_parameters"] == {
        "effect": "day_to_night",
        "domain": "autonomous_driving",
        "method": "unpaired",
        "direction": "a2b",
        "image_prep": "resize_512x512",
        "precision": "FP16",
        "checkpoint": "BDD100K_day2night.pkl",
    }
    assert str(captured["command"][0]).endswith(
        "DiffusionDegrade/.venv/bin/python"
    )
    assert captured["environment"][
        "WARPI2I_DRIVING_DAY_TO_NIGHT_CHECKPOINT"
    ].endswith("cyclegan_turbo/BDD100K_day2night.pkl")
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"


def test_resource_picker_upload_imports_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, app_settings = make_database(tmp_path)
    image_bytes = io.BytesIO()
    Image.new("RGB", (64, 40), (30, 110, 170)).save(
        image_bytes,
        format="PNG",
    )
    image_bytes.seek(0)
    annotation_bytes = io.BytesIO(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "sample.png",
                        "width": 64,
                        "height": 40,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [8, 6, 20, 10],
                        "area": 200,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        ).encode()
    )
    captured: dict[str, object] = {}

    def capture_job(job_type: str, payload: dict[str, object]) -> dict[str, str]:
        captured["job_type"] = job_type
        captured["payload"] = payload
        return {"id": "job_upload_capture", "status": "QUEUED"}

    monkeypatch.setattr(main_module, "settings", app_settings)
    monkeypatch.setattr(main_module, "queue_job", capture_job)
    response = main_module._stage_import_upload(
        name="资源管理器导入",
        scene_domain="无人机航拍",
        annotation_format="COCO",
        images=[UploadFile(image_bytes, filename="sample.png")],
        relative_paths=["selected-folder/nested/sample.png"],
        annotation=UploadFile(annotation_bytes, filename="instances.json"),
        annotation_files=[],
        annotation_relative_paths=[],
        category_template="custom",
        categories=[{"id": 1, "name": "car"}],
    )
    assert response["status"] == "QUEUED"
    assert captured["job_type"] == "DATASET_IMPORT"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    staging_root = Path(str(payload["staged_upload_root"]))
    assert (staging_root / "images/selected-folder/nested/sample.png").is_file()

    job = queue_job("DATASET_IMPORT", payload, database)
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    dataset = database.row(
        "SELECT * FROM datasets WHERE name='资源管理器导入'"
    )
    assert dataset
    artifact = app_settings.artifact_dir / dataset["artifact_path"]
    assert len(list(artifact.glob("*.png"))) == 1
    assert (artifact / "annotations/instances.json").is_file()
    page = list_dataset_samples(dataset["id"], 0, 48, database)
    assert page
    assert page["items"][0]["annotation_source"] == "COCO"
    assert page["items"][0]["boxes"][0]["label"] == "car"
    assert page["items"][0]["boxes"][0]["x"] == pytest.approx(8 / 64)
    assert not staging_root.exists()


def test_upload_endpoint_uses_raised_multipart_file_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}
    file_count = 1001
    upload = UploadFile(io.BytesIO(b"x"), filename="sample.png")

    class FakeForm:
        fields = {
            "name": "大目录上传",
            "scene_domain": "无人机航拍",
            "relative_paths_json": json.dumps(
                [f"images/{index}.png" for index in range(file_count)]
            ),
            "annotation_relative_paths_json": "[]",
            "categories_json": json.dumps([{"id": 1, "name": "car"}]),
            "category_template": "custom",
        }

        def get(self, name: str) -> object | None:
            return self.fields.get(name)

        def getlist(self, name: str) -> list[UploadFile]:
            return [upload] * file_count if name == "images" else []

    class FakeRequest:
        async def form(self, **limits: int) -> FakeForm:
            captured.update(limits)
            return FakeForm()

    def capture_upload(**values: object) -> dict[str, str]:
        captured["images"] = len(values["images"])  # type: ignore[arg-type]
        captured["paths"] = len(values["relative_paths"])  # type: ignore[arg-type]
        return {"id": "job_large_upload", "status": "QUEUED"}

    monkeypatch.setattr(main_module, "_stage_import_upload", capture_upload)
    response = asyncio.run(
        main_module.import_uploaded_dataset(FakeRequest())  # type: ignore[arg-type]
    )
    assert response["status"] == "QUEUED"
    assert captured == {
        "max_files": 20001,
        "max_fields": 20,
        "max_part_size": 2 * 1024 * 1024,
        "images": file_count,
        "paths": file_count,
    }


def test_yolo_annotation_directory_imports_every_label_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, app_settings = make_database(tmp_path)

    def image_upload(name: str, color: tuple[int, int, int]) -> UploadFile:
        content = io.BytesIO()
        Image.new("RGB", (64, 40), color).save(content, format="PNG")
        content.seek(0)
        return UploadFile(content, filename=name)

    captured: dict[str, object] = {}

    def capture_job(job_type: str, payload: dict[str, object]) -> dict[str, str]:
        captured["job_type"] = job_type
        captured["payload"] = payload
        return {"id": "job_yolo_capture", "status": "QUEUED"}

    monkeypatch.setattr(main_module, "settings", app_settings)
    monkeypatch.setattr(main_module, "queue_job", capture_job)
    response = main_module._stage_import_upload(
        name="YOLO目录导入",
        scene_domain="无人机航拍",
        annotation_format="YOLO",
        images=[
            image_upload("first.png", (30, 110, 170)),
            image_upload("second.png", (90, 60, 130)),
        ],
        relative_paths=[
            "images/first.png",
            "images/second.png",
        ],
        annotation=None,
        annotation_files=[
            UploadFile(io.BytesIO(b"0 0.5 0.5 0.2 0.2\n"), filename="first.txt"),
            UploadFile(io.BytesIO(b"1 0.4 0.4 0.1 0.1\n"), filename="second.txt"),
        ],
        annotation_relative_paths=[
            "labels/first.txt",
            "labels/nested/second.txt",
        ],
        category_template="custom",
        categories=[
            {"id": 0, "name": "car"},
            {"id": 1, "name": "pedestrian"},
        ],
    )
    assert response["status"] == "QUEUED"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    staging_root = Path(str(payload["staged_upload_root"]))
    annotation_root = Path(str(payload["annotation_path"]))
    assert len(list(annotation_root.rglob("*.txt"))) == 2

    job = queue_job("DATASET_IMPORT", payload, database)
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    result = json.loads(completed["result"])
    assert result["annotation_files"] == 2
    dataset = database.row("SELECT * FROM datasets WHERE name='YOLO目录导入'")
    assert dataset and dataset["annotation_status"] == "CANDIDATE"
    artifact = app_settings.artifact_dir / dataset["artifact_path"]
    assert (artifact / "first.png").is_file()
    assert (artifact / "second.png").is_file()
    assert (
        artifact / "annotations/yolo/labels/first.txt"
    ).is_file()
    assert (
        artifact / "annotations/yolo/labels/nested/second.txt"
    ).is_file()
    page = list_dataset_samples(dataset["id"], 0, 48, database)
    assert page
    assert [item["annotation_source"] for item in page["items"]] == [
        "YOLO",
        "YOLO",
        ]
    assert page["items"][0]["boxes"][0] == {
        "label": "car",
        "color": "#1677FF",
        "x": pytest.approx(0.4),
        "y": pytest.approx(0.4),
        "width": pytest.approx(0.2),
        "height": pytest.approx(0.2),
    }
    assert not staging_root.exists()


def test_visdrone_import_creates_committed_coco_annotations(
    tmp_path: Path,
) -> None:
    database, app_settings = make_database(tmp_path)
    image_directory = tmp_path / "visdrone-images"
    label_directory = tmp_path / "visdrone-annotations"
    image_directory.mkdir()
    label_directory.mkdir()
    Image.new("RGB", (100, 80), (30, 110, 170)).save(
        image_directory / "sample.jpg"
    )
    (label_directory / "sample.txt").write_text(
        "\n".join(
            [
                "10,20,30,40,1,4,0,0",
                "1,2,3,4,0,0,0,0",
                "5,6,7,8,1,11,0,0",
            ]
        ),
        encoding="utf-8",
    )
    job = queue_job(
        "DATASET_IMPORT",
        {
            "name": "VisDrone目录导入",
            "directory": str(image_directory),
            "annotation_path": str(label_directory),
            "annotation_format": "VISDRONE",
            "scene_domain": "无人机航拍",
            "category_template": "visdrone",
            "categories": template_categories("visdrone", "dataset"),
        },
        database,
    )
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    dataset = database.row(
        "SELECT * FROM datasets WHERE name='VisDrone目录导入'"
    )
    artifact = app_settings.artifact_dir / dataset["artifact_path"]
    coco = json.loads(
        (artifact / "annotations" / "instances.json").read_text(
            encoding="utf-8"
        )
    )
    assert coco["info"]["source_format"] == "VisDrone"
    assert len(coco["images"]) == 1
    assert [item["category_id"] for item in coco["annotations"]] == [3]
    assert [item["id"] for item in coco["categories"]] == list(range(10))
    assert len(coco["categories"]) == 10
    session = get_annotation_session(dataset["id"], database)
    assert session and session["samples"][0]["box_count"] == 1
    annotation = get_sample_annotation(dataset["id"], "sample.jpg", database)
    assert annotation
    assert annotation["width"] == 100
    assert annotation["height"] == 80
    assert annotation["boxes"] == [
        {
            "id": "imported_1",
            "category_id": 3,
            "x": pytest.approx(10),
            "y": pytest.approx(20),
            "width": pytest.approx(30),
            "height": pytest.approx(40),
        }
    ]
