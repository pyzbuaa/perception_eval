from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.config import ROOT_DIR, Settings
from app.db import Database
from app.services import query_results, queue_job
from app.worker import JobAgent


def make_database(tmp_path: Path) -> tuple[Database, Settings]:
    app_settings = Settings(root_dir=ROOT_DIR, data_dir=tmp_path / "data")
    database = Database(app_settings)
    database.initialize()
    return database, app_settings


def test_database_seeds_traceable_demo_data(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    assert database.row("SELECT COUNT(*) AS n FROM datasets")["n"] == 3
    assert database.row("SELECT COUNT(*) AS n FROM models")["n"] == 3
    assert database.row("SELECT COUNT(*) AS n FROM results")["n"] == 27
    assert database.row("SELECT COUNT(*) AS n FROM results WHERE is_official=1")["n"] == 0


def test_result_query_keeps_resolution_as_group_dimension(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    result = query_results(scene="无人机航拍", database=database)
    assert result["count"] == 18
    assert len(result["groups"]) == 6
    assert {group["resolution"] for group in result["groups"]} == {"1920×1080"}
    assert all(group["seed_count"] == 3 for group in result["groups"])


def test_replay_adapter_job_creates_dataset_draft(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    job = queue_job(
        "ACQUISITION",
        {
            "name": "测试航拍回放",
            "adapter_id": "adapter_replay",
            "source_type": "REPLAY_FIXTURE",
            "sample_count": 4,
            "seeds": [1001, 1002, 1003],
            "conditions": {
                "scene": {"domain": "无人机航拍", "weather": "晴朗"},
                "sensor": {"resolution": "1920×1080", "motion_blur": 0.2},
            },
            "model_parameters": {},
        },
        database,
    )
    assert JobAgent(database, app_settings).process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert completed["status"] == "SUCCEEDED"
    dataset = database.row("SELECT * FROM datasets WHERE name='测试航拍回放'")
    assert dataset["sample_count"] == 4
    assert dataset["annotation_status"] == "CANDIDATE"
    assert dataset["frozen"] == 0


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


def test_local_import_can_feed_real_condition_operator(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    source_directory = tmp_path / "source-images"
    source_directory.mkdir()
    Image.new("RGB", (64, 40), (38, 120, 180)).save(source_directory / "aerial.png")
    import_job = queue_job(
        "DATASET_IMPORT",
        {"name": "真实航拍导入", "directory": str(source_directory), "annotation_path": None, "scene_domain": "无人机航拍"},
        database,
    )
    agent = JobAgent(database, app_settings)
    assert agent.process_one()
    imported = database.row("SELECT * FROM datasets WHERE name='真实航拍导入'")
    assert imported and imported["source_type"] == "REAL"
    condition_job = queue_job(
        "ACQUISITION",
        {
            "name": "真实航拍运动模糊",
            "adapter_id": "adapter_condition",
            "source_type": "REAL_TRANSFORMED",
            "sample_count": 1,
            "seeds": [1001],
            "input_dataset_id": imported["id"],
            "conditions": {
                "scene": {"domain": "无人机航拍", "weather": "晴朗"},
                "sensor": {"resolution": "原始分辨率", "motion_blur": 0.4},
            },
            "model_parameters": {},
        },
        database,
    )
    assert agent.process_one()
    completed = database.row("SELECT * FROM jobs WHERE id=?", (condition_job["id"],))
    assert completed["status"] == "SUCCEEDED"
    transformed = database.row("SELECT * FROM datasets WHERE name='真实航拍运动模糊'")
    assert transformed["source_type"] == "REAL_TRANSFORMED"
    outputs = list((app_settings.artifact_dir / transformed["artifact_path"]).glob("*.png"))
    assert len(outputs) == 1
