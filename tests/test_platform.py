from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from adapters.basegen_generator import prepare_plan
from app.config import ROOT_DIR, Settings
from app.db import Database, json_dump, utc_now
from app.services import (
    DatasetAnnotationError,
    DatasetDeletionError,
    complete_dataset_annotations,
    delete_dataset,
    get_annotation_session,
    get_basegen_scene_schema,
    get_sample_annotation,
    list_dataset_samples,
    preview_basegen_plan,
    query_results,
    queue_job,
    save_sample_annotation,
    update_annotation_schema,
)
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
    adapter = database.row("SELECT * FROM adapters WHERE id='adapter_basegen'")
    assert adapter
    assert adapter["runtime_kind"] == "conda_external"
    assert adapter["requires_gpu"] == 1


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


def test_external_conda_adapter_uses_registered_python(tmp_path: Path) -> None:
    database, app_settings = make_database(tmp_path)
    runtime_prefix = Path(sys.executable).parent.parent
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
            "adapters/replay_generator.py",
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
            "source_type": "REPLAY_FIXTURE",
            "sample_count": 2,
            "seeds": [7],
            "conditions": {
                "scene": {"domain": "城市驾驶", "weather": "晴朗"},
                "sensor": {"resolution": "960×540"},
            },
            "model_parameters": {},
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


def test_basegen_scene_schema_exposes_domain_specific_ui_fields() -> None:
    schema = get_basegen_scene_schema()
    assert [domain["value"] for domain in schema["domains"]] == [
        "autonomous-driving",
        "low-altitude-uav",
        "offroad-autonomous-driving",
    ]
    uav = next(
        domain
        for domain in schema["domains"]
        if domain["value"] == "low-altitude-uav"
    )
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


def test_delete_dataset_moves_artifacts_to_trash(tmp_path: Path) -> None:
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
            0,
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
    assert len(session["categories"]) == 6

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


def test_delete_dataset_rejects_frozen_and_run_referenced_data(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    with pytest.raises(DatasetDeletionError, match="冻结"):
        delete_dataset("dataset_aerial_clean", database)
    database.execute(
        "UPDATE datasets SET frozen=0 WHERE id='dataset_aerial_clean'"
    )
    with pytest.raises(DatasetDeletionError, match="评测运行"):
        delete_dataset("dataset_aerial_clean", database)
    assert database.row(
        "SELECT id FROM datasets WHERE id='dataset_aerial_clean'"
    )


def test_delete_dataset_rejects_evaluation_plan_reference(tmp_path: Path) -> None:
    database, _ = make_database(tmp_path)
    database.execute(
        "UPDATE datasets SET frozen=0 WHERE id='dataset_aerial_clean'"
    )
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
    with pytest.raises(DatasetDeletionError, match="评测方案"):
        delete_dataset("dataset_aerial_clean", database)


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
