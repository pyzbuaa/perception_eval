from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from app import __version__
from app.category_templates import list_category_templates, normalize_categories
from app.config import settings
from app.db import db, json_dump, new_id, utc_now
from app.schemas import (
    AcquisitionRequest,
    AdapterRegistrationRequest,
    AnnotationSchemaUpdate,
    DatasetImportRequest,
    EvaluationPlanRequest,
    LocalDetectorModelRequest,
    ModelCreateRequest,
    SampleAnnotationUpdate,
)
from app.services import (
    BaseGenCatalogError,
    CategoryCompatibilityError,
    DatasetAnnotationError,
    DatasetArtifactError,
    DatasetDeletionError,
    JobDeletionError,
    LocalModelRegistrationError,
    ModelDeletionError,
    adapter_health,
    complete_dataset_annotations,
    delete_dataset,
    delete_job,
    delete_model,
    environment_status,
    get_annotation_session,
    get_basegen_scene_schema,
    get_job,
    get_sample_annotation,
    list_adapters,
    list_dataset_samples,
    list_datasets,
    list_jobs,
    list_local_model_resources,
    list_models,
    overview,
    preview_basegen_plan,
    query_results,
    queue_job,
    register_local_detector_model,
    save_sample_annotation,
    update_annotation_schema,
    validate_evaluation_categories,
)
from app.worker import JobAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize()
    agent: JobAgent | None = None
    thread: threading.Thread | None = None
    if os.environ.get("PERCEPTION_EVAL_EMBEDDED_WORKER", "1") == "1":
        agent = JobAgent()
        thread = threading.Thread(target=agent.run_forever, name="perception-eval-agent", daemon=True)
        thread.start()
    app.state.agent = agent
    yield
    if agent:
        agent.stop()
    if thread:
        thread.join(timeout=3)


app = FastAPI(
    title="视觉感知效能评估平台",
    version=__version__,
    description="Single-workstation perception effectiveness evaluation control plane.",
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "version": __version__, "storage": "sqlite-workspace"}


@app.get("/api/overview")
def get_overview() -> dict[str, Any]:
    return overview()


@app.get("/api/adapters")
def get_adapters() -> list[dict[str, Any]]:
    return list_adapters()


@app.post("/api/adapters", status_code=201)
def register_adapter(request: AdapterRegistrationRequest) -> dict[str, Any]:
    adapter_id = new_id("adapter")
    now = utc_now()
    db.execute(
        """
        INSERT INTO adapters
        (id,name,kind,version,maturity,runtime_kind,runtime_prefix,policy,entrypoint,
         requires_gpu,status,description,parameter_schema,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            adapter_id, request.name, request.kind, request.version, request.maturity,
            request.runtime_kind, request.runtime_prefix, "read_only", request.entrypoint,
            int(request.requires_gpu), "REGISTERED", request.description,
            json_dump(request.parameter_schema), now, now,
        ),
    )
    return next(item for item in list_adapters() if item["id"] == adapter_id)


@app.post("/api/adapters/{adapter_id}/health-check")
def check_adapter(adapter_id: str) -> dict[str, Any]:
    result = adapter_health(adapter_id)
    if not result:
        raise HTTPException(status_code=404, detail="适配器不存在")
    return result


@app.get("/api/adapters/adapter_basegen/scene-schema")
def basegen_scene_schema() -> dict[str, Any]:
    try:
        return get_basegen_scene_schema()
    except BaseGenCatalogError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/adapters/adapter_basegen/preview")
def preview_basegen(request: AcquisitionRequest) -> dict[str, Any]:
    if request.adapter_id != "adapter_basegen":
        raise HTTPException(status_code=422, detail="预览请求必须使用 adapter_basegen")
    try:
        return preview_basegen_plan(request.model_dump())
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/datasets")
def get_datasets() -> list[dict[str, Any]]:
    return list(list_datasets())


@app.get("/api/category-templates")
def get_category_templates() -> list[dict[str, Any]]:
    return list_category_templates()


@app.get("/api/datasets/{dataset_id}/samples")
def get_dataset_samples(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=48, ge=1, le=200),
) -> dict[str, Any]:
    try:
        result = list_dataset_samples(dataset_id, offset, limit)
    except DatasetArtifactError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@app.get("/api/datasets/{dataset_id}/annotations")
def get_annotations(dataset_id: str) -> dict[str, Any]:
    try:
        result = get_annotation_session(dataset_id)
    except DatasetArtifactError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@app.put("/api/datasets/{dataset_id}/annotation-schema")
def put_annotation_schema(
    dataset_id: str,
    request: AnnotationSchemaUpdate,
) -> dict[str, Any]:
    try:
        result = update_annotation_schema(
            dataset_id,
            [category.model_dump() for category in request.categories],
        )
    except DatasetAnnotationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@app.get("/api/datasets/{dataset_id}/samples/{sample_name}/annotations")
def get_image_annotations(dataset_id: str, sample_name: str) -> dict[str, Any]:
    try:
        result = get_sample_annotation(dataset_id, sample_name)
    except DatasetArtifactError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集或图片不存在")
    return result


@app.put("/api/datasets/{dataset_id}/samples/{sample_name}/annotations")
def put_image_annotations(
    dataset_id: str,
    sample_name: str,
    request: SampleAnnotationUpdate,
) -> dict[str, Any]:
    try:
        result = save_sample_annotation(
            dataset_id,
            sample_name,
            request.model_dump(),
        )
    except (DatasetAnnotationError, DatasetArtifactError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集或图片不存在")
    return result


@app.post("/api/datasets/{dataset_id}/annotations/complete")
def complete_annotations(dataset_id: str) -> dict[str, Any]:
    try:
        result = complete_dataset_annotations(dataset_id)
    except (DatasetAnnotationError, DatasetArtifactError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@app.post("/api/datasets/import", status_code=202)
def import_dataset(request: DatasetImportRequest) -> dict[str, Any]:
    payload = request.model_dump()
    try:
        payload["categories"] = normalize_categories(payload["categories"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return queue_job("DATASET_IMPORT", payload)


def _safe_upload_path(filename: str | None) -> Path:
    relative = Path((filename or "").replace("\\", "/"))
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise HTTPException(status_code=422, detail="上传文件路径不安全")
    return relative


@app.post("/api/datasets/import-upload", status_code=202)
async def import_uploaded_dataset(request: Request) -> dict[str, Any]:
    form = await request.form(
        max_files=20001,
        max_fields=20,
        max_part_size=2 * 1024 * 1024,
    )

    def path_list(field: str) -> list[str]:
        raw = form.get(field)
        try:
            values = json.loads(str(raw or "[]"))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"{field} 不是有效的 JSON 数组",
            ) from exc
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise HTTPException(
                status_code=422,
                detail=f"{field} 必须是字符串数组",
            )
        return values

    def category_list() -> list[dict[str, Any]]:
        raw = form.get("categories_json")
        try:
            values = json.loads(str(raw or "[]"))
            if not isinstance(values, list):
                raise ValueError
            return normalize_categories(values)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="类别列表无效，请重新选择类别模板",
            ) from exc

    images = [
        item
        for item in form.getlist("images")
        if isinstance(item, UploadFile)
    ]
    annotation_files = [
        item
        for item in form.getlist("annotation_files")
        if isinstance(item, UploadFile)
    ]
    annotation_value = form.get("annotation")
    annotation = (
        annotation_value
        if isinstance(annotation_value, UploadFile)
        else None
    )
    return _stage_import_upload(
        name=str(form.get("name") or ""),
        scene_domain=str(form.get("scene_domain") or "未分类"),
        annotation_format=str(form.get("annotation_format") or "COCO"),
        images=images,
        relative_paths=path_list("relative_paths_json"),
        annotation=annotation,
        annotation_files=annotation_files,
        annotation_relative_paths=path_list(
            "annotation_relative_paths_json"
        ),
        category_template=str(form.get("category_template") or "custom"),
        categories=category_list(),
    )


def _stage_import_upload(
    name: str,
    scene_domain: str,
    annotation_format: str,
    images: list[UploadFile],
    relative_paths: list[str],
    annotation: UploadFile | None,
    annotation_files: list[UploadFile],
    annotation_relative_paths: list[str],
    category_template: str,
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    if not 2 <= len(name.strip()) <= 120:
        raise HTTPException(status_code=422, detail="数据集名称长度必须为 2 到 120")
    try:
        categories = normalize_categories(categories)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    annotation_format = annotation_format.upper()
    if annotation_format not in {"COCO", "YOLO", "VISDRONE"}:
        raise HTTPException(status_code=422, detail="标注格式不受支持")
    if len(images) != len(relative_paths):
        raise HTTPException(status_code=422, detail="上传图片和相对路径数量不一致")
    supported = {".jpg", ".jpeg", ".png", ".webp", ".svg"}
    selected = [
        (upload, relative)
        for upload, raw_path in zip(images, relative_paths)
        if (relative := _safe_upload_path(raw_path)).suffix.lower() in supported
    ]
    if not selected:
        raise HTTPException(status_code=422, detail="所选目录中没有支持的图像")
    if len(selected) > 10000:
        raise HTTPException(status_code=422, detail="单次最多导入 10000 张图像")
    label_uploads = annotation_files or []
    label_paths = annotation_relative_paths or []
    if annotation and annotation.filename and label_uploads:
        raise HTTPException(
            status_code=422,
            detail="COCO 单文件和 YOLO 标注目录不能同时上传",
        )
    if len(label_uploads) != len(label_paths):
        raise HTTPException(
            status_code=422,
            detail="YOLO 标注文件和相对路径数量不一致",
        )
    supported_labels = {".txt", ".yaml", ".yml", ".names"}
    selected_labels = [
        (upload, relative)
        for upload, raw_path in zip(label_uploads, label_paths)
        if (
            relative := _safe_upload_path(raw_path)
        ).suffix.lower() in supported_labels
    ]
    if label_uploads and not selected_labels:
        raise HTTPException(
            status_code=422,
            detail="所选 YOLO 目录中没有 TXT、YAML 或 NAMES 文件",
        )

    upload_id = new_id("upload")
    staging_root = settings.task_dir / "import_uploads" / upload_id
    image_directory = staging_root / "images"
    annotation_path: Path | None = None
    try:
        for upload, relative in selected:
            target = image_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise HTTPException(
                    status_code=422,
                    detail=f"目录中存在重名文件: {relative}",
                )
            with target.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
        if annotation and annotation.filename:
            relative = _safe_upload_path(Path(annotation.filename).name)
            annotation_path = staging_root / "annotation" / relative
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            with annotation_path.open("wb") as output:
                shutil.copyfileobj(annotation.file, output)
        elif selected_labels:
            annotation_path = staging_root / "annotation_directory"
            for upload, relative in selected_labels:
                target = annotation_path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise HTTPException(
                        status_code=422,
                        detail=f"YOLO 目录中存在重名文件: {relative}",
                    )
                with target.open("wb") as output:
                    shutil.copyfileobj(upload.file, output)
        return queue_job(
            "DATASET_IMPORT",
            {
                "name": name,
                "directory": str(image_directory),
                "annotation_path": (
                    str(annotation_path) if annotation_path else None
                ),
                "scene_domain": scene_domain,
                "annotation_format": annotation_format,
                "category_template": category_template,
                "categories": categories,
                "staged_upload_root": str(staging_root),
            },
        )
    except Exception:
        if staging_root.is_dir():
            shutil.rmtree(staging_root)
        raise


@app.post("/api/datasets/{dataset_id}/freeze")
def freeze_dataset(dataset_id: str) -> dict[str, Any]:
    dataset = db.row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if dataset["annotation_status"] not in {"VERIFIED", "CANDIDATE"}:
        raise HTTPException(status_code=409, detail="数据集缺少可校核真值，不能冻结")
    if not db.row(
        "SELECT 1 FROM dataset_annotation_schemas WHERE dataset_id=?",
        (dataset_id,),
    ):
        raise HTTPException(status_code=409, detail="数据集尚未配置类别，不能冻结")
    db.execute(
        "UPDATE datasets SET frozen=1,annotation_status='VERIFIED' WHERE id=?", (dataset_id,)
    )
    return {"id": dataset_id, "frozen": True, "annotation_status": "VERIFIED"}


@app.delete("/api/datasets/{dataset_id}")
def remove_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        result = delete_dataset(dataset_id)
    except DatasetDeletionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return result


@app.post("/api/acquisition-jobs", status_code=202)
def create_acquisition(request: AcquisitionRequest) -> dict[str, Any]:
    adapter = db.row("SELECT * FROM adapters WHERE id=?", (request.adapter_id,))
    if not adapter:
        raise HTTPException(status_code=404, detail="适配器不存在")
    if adapter["requires_gpu"] and not environment_status()["gpu"]["available"]:
        raise HTTPException(status_code=409, detail="GPU 当前不可用，任务已安全阻止")
    payload = request.model_dump()
    try:
        payload["categories"] = normalize_categories(payload["categories"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return queue_job("ACQUISITION", payload)


@app.get("/api/models")
def get_models() -> list[dict[str, Any]]:
    return list_models()


@app.get("/api/local-model-resources")
def get_local_model_resources(
    path: str | None = None,
    scope: str = Query(default="model"),
    kind: str = Query(default="directory"),
) -> dict[str, Any]:
    try:
        return list_local_model_resources(path, scope, kind)
    except LocalModelRegistrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/local-detector-models", status_code=201)
def create_local_detector_model(
    request: LocalDetectorModelRequest,
) -> dict[str, Any]:
    try:
        return register_local_detector_model(request.model_dump())
    except LocalModelRegistrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/models/{model_id}")
def remove_model(model_id: str) -> dict[str, Any]:
    try:
        result = delete_model(model_id)
    except ModelDeletionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="模型不存在")
    return result


@app.post("/api/models", status_code=201)
def create_model(request: ModelCreateRequest) -> dict[str, Any]:
    if not db.row("SELECT id FROM adapters WHERE id=?", (request.adapter_id,)):
        raise HTTPException(status_code=404, detail="检测适配器不存在")
    try:
        categories = normalize_categories(
            [category.model_dump() for category in request.categories]
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model_id = new_id("model")
    db.execute(
        """
        INSERT INTO models
        (id,name,family,architecture,backbone,detector_head,class_count,categories,
         category_template,
         training_dataset,pretrained_dataset,version,precision,adapter_id,
         weight_path,weight_sha256,is_demo,status,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            model_id, request.name, request.family, request.architecture,
            request.backbone, request.detector_head, len(categories),
            json_dump(categories), request.category_template,
            request.training_dataset, request.pretrained_dataset, request.version,
            request.precision, request.adapter_id, request.weight_path, None,
            int(request.is_demo), "REGISTERED", utc_now(),
        ),
    )
    return next(item for item in list_models() if item["id"] == model_id)


@app.post("/api/evaluation-plans", status_code=201)
def create_plan(request: EvaluationPlanRequest) -> dict[str, Any]:
    for dataset_id in request.dataset_ids:
        if not db.row("SELECT id FROM datasets WHERE id=?", (dataset_id,)):
            raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_id}")
    for model_id in request.model_ids:
        if not db.row("SELECT id FROM models WHERE id=?", (model_id,)):
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    try:
        validate_evaluation_categories(request.dataset_ids, request.model_ids)
    except CategoryCompatibilityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    plan_id = new_id("plan")
    protocol = {
        "batch_size": request.batch_size,
        "precision": request.precision,
        "warmup": request.warmup,
        "timing": "cuda-synchronized",
        "official": False,
    }
    db.execute(
        """
        INSERT INTO evaluation_plans
        (id,name,dataset_ids,model_ids,seeds,blur_levels,protocol,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            plan_id, request.name, json_dump(request.dataset_ids), json_dump(request.model_ids),
            json_dump(request.seeds), json_dump(request.blur_levels), json_dump(protocol), utc_now(),
        ),
    )
    return {
        "id": plan_id,
        "name": request.name,
        "combination_count": len(request.dataset_ids) * len(request.model_ids) * len(request.seeds) * len(request.blur_levels),
        "protocol": protocol,
    }


@app.post("/api/evaluation-plans/{plan_id}/runs", status_code=202)
def run_plan(plan_id: str) -> dict[str, Any]:
    plan = db.row("SELECT * FROM evaluation_plans WHERE id=?", (plan_id,))
    if not plan:
        raise HTTPException(status_code=404, detail="评测方案不存在")
    try:
        validate_evaluation_categories(
            json.loads(plan["dataset_ids"]),
            json.loads(plan["model_ids"]),
        )
    except CategoryCompatibilityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    protocol = json.loads(plan["protocol"])
    return queue_job("EVALUATION", {"plan_id": plan_id, **protocol})


@app.get("/api/jobs")
def get_jobs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, Any]]:
    return list_jobs(limit=limit)


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str) -> dict[str, Any]:
    try:
        result = delete_job(job_id)
    except JobDeletionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    return result


@app.get("/api/runs/{run_or_job_id}")
def get_run(run_or_job_id: str) -> dict[str, Any]:
    job = get_job(run_or_job_id)
    if job:
        return job
    run = db.row("SELECT * FROM runs WHERE id=?", (run_or_job_id,))
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    return run


@app.post("/api/runs/{job_id}/cancel")
def cancel_run(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        raise HTTPException(status_code=409, detail="任务已经结束")
    db.execute("UPDATE jobs SET cancel_requested=1,stage='正在取消' WHERE id=?", (job_id,))
    return {"id": job_id, "cancel_requested": True}


@app.get("/api/runs/{job_id}/events")
async def run_events(job_id: str) -> StreamingResponse:
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="任务不存在")

    async def stream():
        previous = None
        while True:
            job = get_job(job_id)
            if not job:
                break
            snapshot = json.dumps(job, ensure_ascii=False)
            if snapshot != previous:
                yield f"event: progress\ndata: {snapshot}\n\n"
                previous = snapshot
            if job["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/results")
def get_results(
    scene: str | None = None,
    condition: str | None = None,
    resolution: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    return query_results(scene, condition, resolution, model_id)


@app.get("/api/environment/status")
def get_environment_status() -> dict[str, Any]:
    return environment_status()


settings.ensure_directories()
app.mount("/artifacts", StaticFiles(directory=settings.artifact_dir), name="artifacts")

frontend_dist = settings.root_dir / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {"name": "视觉感知效能评估平台", "version": __version__, "frontend": "not built", "docs": "/docs"}
        )
