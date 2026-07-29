from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import settings
from app.db import db, json_dump, new_id, utc_now
from app.schemas import (
    AcquisitionRequest,
    AdapterRegistrationRequest,
    DatasetImportRequest,
    EvaluationPlanRequest,
    ModelCreateRequest,
)
from app.services import (
    BaseGenCatalogError,
    DatasetArtifactError,
    DatasetDeletionError,
    adapter_health,
    delete_dataset,
    environment_status,
    get_basegen_scene_schema,
    get_job,
    list_adapters,
    list_dataset_samples,
    list_datasets,
    list_jobs,
    list_models,
    overview,
    preview_basegen_plan,
    query_results,
    queue_job,
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


@app.post("/api/datasets/import", status_code=202)
def import_dataset(request: DatasetImportRequest) -> dict[str, Any]:
    return queue_job("DATASET_IMPORT", request.model_dump())


@app.post("/api/datasets/{dataset_id}/freeze")
def freeze_dataset(dataset_id: str) -> dict[str, Any]:
    dataset = db.row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if dataset["annotation_status"] not in {"VERIFIED", "CANDIDATE"}:
        raise HTTPException(status_code=409, detail="数据集缺少可校核真值，不能冻结")
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
    return queue_job("ACQUISITION", request.model_dump())


@app.get("/api/models")
def get_models() -> list[dict[str, Any]]:
    return list_models()


@app.post("/api/models", status_code=201)
def create_model(request: ModelCreateRequest) -> dict[str, Any]:
    if not db.row("SELECT id FROM adapters WHERE id=?", (request.adapter_id,)):
        raise HTTPException(status_code=404, detail="检测适配器不存在")
    model_id = new_id("model")
    db.execute(
        """
        INSERT INTO models
        (id,name,family,backbone,version,precision,adapter_id,weight_path,weight_sha256,
         is_demo,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            model_id, request.name, request.family, request.backbone, request.version,
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
    protocol = json.loads(plan["protocol"])
    return queue_job("EVALUATION", {"plan_id": plan_id, **protocol})


@app.get("/api/jobs")
def get_jobs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, Any]]:
    return list_jobs(limit=limit)


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
