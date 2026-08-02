from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AcquisitionRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    adapter_id: str
    source_type: Literal[
        "GENERATIVE", "SIMULATOR", "REAL", "REAL_TRANSFORMED", "REPLAY_FIXTURE"
    ] = "REPLAY_FIXTURE"
    sample_count: int = Field(default=12, ge=1, le=1000)
    seeds: list[int] = Field(default_factory=lambda: [1001, 1002, 1003], min_length=1)
    conditions: dict[str, Any] = Field(default_factory=dict)
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    input_dataset_id: str | None = None
    category_template: str = Field(min_length=1, max_length=40)
    categories: list["CategoryInput"] = Field(min_length=1, max_length=1000)


class DatasetImportRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    directory: str
    annotation_path: str | None = None
    scene_domain: str = "未分类"
    category_template: str = Field(min_length=1, max_length=40)
    categories: list["CategoryInput"] = Field(min_length=1, max_length=1000)


class CategoryInput(BaseModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=64)


class AnnotationCategory(BaseModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=64)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class AnnotationSchemaUpdate(BaseModel):
    categories: list[AnnotationCategory] = Field(min_length=1, max_length=100)


class DetectionBox(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    category_id: int = Field(ge=0)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SampleAnnotationUpdate(BaseModel):
    width: int = Field(ge=1, le=100000)
    height: int = Field(ge=1, le=100000)
    boxes: list[DetectionBox] = Field(default_factory=list, max_length=5000)
    completed: bool = False


class ModelCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    family: str
    architecture: str = "未记录"
    backbone: str = "未记录"
    detector_head: str = "未记录"
    categories: list[CategoryInput] = Field(min_length=1, max_length=1000)
    category_template: str = Field(min_length=1, max_length=40)
    training_dataset: str = "未记录"
    pretrained_dataset: str = "未记录"
    version: str = "v1"
    precision: Literal["FP32", "FP16", "INT8"] = "FP32"
    adapter_id: str
    weight_path: str | None = None
    is_demo: bool = False


class LocalDetectorModelRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    family: str = Field(min_length=1, max_length=80)
    architecture: str = Field(min_length=1, max_length=80)
    backbone: str = Field(min_length=1, max_length=80)
    detector_head: str = Field(min_length=1, max_length=80)
    categories: list[CategoryInput] = Field(min_length=1, max_length=1000)
    category_template: str = Field(min_length=1, max_length=40)
    training_dataset: str = Field(min_length=1, max_length=120)
    pretrained_dataset: str = Field(min_length=1, max_length=120)
    version: str = Field(default="v1", min_length=1, max_length=40)
    precision: Literal["FP32", "FP16"] = "FP16"
    project_directory: str
    working_directory: str
    runtime_prefix: str
    command_arguments: list[str] = Field(min_length=1, max_length=100)
    weight_path: str


class EvaluationPlanRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    dataset_ids: list[str] = Field(min_length=1)
    model_ids: list[str] = Field(min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [1001, 1002, 1003], min_length=1)
    blur_levels: list[float] = Field(default_factory=lambda: [0.0])
    batch_size: int = Field(default=1, ge=1, le=64)
    precision: Literal["FP32", "FP16", "INT8"] = "FP16"
    warmup: int = Field(default=20, ge=0, le=200)


class AdapterRegistrationRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    kind: Literal["GENERATOR", "SIMULATOR", "IMPORTER", "DETECTOR", "OPERATOR"]
    version: str = "v1"
    maturity: Literal[
        "REGISTERED", "CONTRACT_OK", "EXPERIMENTAL", "READY", "BENCHMARK_READY"
    ] = "REGISTERED"
    runtime_kind: Literal["platform", "conda_external", "conda_clone", "remote_http"]
    runtime_prefix: str | None = None
    entrypoint: str | None = None
    requires_gpu: bool = False
    description: str = ""
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
