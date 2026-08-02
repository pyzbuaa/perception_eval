from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = Path(os.environ.get("PERCEPTION_EVAL_DATA_DIR", ROOT_DIR / "data"))
    basegen_root: Path = Path(
        os.environ.get("BASEGEN_ROOT", ROOT_DIR.parent / "BaseGen")
    )
    basegen_conda_prefix: Path = Path(
        os.environ.get("BASEGEN_CONDA_PREFIX", "/home/yons/miniforge3/envs/gen")
    )
    dronedets_root: Path = Path(
        os.environ.get("DRONEDETS_ROOT", ROOT_DIR.parent / "DroneDets")
    )
    dronedets_runtime_prefix: Path = Path(
        os.environ.get(
            "DRONEDETS_RUNTIME_PREFIX",
            ROOT_DIR.parent / "DroneDets" / ".venv",
        )
    )
    model_library_root: Path = Path(
        os.environ.get(
            "PERCEPTION_EVAL_MODEL_LIBRARY_ROOT",
            ROOT_DIR.parent,
        )
    )
    model_environment_root: Path = Path(
        os.environ.get(
            "PERCEPTION_EVAL_MODEL_ENVIRONMENT_ROOT",
            "/home/yons/miniforge3/envs",
        )
    )
    adapter_timeout_seconds: int = int(
        os.environ.get("PERCEPTION_EVAL_ADAPTER_TIMEOUT_SECONDS", "7200")
    )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "perception_eval.db"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def task_dir(self) -> Path:
        return self.data_dir / "task_workspaces"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def runtime_dir(self) -> Path:
        return Path(os.environ.get("PERCEPTION_EVAL_RUNTIME_DIR", ROOT_DIR / ".runtime"))

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.artifact_dir, self.task_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
