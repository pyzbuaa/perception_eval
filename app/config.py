from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = Path(os.environ.get("PERCEPTION_EVAL_DATA_DIR", ROOT_DIR / "data"))

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

