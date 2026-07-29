#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_DIR / ".runtime"
BASELINE_PATH = RUNTIME_DIR / "external-environments-baseline.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect() -> dict[str, object]:
    conda = shutil.which("conda") or shutil.which("mamba")
    if not conda:
        raise RuntimeError("未找到 conda 或 mamba")
    result = subprocess.run([conda, "info", "--json"], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    environments = []
    for value in info.get("envs", []):
        prefix = Path(value).resolve()
        try:
            prefix.relative_to(PROJECT_DIR)
            continue
        except ValueError:
            pass
        environments.append(
            {
                "prefix": str(prefix),
                "history_sha256": sha256(prefix / "conda-meta" / "history"),
            }
        )
    files = {}
    for path in (Path(info.get("user_rc_path", "")), Path.home() / ".bashrc", Path.home() / ".profile"):
        if str(path) and path.exists():
            files[str(path)] = sha256(path)
    return {"environments": environments, "configuration_files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that platform setup did not modify external conda environments.")
    parser.add_argument("action", choices=["capture", "verify"])
    args = parser.parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    current = collect()
    if args.action == "capture":
        BASELINE_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已记录外部环境基线：{BASELINE_PATH}")
        return
    if not BASELINE_PATH.is_file():
        raise RuntimeError("缺少环境基线，请先执行 capture")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline != current:
        print("检测到外部环境或 shell 配置发生变化。", file=sys.stderr)
        print(json.dumps({"before": baseline, "after": current}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
    print("环境隔离验证通过：外部 conda 环境与 shell 配置未变化。")


if __name__ == "__main__":
    main()

