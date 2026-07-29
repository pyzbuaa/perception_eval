#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.runtime"
PLATFORM_ENV="${RUNTIME_DIR}/envs/platform"
PID_FILE="${RUNTIME_DIR}/perception-eval.pid"
LOG_FILE="${PROJECT_DIR}/data/logs/server.log"

if [[ ! -x "${PLATFORM_ENV}/bin/python" ]]; then
  echo "平台环境不存在，请先运行 ${PROJECT_DIR}/scripts/bootstrap.sh" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "平台已经运行，PID $(<"${PID_FILE}")"
  exit 0
fi

export PERCEPTION_EVAL_RUNTIME_DIR="${RUNTIME_DIR}"
export PERCEPTION_EVAL_DATA_DIR="${PROJECT_DIR}/data"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache/xdg"
export PIP_CACHE_DIR="${RUNTIME_DIR}/cache/pip"
export HF_HOME="${RUNTIME_DIR}/cache/huggingface"
export TORCH_HOME="${RUNTIME_DIR}/cache/torch"
export CUDA_CACHE_PATH="${RUNTIME_DIR}/cache/cuda"
export MPLCONFIGDIR="${RUNTIME_DIR}/cache/matplotlib"
export TMPDIR="${RUNTIME_DIR}/tmp"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${PROJECT_DIR}/data/logs" "${TMPDIR}"
cd "${PROJECT_DIR}"
nohup "${PLATFORM_ENV}/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 18080 \
  >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${PID_FILE}"

for _ in {1..40}; do
  if "${PLATFORM_ENV}/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/api/health', timeout=1)" 2>/dev/null; then
    echo "平台已启动：http://127.0.0.1:18080"
    echo "日志：${LOG_FILE}"
    exit 0
  fi
  sleep 0.25
done

echo "平台未能在预期时间启动，请检查日志：${LOG_FILE}" >&2
exit 1

