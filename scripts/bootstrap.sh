#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.runtime"
PLATFORM_ENV="${RUNTIME_DIR}/envs/platform"

export CONDA_ENVS_PATH="${RUNTIME_DIR}/envs"
export CONDA_PKGS_DIRS="${RUNTIME_DIR}/pkgs"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache/xdg"
export PIP_CACHE_DIR="${RUNTIME_DIR}/cache/pip"
export HF_HOME="${RUNTIME_DIR}/cache/huggingface"
export TORCH_HOME="${RUNTIME_DIR}/cache/torch"
export CUDA_CACHE_PATH="${RUNTIME_DIR}/cache/cuda"
export MPLCONFIGDIR="${RUNTIME_DIR}/cache/matplotlib"
export TMPDIR="${RUNTIME_DIR}/tmp"

mkdir -p "${CONDA_ENVS_PATH}" "${CONDA_PKGS_DIRS}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
  "${HF_HOME}" "${TORCH_HOME}" "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${TMPDIR}"

MAMBA_BIN="$(command -v mamba || true)"
if [[ -z "${MAMBA_BIN}" ]]; then
  echo "未找到 mamba。请确认 Miniforge 已安装；脚本不会修改系统环境。" >&2
  exit 1
fi

"${MAMBA_BIN}" run -p "$(dirname "$(dirname "${MAMBA_BIN}")")" python \
  "${PROJECT_DIR}/scripts/environment_guard.py" capture

if [[ ! -x "${PLATFORM_ENV}/bin/python" ]]; then
  "${MAMBA_BIN}" create -y --prefix "${PLATFORM_ENV}" python=3.11 nodejs=22 pip
fi

"${PLATFORM_ENV}/bin/python" -m pip install --disable-pip-version-check -r "${PROJECT_DIR}/requirements.txt"
PATH="${PLATFORM_ENV}/bin:${PATH}" "${PLATFORM_ENV}/bin/npm" --prefix "${PROJECT_DIR}/frontend" install
PATH="${PLATFORM_ENV}/bin:${PATH}" "${PLATFORM_ENV}/bin/npm" --prefix "${PROJECT_DIR}/frontend" run build

"${PLATFORM_ENV}/bin/python" "${PROJECT_DIR}/scripts/environment_guard.py" verify

echo
echo "平台隔离环境已就绪：${PLATFORM_ENV}"
echo "已有 conda 环境未安装、升级或卸载任何包。"
echo "运行：${PROJECT_DIR}/scripts/start.sh"
