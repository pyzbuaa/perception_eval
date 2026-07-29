#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${PROJECT_DIR}/.runtime/perception-eval.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "平台未运行。"
  exit 0
fi

SERVER_PID="$(<"${PID_FILE}")"
if kill -0 "${SERVER_PID}" 2>/dev/null; then
  kill "${SERVER_PID}"
  for _ in {1..20}; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
fi
rm -f "${PID_FILE}"
echo "平台已停止；数据和现有 conda 环境均保留。"

