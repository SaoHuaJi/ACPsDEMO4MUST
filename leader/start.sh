#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# Leader 作为独立部署单元启动，只把自身目录加入模块搜索路径。
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" run.py "$@"
