#!/bin/bash
#
# 与 eval_all_combinations_straggler.sh 逻辑完全相同，仅增加：
#   1) 默认 CONDA_ENV=tts-v1（供 serve_gpu*.sh 的 tmux 内 conda activate）
#   2) CONTROLLER_CLIENT_HOST / WORKER_ADVERTISE_HOST（Ray 与 FastChat worker 注册）
#   3) 若当前 shell 不在目标 conda 环境中，自动用「conda run -n <env>」执行主脚本，
#      使 scripts/run.sh 调用的 python / evaluate.py 与 tts-v1 一致（避免未 activate 时用错解释器）。
#
# usage::
#
#   cd compute-optimal-tts/src
#   conda activate tts-v1   # 可选；不激活则由本脚本自动 conda run
#   bash scripts/eval_all_combinations_straggler_ttsv1.sh
#   bash scripts/eval_all_combinations_straggler_ttsv1.sh --dry-run
#

set -eo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAIN="${_SCRIPT_DIR}/eval_all_combinations_straggler.sh"

export CONDA_ENV="${CONDA_ENV:-tts-v1}"

if [ -z "${CONTROLLER_CLIENT_HOST:-}" ]; then
    _auto_ctrl_host="$(hostname -I 2>/dev/null | awk '{print $1}')"
    export CONTROLLER_CLIENT_HOST="${_auto_ctrl_host:-127.0.0.1}"
fi
export WORKER_ADVERTISE_HOST="${WORKER_ADVERTISE_HOST:-${CONTROLLER_CLIENT_HOST}}"

# 保证主评估流程（含 run.sh → python evaluate.py）使用 tts-v1
_use_conda_run=0
if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
        _use_conda_run=1
    fi
else
    echo "[eval_all_combinations_straggler_ttsv1] 警告: 未找到 conda，请手动「conda activate ${CONDA_ENV}」后再运行。" >&2
fi

if [ "${_use_conda_run}" -eq 1 ]; then
    exec conda run -n "${CONDA_ENV}" --no-capture-output bash "${_MAIN}" "$@"
fi

exec bash "${_MAIN}" "$@"
