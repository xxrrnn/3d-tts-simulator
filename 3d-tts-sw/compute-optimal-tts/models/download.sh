#!/usr/bin/env bash
# 使用 HF 国内镜像下载 data/models 下与 eval 脚本一致的模型目录。
# 依赖: pip install huggingface_hub（提供 hf 命令）
# 需接受协议的模型请先在官网同意条款，并设置: export HF_TOKEN=...

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export all_proxy=""
export ALL_PROXY=""
export no_proxy="*"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${DEST_DIR:-$SCRIPT_DIR}"

if ! command -v hf &>/dev/null; then
  echo "未找到 hf 命令，请先安装: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "下载到: $DEST_DIR"
echo

# 本地目录名 | Hugging Face repo_id
# Skywork PRM：官网仓库名为 Qwen-2.5（带连字符），与本地文件夹名一致。

while IFS='|' read -r local_name repo_id; do
  [[ -z "${local_name// }" ]] && continue
  [[ "$local_name" =~ ^# ]] && continue

  out="${DEST_DIR}/${local_name}"
  echo "========== ${local_name} <- ${repo_id} =========="
  hf download "${repo_id}" \
    --local-dir "${out}" 
  echo
done <<'EOF'
Qwen2.5-Math-1.5B-Instruct|Qwen/Qwen2.5-Math-1.5B-Instruct
Skywork-o1-Open-PRM-Qwen-2.5-1.5B|Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
math-shepherd-mistral-7b-prm|peiyi9979/math-shepherd-mistral-7b-prm
Qwen2.5-Math-7B-Instruct|Qwen/Qwen2.5-Math-7B-Instruct
Skywork-o1-Open-PRM-Qwen-2.5-7B|Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
EOF

echo "全部任务结束。"

