#!/bin/bash

# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="/DISK1/data/rnxu_24/.cache/huggingface"

# 使用项目的.venv环境
VENV_PYTHON="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/.venv/bin/python"

MODELS_DIR="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/data/models"

# 获取data/models中的所有模型目录
declare -a model_list=()
for model_path in "${MODELS_DIR}"/*; do
    if [ -d "$model_path" ]; then
        model_list+=("$model_path")
    fi
done

echo "使用Python: ${VENV_PYTHON}"
${VENV_PYTHON} --version
echo "找到 ${#model_list[@]} 个模型"
echo ""

for model in "${model_list[@]}"
do
    model_name=$(basename "${model}")
    echo "========================================"
    echo "Processing: ${model_name}"
    echo "model = ${model}"
    echo "========================================"
    ${VENV_PYTHON} llm_shape_profile.py --model ${model} --cpu-only
    echo ""
done

echo "全部完成！"
