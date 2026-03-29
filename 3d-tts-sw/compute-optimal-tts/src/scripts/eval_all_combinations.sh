#!/bin/bash

# 基础路径配置
BASE_PATH="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/data/models"
export LOGDIR="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/logs"
export HOST_ADDR="0.0.0.0"
export CONTROLLER_PORT=10014
export WORKER_BASE_PORT=10081

# GPU配置 (统一管理)
N_GPUS=2  # 可选值: 1, 2, 3, 4

# 脚本路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_BASE_DIR="${SRC_DIR}/output"

# 卡死检测：单次评估最大运行时长（秒）
EVAL_TIMEOUT_SECONDS=7200

# Policy模型列表
POLICY_MODELS=(
    # "Qwen2.5-1.5B-Instruct"
    # "Qwen2.5-0.5B-Instruct"
    # "Qwen2.5-3B-Instruct"
    "Qwen2.5-Math-7B-Instruct"
    "Qwen2.5-Math-1.5B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "DeepSeek-R1-Distill-Qwen-1.5B"
    # "DeepSeek-R1-Distill-Qwen-7B"
)

# Reward模型列表
REWARD_MODELS=(
    # "math-shepherd-mistral-7b-prm"
    # "Skywork-o1-Open-PRM-Qwen-2.5-7B"
    "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    # "Qwen2.5-Math-PRM-7B"
)

# 数据集配置 (任务名, batch_size)
DATASETS=(
    "AIME24 30"
    "AMC23 40"
    # "MATH 500"
)

# Branch宽度配置
BRANCH_WIDTHS=(
    8
    4
    2
)

# Num_seq配置
NUM_SEQ_VALUES=(
    2
    4
    1
)

# 日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 等待函数
wait_and_log() {
    local seconds=$1
    log_message "等待 ${seconds} 秒..."
    sleep $seconds
    log_message "等待完成"
}

# 停止现有服务
stop_services() {
    log_message "停止现有服务..."
    pkill -f "controller.py" 2>/dev/null || true
    pkill -f "model_worker.py" 2>/dev/null || true
    wait_and_log 10
}

# 启动服务
start_services() {
    local policy_path=$1
    local reward_path=$2
    
    log_message "启动服务: Policy=${policy_path##*/}, Reward=${reward_path##*/}, GPUs=${N_GPUS}"
    
    export VALUE_MODEL_PATH="$reward_path"
    export POLICY_MODEL_PATH="$policy_path"
    
    # 根据 N_GPUS 参数选择对应的服务脚本
    local serve_script="scripts/serve_gpu2_2-3.sh"
    
    if [ ! -f "$serve_script" ]; then
        log_message "错误: 服务脚本不存在: $serve_script"
        log_message "可用的GPU配置: 1, 2, 3, 4"
        exit 1
    fi
    
    # 后台启动服务
    bash "$serve_script" "$policy_path" "$reward_path" "$HOST_ADDR" "$CONTROLLER_PORT" "$WORKER_BASE_PORT" &
    SERVICE_PID=$!
    
    wait_and_log 30
}

# 删除当前组合对应的输出目录（避免不完整输出残留）
cleanup_output_dir() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local branch_width=$4
    local num_seq=$5

    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"

}

# 运行评估
run_evaluation() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local batch_size=$4
    local branch_width=$5
    local num_seq=$6
    local max_retries=3
    local attempt=1
    
    while [ $attempt -le $max_retries ]; do
        log_message "运行评估 (第 ${attempt}/${max_retries} 次): Policy=${policy_path##*/}, Reward=${reward_path##*/}, Task=${task}, BatchSize=${batch_size}, Branch=${branch_width}, NumSeq=${num_seq}"

        timeout --signal=TERM --kill-after=120 "${EVAL_TIMEOUT_SECONDS}" bash scripts/run.sh \
            --method beam_search \
            --LM "$policy_path" \
            --RM "$reward_path" \
            --width "$branch_width" \
            --num_seq "$num_seq" \
            --temperature 0.7 \
            --max_new_tokens 131072 \
            --tree_max_depth 16384 \
            --bs "$batch_size" \
            --mt 120 \
            --n_gpus "$N_GPUS" \
            --double_line_break 1 \
            --local 0 \
            --beam-log 1 \
            --task "$task"

        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_message "评估完成: ${task}, Branch=${branch_width}, NumSeq=${num_seq} - 成功 (第 ${attempt} 次尝试)"
            break
        else
            if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
                log_message "检测到评估卡死/超时: ${task}, Branch=${branch_width}, NumSeq=${num_seq} - 退出码: ${exit_code}"
            else
                log_message "评估失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq} - 退出码: ${exit_code}"
            fi

            # 失败或卡死时：停止服务 -> 删除不完整输出 -> 重启服务
            stop_services
            cleanup_output_dir "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq"
            start_services "$policy_path" "$reward_path"

            attempt=$((attempt + 1))
            if [ $attempt -le $max_retries ]; then
                log_message "将在 30 秒后进行下一次重试..."
                wait_and_log 30
            else
                log_message "评估最终失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq} - 已达到最大重试次数 (${max_retries})"
            fi
        fi
    done

    wait_and_log 30
}

# 主执行函数
main() {
    log_message "开始全组合评估"
    log_message "Policy模型数量: ${#POLICY_MODELS[@]}"
    log_message "Reward模型数量: ${#REWARD_MODELS[@]}"
    log_message "数据集数量: ${#DATASETS[@]}"
    log_message "Branch宽度数量: ${#BRANCH_WIDTHS[@]}"
    log_message "Num_seq数量: ${#NUM_SEQ_VALUES[@]}"
    
    local total_combinations=$((${#POLICY_MODELS[@]} * ${#REWARD_MODELS[@]} * ${#DATASETS[@]} * ${#BRANCH_WIDTHS[@]} * ${#NUM_SEQ_VALUES[@]}))
    log_message "总组合数: ${total_combinations}"
    
    local current_combination=0
    
    # 遍历所有Policy模型
    for policy_model in "${POLICY_MODELS[@]}"; do
        policy_path="${BASE_PATH}/${policy_model}"
        
        # 遍历所有Reward模型
        for reward_model in "${REWARD_MODELS[@]}"; do
            reward_path="${BASE_PATH}/${reward_model}"
            
            # 停止之前的服务并启动新的服务
            stop_services
            start_services "$policy_path" "$reward_path"
            
            # 遍历所有数据集
            for dataset_config in "${DATASETS[@]}"; do
                read -r task batch_size <<< "$dataset_config"
                
                # 遍历所有Branch宽度
                for branch_width in "${BRANCH_WIDTHS[@]}"; do
                    # 遍历所有Num_seq值
                    for num_seq in "${NUM_SEQ_VALUES[@]}"; do
                        # 检查 branch_width 是否大于 num_seq 且能被 num_seq 整除
                        if [ $branch_width -le $num_seq ]; then
                            log_message "跳过不兼容组合: Branch=${branch_width}, NumSeq=${num_seq} (branch_width 必须大于 num_seq)"
                            continue
                        fi
                        if [ $((branch_width % num_seq)) -ne 0 ]; then
                            log_message "跳过不兼容组合: Branch=${branch_width}, NumSeq=${num_seq} (branch_width 必须能被 num_seq 整除)"
                            continue
                        fi
                        
                        current_combination=$((current_combination + 1))
                        
                        log_message "=== 组合 ${current_combination}/${total_combinations} ==="
                        log_message "Policy: ${policy_model}"
                        log_message "Reward: ${reward_model}"
                        log_message "Dataset: ${task}"
                        log_message "Branch Width: ${branch_width}"
                        log_message "Num Seq: ${num_seq}"
                        
                        run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq"
                    done
                done
            done
        done
    done
    
    # 清理
    stop_services
    log_message "所有评估完成！"
}

# 检查必需的脚本文件
check_requirements() {
    local serve_script="scripts/serve_gpu${N_GPUS}.sh"
    
    if [ ! -f "$serve_script" ]; then
        log_message "错误: 服务脚本不存在: $serve_script"
        log_message "可用的GPU配置: 1, 2, 3, 4"
        log_message "当前配置: N_GPUS=${N_GPUS}"
        exit 1
    fi
    
    if [ ! -f "scripts/run.sh" ]; then
        log_message "错误: scripts/run.sh 不存在"
        exit 1
    fi
    
    for policy_model in "${POLICY_MODELS[@]}"; do
        if [ ! -d "${BASE_PATH}/${policy_model}" ]; then
            log_message "错误: Policy模型不存在: ${BASE_PATH}/${policy_model}"
            exit 1
        fi
    done
    
    for reward_model in "${REWARD_MODELS[@]}"; do
        if [ ! -d "${BASE_PATH}/${reward_model}" ]; then
            log_message "错误: Reward模型不存在: ${BASE_PATH}/${reward_model}"
            exit 1
        fi
    done
}

# 脚本入口
if [ "$1" = "--dry-run" ]; then
    log_message "干跑模式 - 只显示将要执行的组合"
    for policy_model in "${POLICY_MODELS[@]}"; do
        for reward_model in "${REWARD_MODELS[@]}"; do
            for dataset_config in "${DATASETS[@]}"; do
                read -r task batch_size <<< "$dataset_config"
                for branch_width in "${BRANCH_WIDTHS[@]}"; do
                    for num_seq in "${NUM_SEQ_VALUES[@]}"; do
                        if [ $branch_width -le $num_seq ]; then
                            echo "SKIP (Branch <= NumSeq): Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}"
                        elif [ $((branch_width % num_seq)) -ne 0 ]; then
                            echo "SKIP (不能整除): Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}"
                        else
                            echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}"
                        fi
                    done
                done
            done
        done
    done
    exit 0
fi

# 检查要求并运行（不记录日志到文件）
check_requirements
main