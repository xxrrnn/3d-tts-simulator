#!/bin/bash
# 副本：在另一张物理 GPU 上跑全组合评估，与 eval_all_combinations_straggler.sh 并行。
# - 使用独立 CONTROLLER_PORT / WORKER_BASE_PORT / tmux 会话，避免与默认实例冲突。
# - stop_services 仅结束本会话，不会对另一路 pkill。
# 改用 GPU3：使用 scripts/eval_all_combinations_straggler_gpu3.sh（独立端口与 output_gpu3）。
# 结果写入 output_gpu2（见 OUTPUT_BASE_DIR），经 run.sh 的 EVAL_SAVE_DIR 传入 evaluate.py，不与默认 src/output 冲突。

# 基础路径配置
BASE_PATH="/DISK1/data/rnxu_24/Paper/models"
export LOGDIR="/DISK1/data/rnxu_24/Paper/xlong_32/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/logs"
export HOST_ADDR="0.0.0.0"
export CONTROLLER_PORT=10034
export WORKER_BASE_PORT=10101
export TMUX_SESSION="tts_iso_gpu2"
PHYSICAL_GPU_INDEX=2

# 逻辑仍为单卡 worker 布局（隔离脚本内只暴露一张物理卡）
N_GPUS=1

# output 子目录名须与 evaluate.py 一致：{tree_max_depth}_{tree_max_width}_{num_sequence}（同 run.sh --tree_max_depth）
EVAL_TREE_MAX_DEPTH=40

# 单卡先后加载 RM+Policy，过短易报「无法获取 worker」；仍失败可再加大
SERVICE_START_WAIT_SECONDS=120

# 脚本路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_BASE_DIR="${SRC_DIR}/output_gpu2"
export EVAL_SAVE_DIR="${OUTPUT_BASE_DIR}"
CHECK_SCRIPT="${SCRIPT_DIR}/process/check_incomplete_questions.py"

# 卡死检测：单次评估最大运行时长（秒）
EVAL_TIMEOUT_SECONDS=720000

# 0405：修改为Policy + Reward 固定组合（仅跑以下三对组合）
# 格式: "Policy目录名|Reward目录名"
POLICY_REWARD_PAIRS=(
    "Qwen2.5-Math-1.5B-Instruct|math-shepherd-mistral-7b-prm"
    "Qwen2.5-Math-7B-Instruct|Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    "Qwen2.5-Math-1.5B-Instruct|Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
)

# 数据集配置 (任务名, batch_size)
# batch_size: 一次性分配给评估的题目数量（降低以减少内存占用）
DATASETS=(
    "AIME24 30"  # 从 30 降到 15（减少50%） straggler出现在question19
    "AMC23 40"   # 从 40 降到 20（减少50%）
    # "MATH 500"
)

# (tree_max_width, num_seq) 仅遍历下列组合
BEAM_WIDTH_NUMSEQ_PAIRS=(
    "8 1"
    "4 1"
    "2 1"
    "8 2"
    "4 2"
)

# 评估随机种子（传入 run.sh --seed → evaluate.py）
EVAL_SEED=42

# ========== STRAGGLER参数扫描配置 ==========
# PRUNE=0: 关闭straggler，其他参数无意义，只测试一个配置
# PRUNE=1: 开启straggler，遍历不同的ratio和min_tokens组合
STRAGGLER_PRUNE_VALUES=(
    0    # 关闭
    1    # 开启
)

# 仅当 PRUNE=1 时，以下参数才会被遍历
STRAGGLER_LENGTH_RATIO_VALUES=(
    # 1.2
    1.5
    # 2.0
)

STRAGGLER_MIN_TOKENS_VALUES=(
    # 80
    100
    # 150
    # 200
)
# ===========================================

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

# 仅停止本实例的 tmux（不 pkill，以免误杀另一路评估）
stop_services() {
    log_message "停止本实例服务 (tmux: ${TMUX_SESSION})..."
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        tmux kill-session -t "${TMUX_SESSION}"
        log_message "已结束 tmux 会话 ${TMUX_SESSION}"
    else
        log_message "tmux 会话 ${TMUX_SESSION} 不存在，跳过"
    fi
    wait_and_log 15
}

# 启动服务
start_services() {
    local policy_path=$1
    local reward_path=$2
    
    log_message "启动服务: Policy=${policy_path##*/}, Reward=${reward_path##*/}, 物理GPU=${PHYSICAL_GPU_INDEX}, controller_port=${CONTROLLER_PORT}, tmux=${TMUX_SESSION}"
    
    export VALUE_MODEL_PATH="$reward_path"
    export POLICY_MODEL_PATH="$policy_path"
    
    local serve_script="scripts/serve_gpu1_isolated.sh"
    if [ ! -f "$serve_script" ]; then
        log_message "错误: 服务脚本不存在: $serve_script"
        exit 1
    fi
    
    bash "$serve_script" "$policy_path" "$reward_path" "$HOST_ADDR" "$CONTROLLER_PORT" "$WORKER_BASE_PORT" "$TMUX_SESSION" "$PHYSICAL_GPU_INDEX" &
    SERVICE_PID=$!
    
    # 增加等待时间，确保大模型有足够时间加载到GPU，并且 controller 和 workers 完全启动
    log_message "等待 controller 和 workers 启动并加载模型（${SERVICE_START_WAIT_SECONDS}s）..."
    wait_and_log "${SERVICE_START_WAIT_SECONDS}"
}

# 删除锁文件（保留已完成的结果）
cleanup_lock_files() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local branch_width=$4
    local num_seq=$5

    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    
    # 构建输出目录路径
    local output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}"
    local lock_dir="${output_dir}/lock_dir"
    
    if [ ! -d "$lock_dir" ]; then
        log_message "锁文件目录不存在: $lock_dir"
        return 0
    fi
    
    # 查找并删除所有锁文件
    local lock_count=$(find "$lock_dir" -name "*.lock" -type f 2>/dev/null | wc -l)
    
    if [ $lock_count -gt 0 ]; then
        log_message "发现 $lock_count 个锁文件，正在删除..."
        find "$lock_dir" -name "*.lock" -type f -delete
        log_message "✓ 已删除所有锁文件"
    else
        log_message "没有发现锁文件"
    fi
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
    
    # 构建输出目录路径
    local output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}"
    
    if [ -d "$output_dir" ]; then
        log_message "删除输出目录: $output_dir"
        rm -rf "$output_dir"
    fi
}

# 检查评估是否已完成
check_evaluation_completed() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local branch_width=$4
    local num_seq=$5
    local straggler_prune=$6
    local straggler_ratio=$7
    local straggler_min=$8
    
    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    
    # 构建重命名后的目录路径
    local renamed_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}"
    local result_file="${renamed_dir}/avg_result.json"
    
    # 检查重命名后的目录中的avg_result.json是否存在
    if [ -f "$result_file" ]; then
        log_message "✓ 跳过已完成的评估: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}"
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 检查评估是否部分完成
check_partial_completion() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local branch_width=$4
    local num_seq=$5
    
    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    
    local output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}"
    
    # 检查是否有部分结果
    if [ -d "$output_dir" ]; then
        local question_count=$(find "$output_dir" -maxdepth 1 -type d -name "question_*" 2>/dev/null | wc -l)
        if [ $question_count -gt 0 ]; then
            log_message "发现 $question_count 个已完成的问题，将保留并继续"
            return 0  # 有部分结果
        fi
    fi
    return 1  # 没有部分结果
}

# 重命名输出目录
rename_output_dir() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local branch_width=$4
    local num_seq=$5
    local straggler_prune=$6
    local straggler_ratio=$7
    local straggler_min=$8
    
    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    
    # 原始输出目录
    local output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}"
    
    # 重命名后的目录（添加straggler参数后缀）
    local new_output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}"
    
    if [ -d "$output_dir" ]; then
        log_message "重命名输出目录: $output_dir -> $new_output_dir"
        mv "$output_dir" "$new_output_dir"
        log_message "✓ 输出目录重命名完成"
    else
        log_message "警告: 输出目录不存在，无法重命名: $output_dir"
    fi
}

# 运行评估
run_evaluation() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local batch_size=$4
    local branch_width=$5
    local num_seq=$6
    local straggler_prune=$7
    local straggler_ratio=$8
    local straggler_min=$9
    local max_retries=3
    local attempt=1
    
    while [ $attempt -le $max_retries ]; do
        log_message "运行评估 (第 ${attempt}/${max_retries} 次): Policy=${policy_path##*/}, Reward=${reward_path##*/}, Task=${task}, BatchSize=${batch_size}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}"
        # 0406:max_new_tokens限制为8192       
        timeout --signal=TERM --kill-after=120 "${EVAL_TIMEOUT_SECONDS}" bash scripts/run.sh \
            --method beam_search \
            --LM "$policy_path" \
            --RM "$reward_path" \
            --width "$branch_width" \
            --num_seq "$num_seq" \
            --temperature 0.7 \
            --max_new_tokens 8192 \
            --tree_max_depth 40 \
            --bs "$batch_size" \
            --mt 120000 \
            --n_gpus "$N_GPUS" \
            --double_line_break 1 \
            --local 0 \
            --beam-log 1 \
            --logprobs-topk 20 \
            --task "$task" \
            --seed "$EVAL_SEED" \
            --straggler_prune "$straggler_prune" \
            --straggler_length_ratio "$straggler_ratio" \
            --straggler_min_tokens "$straggler_min"

        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_message "评估完成: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min} - 成功 (第 ${attempt} 次尝试)"
            
            # 评估成功后，重命名输出目录
            rename_output_dir "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min"
            break
        else
            if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
                log_message "检测到评估卡死/超时: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min} - 退出码: ${exit_code}"
            else
                log_message "评估失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min} - 退出码: ${exit_code}"
            fi

            # 失败时的智能清理策略
            stop_services
            
            # 检查是否有部分完成的结果
            if check_partial_completion "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq"; then
                # 有部分结果：只删除锁文件，保留已完成的部分
                log_message "采用智能重试：保留已完成结果，只删除锁文件"
                cleanup_lock_files "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq"
            else
                # 没有结果或结果很少：删除整个目录
                log_message "没有有效结果，删除整个输出目录重新开始"
                cleanup_output_dir "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq"
            fi
            
            start_services "$policy_path" "$reward_path"

            attempt=$((attempt + 1))
            if [ $attempt -le $max_retries ]; then
                log_message "将在 60 秒后进行下一次重试..."
                wait_and_log 60
            else
                log_message "评估最终失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min} - 已达到最大重试次数 (${max_retries})"
            fi
        fi
    done

    wait_and_log 60
}

# 主执行函数
main() {
    log_message "开始全组合评估 [GPU2 隔离实例]（固定 Policy–Reward 对 + Straggler参数扫描）"
    log_message "CONTROLLER_PORT=${CONTROLLER_PORT} WORKER_BASE_PORT=${WORKER_BASE_PORT} TMUX_SESSION=${TMUX_SESSION} PHYSICAL_GPU_INDEX=${PHYSICAL_GPU_INDEX} OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
    log_message "EVAL_SEED=${EVAL_SEED}"
    log_message "Policy–Reward 对数量: ${#POLICY_REWARD_PAIRS[@]}"
    log_message "数据集数量: ${#DATASETS[@]}"
    log_message "Beam(宽×num_seq) 组合数: ${#BEAM_WIDTH_NUMSEQ_PAIRS[@]}"
    log_message "Straggler配置: Prune=0 (1个配置) + Prune=1 (${#STRAGGLER_LENGTH_RATIO_VALUES[@]} ratios × ${#STRAGGLER_MIN_TOKENS_VALUES[@]} min_tokens)"
    
    # 计算实际组合数: prune=0 只有1个配置，prune=1 有 ratio × min_tokens 个配置
    local base_combinations=$((${#POLICY_REWARD_PAIRS[@]} * ${#DATASETS[@]} * ${#BEAM_WIDTH_NUMSEQ_PAIRS[@]}))
    local straggler_configs=$((1 + ${#STRAGGLER_LENGTH_RATIO_VALUES[@]} * ${#STRAGGLER_MIN_TOKENS_VALUES[@]}))
    local total_combinations=$((base_combinations * straggler_configs))
    log_message "总组合数: ${total_combinations} (基础${base_combinations} × Straggler配置${straggler_configs})"
    
    local current_combination=0
    
    for pair_line in "${POLICY_REWARD_PAIRS[@]}"; do
        IFS='|' read -r policy_model reward_model <<< "${pair_line}"
        policy_path="${BASE_PATH}/${policy_model}"
        reward_path="${BASE_PATH}/${reward_model}"
        
        stop_services
        start_services "$policy_path" "$reward_path"
        
        for dataset_config in "${DATASETS[@]}"; do
            read -r task batch_size <<< "$dataset_config"
            
            for beam_pair in "${BEAM_WIDTH_NUMSEQ_PAIRS[@]}"; do
                read -r branch_width num_seq <<< "$beam_pair"
                if [ $branch_width -le $num_seq ]; then
                    log_message "跳过不兼容组合: Branch=${branch_width}, NumSeq=${num_seq} (branch_width 必须大于 num_seq)"
                    continue
                fi
                if [ $((branch_width % num_seq)) -ne 0 ]; then
                    log_message "跳过不兼容组合: Branch=${branch_width}, NumSeq=${num_seq} (branch_width 必须能被 num_seq 整除)"
                    continue
                fi

                # 扫描 Straggler 参数组合
                for straggler_prune in "${STRAGGLER_PRUNE_VALUES[@]}"; do
                        # 当 prune=0 时，其他参数不起作用，只运行一次即可
                        if [ $straggler_prune -eq 0 ]; then
                            # 使用默认值（实际不会被使用，仅用于命名）
                            straggler_ratio=0
                            straggler_min=0
                            
                            current_combination=$((current_combination + 1))
                            
                            log_message "=== 组合 ${current_combination}/${total_combinations} ==="
                            log_message "Policy: ${policy_model}"
                            log_message "Reward: ${reward_model}"
                            log_message "Dataset: ${task}"
                            log_message "Branch Width: ${branch_width}"
                            log_message "Num Seq: ${num_seq}"
                            log_message "Straggler Prune: ${straggler_prune} (关闭)"
                            
                            if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min"; then
                                continue
                            fi
                            
                            run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min"
                        else
                            # prune=1 时，遍历所有 ratio 和 min_tokens 组合
                            for straggler_ratio in "${STRAGGLER_LENGTH_RATIO_VALUES[@]}"; do
                                for straggler_min in "${STRAGGLER_MIN_TOKENS_VALUES[@]}"; do
                                    current_combination=$((current_combination + 1))
                                    
                                    log_message "=== 组合 ${current_combination}/${total_combinations} ==="
                                    log_message "Policy: ${policy_model}"
                                    log_message "Reward: ${reward_model}"
                                    log_message "Dataset: ${task}"
                                    log_message "Branch Width: ${branch_width}"
                                    log_message "Num Seq: ${num_seq}"
                                    log_message "Straggler Prune: ${straggler_prune} (开启)"
                                    log_message "Straggler Length Ratio: ${straggler_ratio}"
                                    log_message "Straggler Min Tokens: ${straggler_min}"
                                    
                                    if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min"; then
                                        continue
                                    fi
                                    
                                    run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min"
                                done
                            done
                        fi
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
    local serve_script="scripts/serve_gpu1_isolated.sh"
    if [ ! -f "$serve_script" ]; then
        log_message "错误: 服务脚本不存在: $serve_script"
        exit 1
    fi
    
    if [ ! -f "scripts/run.sh" ]; then
        log_message "错误: scripts/run.sh 不存在"
        exit 1
    fi
    
    for pair_line in "${POLICY_REWARD_PAIRS[@]}"; do
        IFS='|' read -r policy_model reward_model <<< "${pair_line}"
        if [ ! -d "${BASE_PATH}/${policy_model}" ]; then
            log_message "错误: Policy模型不存在: ${BASE_PATH}/${policy_model}"
            exit 1
        fi
        if [ ! -d "${BASE_PATH}/${reward_model}" ]; then
            log_message "错误: Reward模型不存在: ${BASE_PATH}/${reward_model}"
            exit 1
        fi
    done
}

# 脚本入口
if [ "$1" = "--dry-run" ]; then
    log_message "干跑模式 - 只显示将要执行的组合"
    for pair_line in "${POLICY_REWARD_PAIRS[@]}"; do
        IFS='|' read -r policy_model reward_model <<< "${pair_line}"
        for dataset_config in "${DATASETS[@]}"; do
            read -r task batch_size <<< "$dataset_config"
            for beam_pair in "${BEAM_WIDTH_NUMSEQ_PAIRS[@]}"; do
                read -r branch_width num_seq <<< "$beam_pair"
                if [ $branch_width -le $num_seq ]; then
                    echo "SKIP (Branch <= NumSeq): Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, Branch: ${branch_width}, NumSeq: ${num_seq}"
                    continue
                fi
                if [ $((branch_width % num_seq)) -ne 0 ]; then
                    echo "SKIP (不能整除): Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, Branch: ${branch_width}, NumSeq: ${num_seq}"
                    continue
                fi

                for straggler_prune in "${STRAGGLER_PRUNE_VALUES[@]}"; do
                        if [ $straggler_prune -eq 0 ]; then
                            # prune=0 只运行一个配置
                            echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=0 (关闭)"
                        else
                            # prune=1 遍历所有参数组合
                            for straggler_ratio in "${STRAGGLER_LENGTH_RATIO_VALUES[@]}"; do
                                for straggler_min in "${STRAGGLER_MIN_TOKENS_VALUES[@]}"; do
                                    echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=1 RATIO=${straggler_ratio} MIN=${straggler_min}"
                                done
                            done
                        fi
                done
            done
        done
    done
    exit 0
fi

# 检查要求并运行
check_requirements
main
