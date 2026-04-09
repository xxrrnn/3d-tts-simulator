#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate tts

# 路径相对本仓库根目录（REPO_ROOT）；BASE_PATH 指向 Paper/models（与仓库同级目录下的 models）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../../.." && pwd)"
BASE_PATH="${REPO_ROOT}/../../models"
export LOGDIR="${REPO_ROOT}/3d-tts-sw/compute-optimal-tts/src/logs"
export HOST_ADDR="0.0.0.0"
export CONTROLLER_PORT=10014
export WORKER_BASE_PORT=10081

# GPU配置 (统一管理)
N_GPUS=1  # 可选值: 1, 2, 3, 4

OUTPUT_BASE_DIR="${SRC_DIR}/output"
CHECK_SCRIPT="${SCRIPT_DIR}/process/check_incomplete_questions.py"

# 卡死检测：单次评估最大运行时长（秒）
EVAL_TIMEOUT_SECONDS=720000

# 0405：修改为Policy + Reward 固定组合（仅跑以下三对组合）
# 格式: "Policy目录名|Reward目录名"
POLICY_REWARD_PAIRS=(
    "Qwen2.5-Math-1.5B-Instruct|math-shepherd-mistral-7b-prm"
    # "Qwen2.5-Math-7B-Instruct|Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    # "Qwen2.5-Math-1.5B-Instruct|Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
)

# 数据集配置 (任务名, batch_size)
# batch_size: 一次性分配给评估的题目数量（降低以减少内存占用）
DATASETS=(
    "AIME24 5"  # 从 30 降到 15（减少50%） straggler出现在question19
    # "AMC23 40"   # 从 40 降到 20（减少50%）
    # "MATH 500"
)

# (tree_max_width, num_seq) 仅遍历下列组合
BEAM_WIDTH_NUMSEQ_PAIRS=(
    "8 1"
    "4 1"
    # "2 1"
    # "8 2"
    # "4 2"
)

# 评估随机种子（传入 run.sh --seed → evaluate.py）
# 同一 straggler 配置重复跑 EVAL_REPEAT_COUNT 次时，每次均使用 EVAL_SEED（不随 run 递增）
EVAL_SEED=42

# 0408：增加循环次数，循环三次取均值。
# 每个 (Policy, Reward, Task, Beam, Straggler) 配置连续评估次数；取平均时请合并各 _run* 目录下的 avg_result.json
# 为 1 时不改输出目录名（与改前一致）；大于 1 时在目录名末尾追加 _run1 … _runN
EVAL_REPEAT_COUNT=3

# output 子目录名第一段须与 run.sh 里 --tree_max_depth 及 evaluate.py 中 params 一致（tree_max_depth_width_num_seq）
EVAL_TREE_MAX_DEPTH=40

# ========== STRAGGLER参数扫描配置 ==========
# PRUNE=0: 关闭straggler，其他参数无意义，只测试一个配置
# PRUNE=1: 开启straggler，遍历不同的ratio和min_tokens组合
STRAGGLER_PRUNE_VALUES=(
    0    # 关闭
    # 1    # 开启
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

# 兄弟分支 PRM 门控：仅当 gate=1 且「非 straggler 分支 PRM 分最大值 > 阈值」时才剪 straggler
# gate=0 时与改前一致（仅长度倍率）；此时 threshold 仅用于输出目录占位，传 0 即可
STRAGGLER_OTHER_REWARD_GATE_VALUES=(
    0
    # 1
)
STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES=(
    0.9
)

# 跨 step 延迟 straggler 剪枝（须 PRUNE=1；与 tree 中 straggler_deferred_prune 对应）。PRUNE=0 时脚本固定传 0。
STRAGGLER_DEFERRED_PRUNE_VALUES=(
    0
    # 1
)

REWARD_SUBCONFIG_COUNT=0
for _srg in "${STRAGGLER_OTHER_REWARD_GATE_VALUES[@]}"; do
    if [ "${_srg}" -eq 0 ]; then
        REWARD_SUBCONFIG_COUNT=$((REWARD_SUBCONFIG_COUNT + 1))
    else
        REWARD_SUBCONFIG_COUNT=$((REWARD_SUBCONFIG_COUNT + ${#STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES[@]}))
    fi
done
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

# 健康检查函数 - 检查服务是否就绪
check_service_health() {
    local max_attempts=30  # 最多检查30次
    local attempt=1
    
    log_message "开始健康检查..."
    
    while [ $attempt -le $max_attempts ]; do
        # 检查 controller
        if ! curl -s "http://${HOST_ADDR}:${CONTROLLER_PORT}/test_connection" > /dev/null 2>&1; then
            log_message "尝试 ${attempt}/${max_attempts}: Controller 未就绪，继续等待..."
            sleep 5
            attempt=$((attempt + 1))
            continue
        fi
        
        # 检查 reward model worker (10081)
        if ! curl -s -X POST "http://${HOST_ADDR}:10081/worker_get_status" -H "Content-Type: application/json" --max-time 5 > /dev/null 2>&1; then
            log_message "尝试 ${attempt}/${max_attempts}: Reward Model Worker 未就绪，继续等待..."
            sleep 5
            attempt=$((attempt + 1))
            continue
        fi
        
        # 检查 policy model worker (10082)
        if ! curl -s -X POST "http://${HOST_ADDR}:10082/worker_get_status" -H "Content-Type: application/json" --max-time 5 > /dev/null 2>&1; then
            log_message "尝试 ${attempt}/${max_attempts}: Policy Model Worker 未就绪，继续等待..."
            sleep 5
            attempt=$((attempt + 1))
            continue
        fi
        
        log_message "✓ 所有服务健康检查通过！"
        return 0
    done
    
    log_message "警告: 健康检查超时，但将继续执行..."
    return 1
}

# 停止现有服务
stop_services() {
    log_message "停止现有服务..."
    # 停止所有相关进程
    pkill -f "controller.py" 2>/dev/null || true
    pkill -f "vllm_worker" 2>/dev/null || true
    pkill -f "reward_model_worker" 2>/dev/null || true
    pkill -f "model_worker.py" 2>/dev/null || true
    # 杀掉 tmux session
    tmux kill-session -t tts 2>/dev/null || true
    wait_and_log 15
}

# 启动服务
start_services() {
    local policy_path=$1
    local reward_path=$2
    
    log_message "启动服务: Policy=${policy_path##*/}, Reward=${reward_path##*/}, GPUs=${N_GPUS}"
    
    export VALUE_MODEL_PATH="$reward_path"
    export POLICY_MODEL_PATH="$policy_path"
    
    # 根据 N_GPUS 参数选择对应的服务脚本
    local serve_script="scripts/serve_gpu${N_GPUS}.sh"
    
    if [ ! -f "$serve_script" ]; then
        log_message "错误: 服务脚本不存在: $serve_script"
        log_message "可用的GPU配置: 1, 2, 3, 4"
        exit 1
    fi
    
    # 后台启动服务
    bash "$serve_script" "$policy_path" "$reward_path" "$HOST_ADDR" "$CONTROLLER_PORT" "$WORKER_BASE_PORT" &
    SERVICE_PID=$!
    
    # 等待服务启动并通过健康检查
    log_message "等待 controller 和 workers 启动并加载模型..."
    wait_and_log 30  # 先等待30秒让服务有时间启动
    check_service_health  # 然后进行健康检查
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

# 单次重复运行的输出目录后缀（EVAL_REPEAT_COUNT>1 时为 _run${run_idx}）
eval_repeat_run_suffix() {
    local run_idx=$1
    if [ "${EVAL_REPEAT_COUNT}" -gt 1 ]; then
        echo "_run${run_idx}"
    else
        echo ""
    fi
}

# 检查评估是否已完成（run_idx：第几次重复，1..EVAL_REPEAT_COUNT）
check_evaluation_completed() {
    local policy_path=$1
    local reward_path=$2
    local task=$3
    local branch_width=$4
    local num_seq=$5
    local straggler_prune=$6
    local straggler_ratio=$7
    local straggler_min=$8
    local straggler_other_gate=$9
    local straggler_other_thr=${10}
    local straggler_deferred=${11:-0}
    local run_idx=${12:-1}
    
    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    local run_suffix
    run_suffix=$(eval_repeat_run_suffix "$run_idx")
    
    # 构建重命名后的目录路径
    local renamed_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}_${straggler_other_gate}_${straggler_other_thr}_def${straggler_deferred}${run_suffix}"
    local result_file="${renamed_dir}/avg_result.json"
    
    # 检查重命名后的目录中的avg_result.json是否存在
    if [ -f "$result_file" ]; then
        log_message "✓ 跳过已完成的评估: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}, run=${run_idx}/${EVAL_REPEAT_COUNT}"
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
    local straggler_other_gate=$9
    local straggler_other_thr=${10}
    local straggler_deferred=${11:-0}
    local run_idx=${12:-1}
    
    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    local run_suffix
    run_suffix=$(eval_repeat_run_suffix "$run_idx")
    
    # 原始输出目录
    local output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}"
    
    # 重命名后的目录（添加straggler参数后缀，及可选的重复次序后缀）
    local new_output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}_${straggler_other_gate}_${straggler_other_thr}_def${straggler_deferred}${run_suffix}"
    
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
    local straggler_other_gate=${10}
    local straggler_other_thr=${11}
    local straggler_deferred=${12:-0}
    local run_idx=${13:-1}
    local run_seed=$EVAL_SEED
    local max_retries=3
    local attempt=1
    
    while [ $attempt -le $max_retries ]; do
        log_message "运行评估 (重试 ${attempt}/${max_retries}, 配置重复 ${run_idx}/${EVAL_REPEAT_COUNT}, seed=${run_seed}): Policy=${policy_path##*/}, Reward=${reward_path##*/}, Task=${task}, BatchSize=${batch_size}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}"
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
            --seed "$run_seed" \
            --straggler_prune "$straggler_prune" \
            --straggler_length_ratio "$straggler_ratio" \
            --straggler_min_tokens "$straggler_min" \
            --straggler_prune_other_reward_gate "$straggler_other_gate" \
            --straggler_prune_other_reward_threshold "$straggler_other_thr" \
            --straggler_deferred_prune "$straggler_deferred" \
            --deterministic 1

        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_message "评估完成: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 成功 (重试第 ${attempt} 次)"
            
            # 评估成功后，重命名输出目录
            rename_output_dir "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$run_idx"
            break
        else
            if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
                log_message "检测到评估卡死/超时: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 退出码: ${exit_code}"
            else
                log_message "评估失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 退出码: ${exit_code}"
            fi

            # 失败时的智能清理策略
            stop_services
            
            log_message "采用智能重试：保留已完成结果，只删除锁文件"
            cleanup_lock_files "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq"
            
            start_services "$policy_path" "$reward_path"

            attempt=$((attempt + 1))
            if [ $attempt -le $max_retries ]; then
                log_message "将在 10 秒后进行下一次重试..."
                wait_and_log 10
            else
                log_message "评估最终失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 已达到最大重试次数 (${max_retries})"
            fi
        fi
    done

    wait_and_log 20
}

# 主执行函数
main() {
    if ! [[ "${EVAL_REPEAT_COUNT:-x}" =~ ^[1-9][0-9]*$ ]]; then
        log_message "错误: EVAL_REPEAT_COUNT 须为正整数，当前=${EVAL_REPEAT_COUNT}"
        exit 1
    fi

    log_message "开始全组合评估（固定 Policy–Reward 对 + Straggler参数扫描）"
    log_message "EVAL_SEED=${EVAL_SEED}，每配置重复次数 EVAL_REPEAT_COUNT=${EVAL_REPEAT_COUNT}（各次 run 均使用同一 seed）"
    log_message "Policy–Reward 对数量: ${#POLICY_REWARD_PAIRS[@]}"
    log_message "数据集数量: ${#DATASETS[@]}"
    log_message "Beam(宽×num_seq) 组合数: ${#BEAM_WIDTH_NUMSEQ_PAIRS[@]}"
    log_message "Straggler配置: Prune=0 (1) + Prune=1 (${#STRAGGLER_LENGTH_RATIO_VALUES[@]}×${#STRAGGLER_MIN_TOKENS_VALUES[@]}×兄弟PRM子配置=${REWARD_SUBCONFIG_COUNT}×def=${#STRAGGLER_DEFERRED_PRUNE_VALUES[@]}；gate=0 仅长度门控，gate=1 再扫 threshold)"
    
    # prune=0: 1；prune=1: ratio×min×(各 gate 子配置数之和)×延迟剪枝档位数
    local base_combinations=$((${#POLICY_REWARD_PAIRS[@]} * ${#DATASETS[@]} * ${#BEAM_WIDTH_NUMSEQ_PAIRS[@]}))
    local straggler_configs=$((1 + ${#STRAGGLER_LENGTH_RATIO_VALUES[@]} * ${#STRAGGLER_MIN_TOKENS_VALUES[@]} * REWARD_SUBCONFIG_COUNT * ${#STRAGGLER_DEFERRED_PRUNE_VALUES[@]}))
    local total_combinations=$((base_combinations * straggler_configs * EVAL_REPEAT_COUNT))
    log_message "总运行次数: ${total_combinations} (基础${base_combinations} × Straggler配置${straggler_configs} × 每配置重复${EVAL_REPEAT_COUNT})"
    
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
                            straggler_other_gate=0
                            straggler_other_thr=0
                            straggler_deferred=0
                            
                            for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                            current_combination=$((current_combination + 1))
                            
                            log_message "=== 运行 ${current_combination}/${total_combinations}（配置重复 ${repeat_idx}/${EVAL_REPEAT_COUNT}，seed=${EVAL_SEED}）==="
                            log_message "Policy: ${policy_model}"
                            log_message "Reward: ${reward_model}"
                            log_message "Dataset: ${task}"
                            log_message "Branch Width: ${branch_width}"
                            log_message "Num Seq: ${num_seq}"
                            log_message "Straggler Prune: ${straggler_prune} (关闭)"
                            
                            if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$repeat_idx"; then
                                continue
                            fi
                            
                            run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$repeat_idx"
                            done
                        else
                            # prune=1 时，遍历所有 ratio 和 min_tokens 组合
                            for straggler_ratio in "${STRAGGLER_LENGTH_RATIO_VALUES[@]}"; do
                                for straggler_min in "${STRAGGLER_MIN_TOKENS_VALUES[@]}"; do
                                    for straggler_other_gate in "${STRAGGLER_OTHER_REWARD_GATE_VALUES[@]}"; do
                                        if [ "$straggler_other_gate" -eq 0 ]; then
                                            straggler_other_thr=0
                                            for straggler_deferred in "${STRAGGLER_DEFERRED_PRUNE_VALUES[@]}"; do
                                            for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                                            current_combination=$((current_combination + 1))
                                            
                                            log_message "=== 运行 ${current_combination}/${total_combinations}（配置重复 ${repeat_idx}/${EVAL_REPEAT_COUNT}，seed=${EVAL_SEED}）==="
                                            log_message "Policy: ${policy_model}"
                                            log_message "Reward: ${reward_model}"
                                            log_message "Dataset: ${task}"
                                            log_message "Branch Width: ${branch_width}"
                                            log_message "Num Seq: ${num_seq}"
                                            log_message "Straggler Prune: ${straggler_prune} (开启)"
                                            log_message "Straggler Length Ratio: ${straggler_ratio}"
                                            log_message "Straggler Min Tokens: ${straggler_min}"
                                            log_message "Straggler 兄弟PRM门控: 关闭 (og=0 thr=0)"
                                            log_message "Straggler 延迟剪枝 def: ${straggler_deferred}"
                                            
                                            if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$repeat_idx"; then
                                                continue
                                            fi
                                            
                                            run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$repeat_idx"
                                            done
                                            done
                                        else
                                            for straggler_other_thr in "${STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES[@]}"; do
                                                for straggler_deferred in "${STRAGGLER_DEFERRED_PRUNE_VALUES[@]}"; do
                                                for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                                                current_combination=$((current_combination + 1))
                                                
                                                log_message "=== 运行 ${current_combination}/${total_combinations}（配置重复 ${repeat_idx}/${EVAL_REPEAT_COUNT}，seed=${EVAL_SEED}）==="
                                                log_message "Policy: ${policy_model}"
                                                log_message "Reward: ${reward_model}"
                                                log_message "Dataset: ${task}"
                                                log_message "Branch Width: ${branch_width}"
                                                log_message "Num Seq: ${num_seq}"
                                                log_message "Straggler Prune: ${straggler_prune} (开启)"
                                                log_message "Straggler Length Ratio: ${straggler_ratio}"
                                                log_message "Straggler Min Tokens: ${straggler_min}"
                                                log_message "Straggler 兄弟PRM门控: 开启 og=1 threshold=${straggler_other_thr}"
                                                log_message "Straggler 延迟剪枝 def: ${straggler_deferred}"
                                                
                                                if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$repeat_idx"; then
                                                    continue
                                                fi
                                                
                                                run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$repeat_idx"
                                                done
                                                done
                                            done
                                        fi
                                    done
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
    log_message "干跑模式 - 只显示将要执行的组合（每配置 ${EVAL_REPEAT_COUNT} 次，各次 seed=${EVAL_SEED}；EVAL_REPEAT_COUNT 可改脚本顶部变量）"
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
                            for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                            echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=0 (关闭) og=0 thr=0 def=0, run=${repeat_idx}/${EVAL_REPEAT_COUNT}, seed=${EVAL_SEED}"
                            done
                        else
                            for straggler_ratio in "${STRAGGLER_LENGTH_RATIO_VALUES[@]}"; do
                                for straggler_min in "${STRAGGLER_MIN_TOKENS_VALUES[@]}"; do
                                    for straggler_other_gate in "${STRAGGLER_OTHER_REWARD_GATE_VALUES[@]}"; do
                                        if [ "$straggler_other_gate" -eq 0 ]; then
                                            for straggler_deferred in "${STRAGGLER_DEFERRED_PRUNE_VALUES[@]}"; do
                                            for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                                            echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=1 RATIO=${straggler_ratio} MIN=${straggler_min} og=0 thr=0 def=${straggler_deferred}, run=${repeat_idx}/${EVAL_REPEAT_COUNT}, seed=${EVAL_SEED}"
                                            done
                                            done
                                        else
                                            for straggler_other_thr in "${STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES[@]}"; do
                                                for straggler_deferred in "${STRAGGLER_DEFERRED_PRUNE_VALUES[@]}"; do
                                                for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                                                echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=1 RATIO=${straggler_ratio} MIN=${straggler_min} og=1 thr=${straggler_other_thr} def=${straggler_deferred}, run=${repeat_idx}/${EVAL_REPEAT_COUNT}, seed=${EVAL_SEED}"
                                                done
                                                done
                                            done
                                        fi
                                    done
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
