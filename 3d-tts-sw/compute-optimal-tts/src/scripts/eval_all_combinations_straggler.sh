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

# 仅 N_GPUS=1：指定本机物理 GPU 编号（供 serve_gpu1 与 run.sh/评估子进程继承；多卡请改用其它 serve_gpu*.sh，勿依赖此项）
if [ "${N_GPUS}" -eq 1 ]; then
    : "${CUDA_VISIBLE_DEVICES:=0}"
    export CUDA_VISIBLE_DEVICES
fi

OUTPUT_BASE_DIR="${SRC_DIR}/output"
CHECK_SCRIPT="${SCRIPT_DIR}/process/check_incomplete_questions.py"

# ========== 功耗记录配置 ==========
# 是否启用功耗记录（0=关闭，1=开启）
RECORD_POWER=1
# 功耗数据保存目录
POWER_OUTPUT_DIR="${SRC_DIR}/power"
# ================================

# 卡死检测：单次评估最大运行时长（秒）
EVAL_TIMEOUT_SECONDS=720000

# 0405：修改为Policy + Reward 固定组合（仅跑以下三对组合）
# 格式: "Policy目录名|Reward目录名"
POLICY_REWARD_PAIRS=(
    # "Qwen2.5-Math-1.5B-Instruct|math-shepherd-mistral-7b-prm"
    "Qwen2.5-Math-7B-Instruct|Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    # "Qwen2.5-Math-1.5B-Instruct|Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
)

# 数据集配置 (任务名, batch_size)
# batch_size: 一次性分配给评估的题目数量（降低以减少内存占用）
DATASETS=(
    "AIME24 5"  # 从 30 降到 15（减少50%） straggler出现在question19
    "AMC23 5"   # 从 40 降到 20（减少50%）
    # "MATH 500"
)

# (tree_max_width, num_seq) 仅遍历下列组合
BEAM_WIDTH_NUMSEQ_PAIRS=(
    "4 1"
    "8 1"
    "2 1"
    # "8 2"
    # "4 2"
)

# 评估随机种子（传入 run.sh --seed → evaluate.py）
# 同一 straggler 配置重复跑 EVAL_REPEAT_COUNT 次时，每次均使用 EVAL_SEED（不随 run 递增）
EVAL_SEED=43

# 0408：增加循环次数，循环三次取均值。
# 每个 (Policy, Reward, Task, Beam, Straggler) 配置连续评估次数；取平均时请合并各 _run* 目录下的 avg_result.json
# 为 1 时不改输出目录名（与改前一致）；大于 1 时在目录名末尾追加 _run1 … _runN
EVAL_REPEAT_COUNT=1

# output 子目录名第一段须与 run.sh 里 --tree_max_depth 及 evaluate.py 中 params 一致（tree_max_depth_width_num_seq）
EVAL_TREE_MAX_DEPTH=40

# ========== STRAGGLER参数扫描配置 ==========
# PRUNE 优先级最高：仅 STRAGGLER_PRUNE_VALUES=1 时启用剪枝；PRUNE=0 时 ratio/min/门控/deferred 均无意义（脚本固定传 0）
STRAGGLER_PRUNE_VALUES=(
    0    # 关闭
    # 1    # 开启
)

# 仅当 PRUNE=1 时遍历 ratio / min_tokens (ratio, min_tokens)
STRAGGLER_RATIO_MIN_PAIRS=(
    # "1.5 80"
    "1.5 100"
    # "1.2 80"
    # "1.2 100"
    # "2.0 80"
    # "2.0 100"
    # "1.5 150"
    # "1.5 200"
)

# PRUNE=1 时：STRAGGLER_OTHER_REWARD_GATE_VALUES 含 1 → 兄弟分支门控，扫 STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES，且 **deferred_prune 固定 0**（与 deferred 不同时开）
# PRUNE=1 时：STRAGGLER_OTHER_REWARD_GATE_VALUES 含 0 → og=0、thr=0，扫 STRAGGLER_DEFERRED_PRUNE_VALUES
STRAGGLER_OTHER_REWARD_GATE_VALUES=(
    0
    # 1
)
STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES=(
    0.9
)

# 跨 step 延迟剪枝（仅 PRUNE=1 且走 og=0 分支时参与扫描；PRUNE=0 时固定 0）
STRAGGLER_DEFERRED_PRUNE_VALUES=(
    0
    # 1
)

GATE_HAS_0=0
GATE_HAS_1=0
for _srg in "${STRAGGLER_OTHER_REWARD_GATE_VALUES[@]}"; do
    [ "${_srg}" -eq 0 ] && GATE_HAS_0=1
    [ "${_srg}" -eq 1 ] && GATE_HAS_1=1
done
PRUNE1_SUBCONFIGS=0
[ "$GATE_HAS_1" -eq 1 ] && PRUNE1_SUBCONFIGS=$((PRUNE1_SUBCONFIGS + ${#STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES[@]}))
[ "$GATE_HAS_0" -eq 1 ] && PRUNE1_SUBCONFIGS=$((PRUNE1_SUBCONFIGS + ${#STRAGGLER_DEFERRED_PRUNE_VALUES[@]}))
# ===========================================

# ========== MLP预测器配置 ==========
# 是否启用MLP预测器进行straggler检测（0=关闭，1=开启）
STRAGGLER_PREDICTOR_ENABLED=(
    # 0
    1
)

# 预测器权重文件路径（相对于repo根目录或绝对路径）
# 示例: "${REPO_ROOT}/3d-tts-sw/predictor/adaptive_threshold_weights.json"
STRAGGLER_PREDICTOR_WEIGHTS="${REPO_ROOT}/3d-tts-sw/predictor/adaptive_threshold_weights_low_fp.json"

# 预测器先验文件路径（相对于repo根目录或绝对路径）
# 示例: "${REPO_ROOT}/3d-tts-sw/predictor/adaptive_threshold_priors.json"
STRAGGLER_PREDICTOR_PRIORS="${REPO_ROOT}/3d-tts-sw/predictor/adaptive_threshold_priors.json"

# 活跃分支门控阈值（当活跃分支数 <= 此值时调用预测器）
ACTIVE_BRANCH_GATE=1

# Straggler 预算保护（BUDGET_ON=1 时启用：beam_search_step < STRAGGLER_BUDGET 时跳过剪枝）
STRAGGLER_BUDGET_ON=(
    0
    # 1   
)
# 0=关闭, 1=启用
STRAGGLER_BUDGET=2      # 前 N 步不剪枝（默认 2）
# ====================================

# 日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ========== 功耗监控函数 ==========
# 全局变量：功耗监控进程PID和临时文件
POWER_MONITOR_PID=""
POWER_TEMP_FILE=""

# 获取当前使用的GPU ID列表
get_gpu_ids() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '
    else
        # 默认使用GPU 0
        echo "0"
    fi
}

# 启动功耗监控（后台每秒采样nvidia-smi）
start_power_monitor() {
    local question_id=$1
    local config_name=$2
    
    if [ "${RECORD_POWER}" -ne 1 ]; then
        return
    fi
    
    # 创建临时文件存储功耗数据
    POWER_TEMP_FILE=$(mktemp "/tmp/power_monitor_${question_id}_XXXXXX.csv")
    
    # 获取GPU ID列表
    local gpu_ids
    gpu_ids=$(get_gpu_ids)
    local gpu_id_csv
    gpu_id_csv=$(echo "$gpu_ids" | tr ' ' ',')
    
    log_message "启动功耗监控: question=${question_id}, GPUs=${gpu_id_csv}, 输出文件=${POWER_TEMP_FILE}"
    
    # 后台启动nvidia-smi循环采样（每秒一次）
    (
        while true; do
            # 获取时间戳和功耗（格式: timestamp, gpu_id, power.draw [W]）
            nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,noheader,nounits -i "$gpu_id_csv" >> "$POWER_TEMP_FILE" 2>/dev/null
            sleep 1
        done
    ) &
    POWER_MONITOR_PID=$!
    
    log_message "功耗监控进程已启动: PID=${POWER_MONITOR_PID}"
}

# 停止功耗监控并计算平均功耗
stop_power_monitor() {
    local question_id=$1
    local config_power_dir=$2
    
    if [ "${RECORD_POWER}" -ne 1 ]; then
        echo "0"
        return
    fi
    
    if [ -z "$POWER_MONITOR_PID" ] || ! kill -0 "$POWER_MONITOR_PID" 2>/dev/null; then
        log_message "警告: 功耗监控进程不存在或已停止" >&2
        echo "0"
        return
    fi
    
    # 停止监控进程
    kill "$POWER_MONITOR_PID" 2>/dev/null || true
    wait "$POWER_MONITOR_PID" 2>/dev/null || true
    POWER_MONITOR_PID=""
    
    log_message "功耗监控进程已停止，开始计算平均功耗..." >&2
    
    if [ ! -f "$POWER_TEMP_FILE" ] || [ ! -s "$POWER_TEMP_FILE" ]; then
        log_message "警告: 功耗数据文件为空或不存在: ${POWER_TEMP_FILE}" >&2
        echo "0"
        return
    fi
    
    # 计算平均功耗（所有GPU的总功耗的时间平均值）
    # CSV格式: timestamp, gpu_index, power_draw
    local avg_power
    avg_power=$(awk -F', ' '
        BEGIN { total=0; count=0 }
        {
            # 第三列是功耗值
            power = $3
            if (power ~ /^[0-9.]+$/) {
                total += power
                count++
            }
        }
        END {
            if (count > 0) {
                printf "%.2f", total / count
            } else {
                print "0"
            }
        }
    ' "$POWER_TEMP_FILE")
    
    local sample_count
    sample_count=$(wc -l < "$POWER_TEMP_FILE")
    
    log_message "配置 ${question_id} 平均功耗: ${avg_power} W (采样点数: ${sample_count})" >&2
    
    # 保存原始采样数据到配置目录（CSV格式：timestamp, gpu_index, power_draw）
    if [ -n "$config_power_dir" ] && [ -d "$config_power_dir" ]; then
        local raw_data_file="${config_power_dir}/power_samples.csv"
        # 添加CSV头部（如果文件不存在）
        if [ ! -f "$raw_data_file" ]; then
            echo "timestamp,gpu_index,power_draw_w" > "$raw_data_file"
        fi
        # 追加采样数据（去除nvidia-smi输出中的空格）
        sed 's/, /,/g' "$POWER_TEMP_FILE" >> "$raw_data_file"
        log_message "原始采样数据已保存: ${raw_data_file}" >&2
    fi
    
    # 清理临时文件
    rm -f "$POWER_TEMP_FILE"
    POWER_TEMP_FILE=""
    
    echo "$avg_power"
}

# 计算配置的所有问题平均功耗并保存汇总
calculate_config_avg_power() {
    local config_power_dir=$1
    local config_name=$2
    
    if [ "${RECORD_POWER}" -ne 1 ]; then
        return
    fi
    
    if [ ! -d "$config_power_dir" ]; then
        log_message "警告: 配置功耗目录不存在: ${config_power_dir}"
        return
    fi
    
    # 统计所有问题的功耗
    local total_power=0
    local question_count=0
    local power_values=""
    
    for power_file in "${config_power_dir}"/question_*_power.json; do
        if [ -f "$power_file" ]; then
            local power
            power=$(grep -o '"avg_power_w": [0-9.]*' "$power_file" | awk -F': ' '{print $2}')
            if [ -n "$power" ] && [ "$power" != "0" ]; then
                total_power=$(echo "$total_power + $power" | bc)
                question_count=$((question_count + 1))
                if [ -n "$power_values" ]; then
                    power_values="${power_values}, ${power}"
                else
                    power_values="${power}"
                fi
            fi
        fi
    done
    
    if [ $question_count -gt 0 ]; then
        local avg_power
        avg_power=$(echo "scale=2; $total_power / $question_count" | bc)
        
        log_message "配置 ${config_name} 平均功耗: ${avg_power} W (问题数: ${question_count})"
        
        # 保存配置汇总
        local summary_file="${config_power_dir}/power_summary.json"
        echo "{
    \"config_name\": \"${config_name}\",
    \"avg_power_w\": ${avg_power},
    \"total_questions\": ${question_count},
    \"question_powers\": [${power_values}]
}" > "$summary_file"
        
        log_message "功耗汇总已保存: ${summary_file}"
    else
        log_message "警告: 配置 ${config_name} 没有有效的功耗数据"
    fi
}

# 创建配置的功耗输出目录
create_power_output_dir() {
    local config_name=$1
    
    if [ "${RECORD_POWER}" -ne 1 ]; then
        echo ""
        return
    fi
    
    # 确保功耗输出根目录存在
    mkdir -p "${POWER_OUTPUT_DIR}"
    
    # 创建配置专属目录
    local config_power_dir="${POWER_OUTPUT_DIR}/${config_name}"
    mkdir -p "$config_power_dir"
    
    echo "$config_power_dir"
}

# 保存配置的功耗汇总（整个评估过程的平均功耗）
save_config_power_summary() {
    local config_power_dir=$1
    local config_name=$2
    local avg_power=$3
    local task=$4
    local batch_size=$5
    
    if [ "${RECORD_POWER}" -ne 1 ]; then
        return
    fi
    
    if [ -z "$config_power_dir" ] || [ ! -d "$config_power_dir" ]; then
        log_message "警告: 功耗目录无效，无法保存汇总"
        return
    fi
    
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    local summary_file="${config_power_dir}/power_summary.json"
    echo "{
    \"config_name\": \"${config_name}\",
    \"task\": \"${task}\",
    \"batch_size\": ${batch_size},
    \"avg_power_w\": ${avg_power},
    \"timestamp\": \"${timestamp}\",
    \"gpu_ids\": \"${CUDA_VISIBLE_DEVICES:-0}\"
}" > "$summary_file"
    
    log_message "✓ 功耗汇总已保存: ${summary_file} (平均功耗: ${avg_power} W)"
}
# ================================

# 等待函数
wait_and_log() {
    local seconds=$1
    log_message "等待 ${seconds} 秒..."
    sleep $seconds
    log_message "等待完成"
}

# 健康检查函数 - 检查服务是否就绪
check_service_health() {
    local max_attempts=150  # 最多检查150次（约12.5分钟，reward model加载需~10分钟）真要这么久？？
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
    
    log_message "启动服务: Policy=${policy_path##*/}, Reward=${reward_path##*/}, GPUs=${N_GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    
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
    local predictor_enabled=${12:-0}
    local budget_on=${13:-0}
    local run_idx=${14:-1}

    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    local run_suffix
    run_suffix=$(eval_repeat_run_suffix "$run_idx")

    # 构建重命名后的目录路径
    local renamed_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}_${straggler_other_gate}_${straggler_other_thr}_def${straggler_deferred}_pred${predictor_enabled}_bud${budget_on}${run_suffix}"
    local result_file="${renamed_dir}/avg_result.json"

    # 检查重命名后的目录中的avg_result.json是否存在
    if [ -f "$result_file" ]; then
        log_message "✓ 跳过已完成的评估: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}_pred${predictor_enabled}_bud${budget_on}, run=${run_idx}/${EVAL_REPEAT_COUNT}"
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
    local predictor_enabled=${12:-0}
    local budget_on=${13:-0}
    local run_idx=${14:-1}

    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    local run_suffix
    run_suffix=$(eval_repeat_run_suffix "$run_idx")

    # 原始输出目录
    local output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}"
    
    # 重命名后的目录（添加straggler参数后缀，及可选的重复次序后缀；须与 check_evaluation_completed 中路径一致）
    local new_output_dir="${OUTPUT_BASE_DIR}/${task}_beam_search/${policy_model}/${reward_model}/${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}_${straggler_other_gate}_${straggler_other_thr}_def${straggler_deferred}_pred${predictor_enabled}_bud${budget_on}${run_suffix}"
    
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
    local straggler_predictor_enabled=${13:-0}
    local straggler_budget_on=${14:-0}
    local run_idx=${15:-1}
    local run_seed=$EVAL_SEED
    local max_retries=3
    local attempt=1

    # 构建配置名称（用于功耗记录目录）
    local policy_model="${policy_path##*/}"
    local reward_model="${reward_path##*/}"
    local run_suffix
    run_suffix=$(eval_repeat_run_suffix "$run_idx")
    local config_name="${task}_${policy_model}_${reward_model}_${EVAL_TREE_MAX_DEPTH}_${branch_width}_${num_seq}_straggler_${straggler_prune}_${straggler_ratio}_${straggler_min}_${straggler_other_gate}_${straggler_other_thr}_def${straggler_deferred}_pred${straggler_predictor_enabled}_bud${straggler_budget_on}${run_suffix}"
    
    # 创建功耗输出目录
    local config_power_dir=""
    if [ "${RECORD_POWER}" -eq 1 ]; then
        config_power_dir=$(create_power_output_dir "$config_name")
        log_message "功耗记录已启用，输出目录: ${config_power_dir}"
    fi

    while [ $attempt -le $max_retries ]; do
        log_message "运行评估 (重试 ${attempt}/${max_retries}, 配置重复 ${run_idx}/${EVAL_REPEAT_COUNT}, seed=${run_seed}): Policy=${policy_path##*/}, Reward=${reward_path##*/}, Task=${task}, BatchSize=${batch_size}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}_pred${straggler_predictor_enabled}_bud_on=${straggler_budget_on}_budget=${STRAGGLER_BUDGET}"
        
        # 启动功耗监控（整个配置的评估过程）
        start_power_monitor "config_${attempt}" "$config_name"
        
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
            --straggler_predictor_enabled "$straggler_predictor_enabled" \
            --straggler_predictor_weights "$STRAGGLER_PREDICTOR_WEIGHTS" \
            --straggler_predictor_priors "$STRAGGLER_PREDICTOR_PRIORS" \
            --active_branch_gate "$ACTIVE_BRANCH_GATE" \
            --straggler_budget_on "$straggler_budget_on" \
            --straggler_budget "$STRAGGLER_BUDGET" \
            --deterministic 0

        local exit_code=$?
        
        # 停止功耗监控并保存结果
        local avg_power
        avg_power=$(stop_power_monitor "config_${attempt}" "$config_power_dir")
        
        if [ $exit_code -eq 0 ]; then
            log_message "评估完成: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}_pred${straggler_predictor_enabled}_bud_on=${straggler_budget_on}_budget=${STRAGGLER_BUDGET}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 成功 (重试第 ${attempt} 次)"
            
            # 保存功耗汇总（成功时）
            if [ "${RECORD_POWER}" -eq 1 ] && [ -n "$config_power_dir" ]; then
                save_config_power_summary "$config_power_dir" "$config_name" "$avg_power" "$task" "$batch_size"
            fi

            # 评估成功后，重命名输出目录
            rename_output_dir "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$straggler_predictor_enabled" "$straggler_budget_on" "$run_idx"
            break
        else
            if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
                log_message "检测到评估卡死/超时: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}_pred${straggler_predictor_enabled}_bud_on=${straggler_budget_on}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 退出码: ${exit_code}"
            else
                log_message "评估失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}_pred${straggler_predictor_enabled}_bud_on=${straggler_budget_on}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 退出码: ${exit_code}"
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
                log_message "评估最终失败: ${task}, Branch=${branch_width}, NumSeq=${num_seq}, Straggler=${straggler_prune}_${straggler_ratio}_${straggler_min}_og${straggler_other_gate}_thr${straggler_other_thr}_def${straggler_deferred}_pred${straggler_predictor_enabled}_bud_on=${straggler_budget_on}, run=${run_idx}/${EVAL_REPEAT_COUNT} - 已达到最大重试次数 (${max_retries})"
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
    log_message "输出根目录 OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}（PRUNE=1 时结果子目录名含 _bud0 或 _bud1 等，与 STRAGGLER_BUDGET_ON 一致）"
    log_message "EVAL_SEED=${EVAL_SEED}，每配置重复次数 EVAL_REPEAT_COUNT=${EVAL_REPEAT_COUNT}（各次 run 均使用同一 seed）"
    log_message "Policy–Reward 对数量: ${#POLICY_REWARD_PAIRS[@]}"
    log_message "数据集数量: ${#DATASETS[@]}"
    log_message "Beam(宽×num_seq) 组合数: ${#BEAM_WIDTH_NUMSEQ_PAIRS[@]}"
    log_message "Straggler配置: Prune=0 (1) + Prune=1 (ratio×min×每档子配置=${PRUNE1_SUBCONFIGS}；og=1 仅扫 threshold 且 def=0；og=0 扫 deferred，与 og=1 互斥)"
    log_message "Straggler Budget: BUDGET_ON 扫描值=$(printf '%s ' "${STRAGGLER_BUDGET_ON[@]}"), BUDGET=${STRAGGLER_BUDGET}（前 ${STRAGGLER_BUDGET} 个 workload step 不剪枝）"
    if [ "${N_GPUS}" -eq 1 ]; then
        log_message "N_GPUS=1，CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}（serve_gpu1 与 scripts/run.sh 继承）"
    fi
    if [ "${RECORD_POWER}" -eq 1 ]; then
        log_message "功耗记录: 已启用，输出目录=${POWER_OUTPUT_DIR}"
    else
        log_message "功耗记录: 已关闭"
    fi
    
    # prune=0: 1；prune=1: ratio×min×([gate 含1] threshold 档 + [gate 含0] deferred 档)；og=1 与 def=1 从不同时
    local base_combinations=$((${#POLICY_REWARD_PAIRS[@]} * ${#DATASETS[@]} * ${#BEAM_WIDTH_NUMSEQ_PAIRS[@]}))
    local straggler_configs=$((1 + ${#STRAGGLER_RATIO_MIN_PAIRS[@]} * PRUNE1_SUBCONFIGS * ${#STRAGGLER_PREDICTOR_ENABLED[@]} * ${#STRAGGLER_BUDGET_ON[@]}))
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
                            log_message "Straggler Budget: PRUNE=0 时不扫描 bud_on（记录与目录名固定 bud_on=0, budget 参数无效）"
                            
                            if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "0" "0" "$repeat_idx"; then
                                continue
                            fi

                            run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "0" "0" "$repeat_idx"
                            done
                        else
                            # prune=1：兄弟分支门控(og=1) 与 deferred 互斥 — og=1 时 def=0；og=0 时扫 deferred
                            for ratio_min_pair in "${STRAGGLER_RATIO_MIN_PAIRS[@]}"; do
                                read -r straggler_ratio straggler_min <<< "$ratio_min_pair"
                                    if [ "$GATE_HAS_1" -eq 1 ]; then
                                        straggler_other_gate=1
                                        straggler_deferred=0
                                        for straggler_other_thr in "${STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES[@]}"; do
                                            for straggler_predictor_enabled in "${STRAGGLER_PREDICTOR_ENABLED[@]}"; do
                                            for straggler_budget_on in "${STRAGGLER_BUDGET_ON[@]}"; do
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
                                            log_message "Straggler 兄弟PRM门控: 开启 og=1 threshold=${straggler_other_thr}（deferred 关闭 def=0）"
                                            log_message "Straggler 预测器: pred=${straggler_predictor_enabled}"
                                            log_message "Straggler Budget: bud_on=${straggler_budget_on} budget=${STRAGGLER_BUDGET}"

                                            if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$straggler_predictor_enabled" "$straggler_budget_on" "$repeat_idx"; then
                                                continue
                                            fi

                                            run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$straggler_predictor_enabled" "$straggler_budget_on" "$repeat_idx"
                                            done
                                            done
                                            done
                                        done
                                    fi
                                    if [ "$GATE_HAS_0" -eq 1 ]; then
                                        straggler_other_gate=0
                                        straggler_other_thr=0
                                        for straggler_deferred in "${STRAGGLER_DEFERRED_PRUNE_VALUES[@]}"; do
                                            for straggler_predictor_enabled in "${STRAGGLER_PREDICTOR_ENABLED[@]}"; do
                                            for straggler_budget_on in "${STRAGGLER_BUDGET_ON[@]}"; do
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
                                            log_message "Straggler 预测器: pred=${straggler_predictor_enabled}"
                                            log_message "Straggler Budget: bud_on=${straggler_budget_on} budget=${STRAGGLER_BUDGET}"

                                            if check_evaluation_completed "$policy_path" "$reward_path" "$task" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$straggler_predictor_enabled" "$straggler_budget_on" "$repeat_idx"; then
                                                continue
                                            fi

                                            run_evaluation "$policy_path" "$reward_path" "$task" "$batch_size" "$branch_width" "$num_seq" "$straggler_prune" "$straggler_ratio" "$straggler_min" "$straggler_other_gate" "$straggler_other_thr" "$straggler_deferred" "$straggler_predictor_enabled" "$straggler_budget_on" "$repeat_idx"
                                            done
                                            done
                                            done
                                        done
                                    fi
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
    echo ""
    echo "=== Straggler MLP 预测器（与 run.sh 一致；每行 pred=0/1 对应是否启用）==="
    echo "  STRAGGLER_PREDICTOR_WEIGHTS=${STRAGGLER_PREDICTOR_WEIGHTS}"
    echo "  STRAGGLER_PREDICTOR_PRIORS=${STRAGGLER_PREDICTOR_PRIORS}"
    echo "  ACTIVE_BRANCH_GATE=${ACTIVE_BRANCH_GATE}"
    echo "  STRAGGLER_PREDICTOR_ENABLED 扫描值: $(printf '%s ' "${STRAGGLER_PREDICTOR_ENABLED[@]}")"
    echo "  说明: PRUNE=0 时 pred 固定为 0；PRUNE=1 时按上表扫描 pred。run.sh 始终传入上述 weights/priors/active_branch_gate。"
    if [ -f "${STRAGGLER_PREDICTOR_WEIGHTS}" ]; then
        echo "  （weights 文件存在）"
    else
        echo "  （警告: weights 文件不存在: ${STRAGGLER_PREDICTOR_WEIGHTS}）"
    fi
    if [ -f "${STRAGGLER_PREDICTOR_PRIORS}" ]; then
        echo "  （priors 文件存在）"
    else
        echo "  （警告: priors 文件不存在: ${STRAGGLER_PREDICTOR_PRIORS}）"
    fi
    echo "======================================================================"
    echo ""
    echo "=== Straggler Budget（前 N 步不剪枝）==="
    echo "  STRAGGLER_BUDGET_ON 扫描值: $(printf '%s ' "${STRAGGLER_BUDGET_ON[@]}")"
    echo "  STRAGGLER_BUDGET=${STRAGGLER_BUDGET}（bud_on=1 时前 ${STRAGGLER_BUDGET} 个 workload step 跳过剪枝）"
    echo "  说明: PRUNE=0 时 bud 固定为 0；PRUNE=1 时按上表扫描 bud_on。"
    echo "======================================================================"
    echo ""
    echo "=== 功耗记录配置 ==="
    echo "  RECORD_POWER=${RECORD_POWER}（0=关闭，1=开启）"
    echo "  POWER_OUTPUT_DIR=${POWER_OUTPUT_DIR}"
    if [ "${RECORD_POWER}" -eq 1 ]; then
        echo "  说明: 功耗记录已启用，每个配置的评估过程中每秒采样nvidia-smi功耗，结果保存到上述目录。"
    else
        echo "  说明: 功耗记录已关闭。如需启用，请将 RECORD_POWER 设为 1。"
    fi
    echo "======================================================================"
    echo ""
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
                            for ratio_min_pair in "${STRAGGLER_RATIO_MIN_PAIRS[@]}"; do
                                read -r straggler_ratio straggler_min <<< "$ratio_min_pair"
                                    if [ "$GATE_HAS_1" -eq 1 ]; then
                                        for straggler_other_thr in "${STRAGGLER_OTHER_REWARD_THRESHOLD_VALUES[@]}"; do
                                            for straggler_predictor_enabled in "${STRAGGLER_PREDICTOR_ENABLED[@]}"; do
                                            for straggler_budget_on in "${STRAGGLER_BUDGET_ON[@]}"; do
                                            for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                                            echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=1 RATIO=${straggler_ratio} MIN=${straggler_min} og=1 thr=${straggler_other_thr} def=0 pred=${straggler_predictor_enabled} bud=${straggler_budget_on}, run=${repeat_idx}/${EVAL_REPEAT_COUNT}, seed=${EVAL_SEED}"
                                            done
                                            done
                                            done
                                        done
                                    fi
                                    if [ "$GATE_HAS_0" -eq 1 ]; then
                                        for straggler_deferred in "${STRAGGLER_DEFERRED_PRUNE_VALUES[@]}"; do
                                            for straggler_predictor_enabled in "${STRAGGLER_PREDICTOR_ENABLED[@]}"; do
                                            for straggler_budget_on in "${STRAGGLER_BUDGET_ON[@]}"; do
                                            for ((repeat_idx=1; repeat_idx<=EVAL_REPEAT_COUNT; repeat_idx++)); do
                                            echo "Policy: ${policy_model}, Reward: ${reward_model}, Task: ${task}, BatchSize: ${batch_size}, Branch: ${branch_width}, NumSeq: ${num_seq}, Straggler: PRUNE=1 RATIO=${straggler_ratio} MIN=${straggler_min} og=0 thr=0 def=${straggler_deferred} pred=${straggler_predictor_enabled} bud=${straggler_budget_on}, run=${repeat_idx}/${EVAL_REPEAT_COUNT}, seed=${EVAL_SEED}"
                                            done
                                            done
                                            done
                                        done
                                    fi
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
