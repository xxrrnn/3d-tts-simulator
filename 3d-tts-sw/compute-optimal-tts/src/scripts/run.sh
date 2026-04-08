#!/bin/bash

# <<<<<<< HEAD
# 使用 uv 激活虚拟环境，如果没有则跳过
# if command -v uv &> /dev/null; then
#     echo "Using uv environment..."
# elif command -v conda &> /dev/null; then
#     echo "Using conda environment..."
#     # 正确初始化 conda
#     source /root/miniconda3/etc/profile.d/conda.sh
#     conda activate tts
# else
#     echo "Neither uv nor conda found, using system python..."
# fi
# =======
# >>>>>>> 207a1d0 (A6000 0401)

# Default arguments
LM=models--meta-llama--Llama-3.2-1B-Instruct
RM=models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B
task_name=MATH
method=beam_search
temperature=0.7
max_new_tokens=204800
tree_max_depth=40
tree_max_width=8
num_sequence=1
question_parallel_num=0
batch_size=500
max_time=3
n_gpus=1
double_line_break=1
local=0
### beam search start
beam_search_detailed_log=0  # 新增：控制是否输出详细beam search日志 (0=关闭, 1=开启)
logprobs_topk=20  # 新增：控制记录 top-k logits 的 k 值（vLLM最大支持20）
### beam search end

seed=42 #0405 新增seed 可以调整
# straggler（与 SearchTree / record jsonl 中 straggler_log 一致）
straggler_prune=0 #是否开启straggler剪枝，1=剪枝
straggler_length_ratio=1.5 #straggler判定的长度倍率阈值
straggler_min_tokens=80 #straggler最短token阈值
straggler_prune_other_reward_gate=0 #1=仅当兄弟分支PRM最大分>阈值时才剪straggler
straggler_prune_other_reward_threshold=0.0 #与 gate=1 配合：须严格大于该值

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --LM)
        LM="$2"
        shift 2
        ;;
    --RM)
        RM="$2"
        shift 2
        ;;
    --task)
        task_name="$2"
        shift 2
        ;;
    --method)
        method="$2"
        shift 2
        ;;
    --temperature)
        temperature="$2"
        shift 2
        ;;
    --max_new_tokens)
        max_new_tokens="$2"
        shift 2
        ;;
    --tree_max_depth)
        tree_max_depth="$2"
        shift 2
        ;;
    --width)
        tree_max_width="$2"
        shift 2
        ;;
    --num_seq)
        num_sequence="$2"
        shift 2
        ;;
    --num_q)
        question_parallel_num="$2"
        shift 2
        ;;
    --bs)
        batch_size="$2"
        shift 2
        ;;
    --mt)
        max_time="$2"
        shift 2
        ;;
    --n_gpus)
        n_gpus="$2"
        shift 2
        ;;
    --double_line_break)
        double_line_break="$2"
        shift 2
        ;;
    --local)
        local="$2"
        shift 2
        ;;
    ## seed
    --seed)
        seed="$2"
        shift 2
        ;;
    --straggler_prune)
        straggler_prune="$2"
        shift 2
        ;;
    --straggler_length_ratio)
        straggler_length_ratio="$2"
        shift 2
        ;;
    --straggler_min_tokens)
        straggler_min_tokens="$2"
        shift 2
        ;;
    --straggler_prune_other_reward_gate)
        straggler_prune_other_reward_gate="$2"
        shift 2
        ;;
    --straggler_prune_other_reward_threshold)
        straggler_prune_other_reward_threshold="$2"
        shift 2
        ;;
    ### beam search start
    --beam-log)
        beam_search_detailed_log="$2"
        shift 2
        ;;
    --logprobs-topk)
        logprobs_topk="$2"
        shift 2
        ;;
    ### beam search end
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done
echo "LM: $LM, RM: $RM, task: $task_name, tree_max_width: $tree_max_width, num_sequence: $num_sequence, question_parallel_num: $question_parallel_num"
echo "batch_size: $batch_size, max_time: $max_time, n_gpus: $n_gpus, double_line_break: $double_line_break, seed:$seed"
echo "straggler_prune: $straggler_prune, straggler_length_ratio: $straggler_length_ratio, straggler_min_tokens: $straggler_min_tokens, straggler_prune_other_reward_gate: $straggler_prune_other_reward_gate, straggler_prune_other_reward_threshold: $straggler_prune_other_reward_threshold"
### beam search start
echo "beam_search_detailed_log: $beam_search_detailed_log, logprobs_topk: $logprobs_topk"
### beam search end

if [ $method == "beam_search" ]; then
    # 只有当max_new_tokens未设置或为默认值时才设置
    if [ $max_new_tokens -eq 204800 ]; then
        max_new_tokens=2048
    fi
    # 只有当tree_max_depth未设置或为默认值时才设置
    if [ $tree_max_depth -eq 40 ]; then
        tree_max_depth=40
    fi
elif [ $method == "best_of_n" ]; then
    temperature=0.7
    max_new_tokens=8192
    tree_max_depth=1
elif [ $method == "cot" ]; then
    temperature=0.0
    max_new_tokens=8192
    tree_max_depth=1
else
    echo "Invalid method: $method"
    exit
fi
if [[ "$LM" =~ "DeepSeek-R1" ]]; then
    temperature=0.6
    max_new_tokens=32768
fi
POLICY_MODEL_PATH=${LM}
VALUE_MODEL_PATH=${RM}

### beam search start
export BEAM_SEARCH_DETAILED_LOG=$beam_search_detailed_log
export LOGPROBS_TOPK=$logprobs_topk
### beam search end

export PYTHONPATH=$(pwd)
cd ${PYTHONPATH}

# 根据 n_gpus 动态设置 CUDA_VISIBLE_DEVICES 和 GPU_LIST
if [ $n_gpus -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
    GPU_LIST=(0 0)
elif [ $n_gpus -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    GPU_LIST=(0 1)
elif [ $n_gpus -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2
    GPU_LIST=(0 1 2)
elif [ $n_gpus -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    GPU_LIST=(0 1 2 3)
else
    echo "Error: n_gpus must be 1, 2, 3, or 4"
    exit 1
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, n_gpus: $n_gpus"
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

num_worker=6  #=12 改小一点？
# 并行多实例评估时可 export EVAL_SAVE_DIR 指向独立目录（与 eval_all_combinations_straggler_gpu*.sh 一致）
save_dir="${EVAL_SAVE_DIR:-${PYTHONPATH}/output}"
LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR
# 客户端访问 controller 时不能用 0.0.0.0 作为目标地址，否则常见为 Connection refused
_controller_host="${HOST_ADDR}"
if [ "${_controller_host}" = "0.0.0.0" ] || [ -z "${_controller_host}" ]; then
    _controller_host=127.0.0.1
fi
controller_addr="http://${_controller_host}:${CONTROLLER_PORT}"

echo "Running $method evaluation ..."

python reason/evaluation/evaluate.py \
    --LM $POLICY_MODEL_PATH \
    --RM $VALUE_MODEL_PATH \
    --task_name $task_name \
    --temperature $temperature \
    --max_new_tokens $max_new_tokens \
    --num_sequence $num_sequence \
    --tree_max_width $tree_max_width \
    --tree_max_depth $tree_max_depth \
    --save_dir $save_dir \
    --method $method \
    --num_worker $num_worker \
    --controller_addr $controller_addr \
    --add_step_prompt \
    --question_parallel_num $question_parallel_num \
    --double_line_break $double_line_break \
    --batch_size $batch_size \
    --max_time $max_time \
    --local $local \
    --seed $seed \
    --straggler_prune $straggler_prune \
    --straggler_length_ratio $straggler_length_ratio \
    --straggler_min_tokens $straggler_min_tokens \
    --straggler_prune_other_reward_gate $straggler_prune_other_reward_gate \
    --straggler_prune_other_reward_threshold $straggler_prune_other_reward_threshold
