#!/bin/bash

conda activate tts

# Default arguments
LM=models--meta-llama--Llama-3.2-1B-Instruct
RM=models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B
task_name=MATH
method=beam_search
temperature=0.7
max_new_tokens=2048
tree_max_depth=40
tree_max_width=4
num_sequence=1
question_parallel_num=0
batch_size=500
max_time=3
n_gpus=1
double_line_break=1
local=0

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
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done
echo "LM: $LM, RM: $RM, task: $task_name, tree_max_width: $tree_max_width, num_sequence: $num_sequence, question_parallel_num: $question_parallel_num"
echo "batch_size: $batch_size, max_time: $max_time, n_gpus: $n_gpus, double_line_break: $double_line_break"

if [ $method == "beam_search" ]; then
    temperature=0.7
    max_new_tokens=2048
    tree_max_depth=40
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

export PYTHONPATH=$(pwd)
cd ${PYTHONPATH}

export CUDA_VISIBLE_DEVICES=0
GPU_LIST=(0 0)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, n_gpus: $n_gpus"
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

num_worker=12
save_dir=${PYTHONPATH}/output
LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR
controller_addr=http://$HOST_ADDR:$CONTROLLER_PORT

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
    --local $local
