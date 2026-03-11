#!/bin/bash

conda activate tts

POLICY_MODEL_PATH=$1
VALUE_MODEL_PATH=$2
NUM_RM_WORKER=1
NUM_LM_WORKER=1

export CUDA_VISIBLE_DEVICES=0,1,2
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "n_gpus: $n_gpus"

GPU_LIST=(0 1 2)
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

HOST_ADDR=$3
CONTROLLER_PORT=$4
WORKER_BASE_PORT=$5
export PYTHONPATH=$(pwd)
PYTHON_EXECUTABLE=$(which python)

LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR
session_name=tts
if tmux has-session -t $session_name 2>/dev/null; then
    echo "Session $session_name already exists. Killing it."
    tmux kill-session -t $session_name
fi
tmux start-server
tmux new-session -s $session_name -n controller -d
tmux send-keys "source ~/.bashrc && conda activate tts && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} " Enter
tmux send-keys "${PYTHON_EXECUTABLE} -m fastchat.serve.controller --port ${CONTROLLER_PORT} --host $HOST_ADDR" Enter

echo "Wait 5 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n reward_$i
    tmux send-keys "source ~/.bashrc && conda activate tts && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} " Enter
    if [[ "$VALUE_MODEL_PATH" =~ "dummy" ]]; then
        command="pwd"
    else
        command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]},${GPU_LIST[$i+1]} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.reward_model_worker --num-gpus 2 --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT"
    fi
    tmux send-keys "$command" Enter
    echo "Reward worker $i started on GPU ${GPU_LIST[$i]},${GPU_LIST[$i+1]} with port $WORKER_PORT, model: $VALUE_MODEL_PATH"
done

for i in $(seq $((NUM_RM_WORKER)) $((NUM_LM_WORKER+NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n policy_$i
    tmux send-keys "source ~/.bashrc && conda activate tts && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} " Enter

    max_model_length=8192
    max_num_sequences=0
    enforce_eager=false
    cpu_offload_gb=0

    gpu_memory_utilization=0.95
    tensor_parallel_size=1

    if [[ "$POLICY_MODEL_PATH" =~ "DeepSeek-R1" ]]; then
        max_model_length=32768
    fi

    command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i-$NUM_RM_WORKER+2]} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker --max_model_length ${max_model_length} --num-gpus ${tensor_parallel_size} --gpu_memory_utilization ${gpu_memory_utilization} --swap_space 16 --model-path $POLICY_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT"
    if [[ $max_num_sequences -gt 0 ]]; then
        command="$command --max_num_sequences $max_num_sequences"
    fi
    if [[ $enforce_eager == true ]]; then
        command="$command --enforce-eager"
    fi
    if [[ $cpu_offload_gb -gt 0 ]]; then
        command="$command --cpu-offload-gb $cpu_offload_gb"
    fi
    if [[ "$POLICY_MODEL_PATH" =~ "Qwen2.5-Math-1.5B" ]] || [[ "$POLICY_MODEL_PATH" =~ "Qwen2.5-Math-7B" ]]; then
        command="VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 $command"
    fi

    tmux send-keys "$command" Enter
    echo "Policy worker $i started on GPU ${GPU_LIST[$i-$NUM_RM_WORKER+2]} with port $WORKER_PORT, model: $POLICY_MODEL_PATH"
done
