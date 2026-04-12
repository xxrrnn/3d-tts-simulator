#!/bin/bash

POLICY_MODEL_PATH=$1
VALUE_MODEL_PATH=$2
NUM_RM_WORKER=1
NUM_LM_WORKER=1

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "n_gpus: $n_gpus"

GPU_LIST=(0 0)
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

HOST_ADDR=$3
CONTROLLER_PORT=$4
WORKER_BASE_PORT=$5
CONDA_ENV_NAME="${CONDA_ENV:-tts}"
export PYTHONPATH=$(pwd)
# 勿用脚本外层的 which python：可能与 tmux 内 conda activate 后的解释器不一致，导致 worker 未真正启动。
# 各 tmux 窗口在 conda activate 后使用「python」命令。

# 0.0.0.0 仅用于 bind；向 controller 注册的 worker 地址必须是客户端/Ray 可访问的主机名或 IP（勿写 0.0.0.0）
REGISTER_HOST="${WORKER_ADVERTISE_HOST:-}"
if [ -z "${REGISTER_HOST}" ]; then
    if [ "${HOST_ADDR}" = "0.0.0.0" ]; then
        REGISTER_HOST="127.0.0.1"
    else
        REGISTER_HOST="${HOST_ADDR}"
    fi
fi
# workers 连本机 controller 时，URL 中 host 也不能是 0.0.0.0
CTRL_CONNECT_HOST="${HOST_ADDR}"
if [ "${CTRL_CONNECT_HOST}" = "0.0.0.0" ]; then
    CTRL_CONNECT_HOST="127.0.0.1"
fi
CONTROLLER_URL="http://${CTRL_CONNECT_HOST}:${CONTROLLER_PORT}"

LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR
session_name=tts
if tmux has-session -t $session_name 2>/dev/null; then
    echo "Session $session_name already exists. Killing it."
    tmux kill-session -t $session_name
fi
tmux start-server
tmux new-session -s $session_name -n controller -d
tmux send-keys "source ~/.bashrc && conda activate ${CONDA_ENV_NAME} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} " Enter
tmux send-keys "python -m fastchat.serve.controller --port ${CONTROLLER_PORT} --host $HOST_ADDR" Enter

echo "REGISTER_HOST=${REGISTER_HOST}（向 controller 注册的 worker 地址中的主机名）"
echo "CONTROLLER_URL=${CONTROLLER_URL}"
echo "Wait 20 seconds for controller to fully initialize..."
sleep 20

echo "Starting workers"
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n reward_$i
    tmux send-keys "source ~/.bashrc && conda activate ${CONDA_ENV_NAME} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} " Enter
    if [[ "$VALUE_MODEL_PATH" =~ "dummy" ]]; then
        command="pwd"
    else
        command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address ${CONTROLLER_URL} --host $HOST_ADDR --port $WORKER_PORT --worker-address http://${REGISTER_HOST}:$WORKER_PORT"
    fi
    tmux send-keys "$command" Enter
    echo "Reward worker $i started on GPU ${GPU_LIST[$i]} with port $WORKER_PORT, model: $VALUE_MODEL_PATH"
done

for i in $(seq $((NUM_RM_WORKER)) $((NUM_LM_WORKER+NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n policy_$i
    tmux send-keys "source ~/.bashrc && conda activate ${CONDA_ENV_NAME} && export LOGDIR=${LOGDIR} && export VLLM_ATTENTION_BACKEND=XFORMERS && cd ${PYTHONPATH} " Enter

    max_model_length=8192
    max_num_sequences=4
    enforce_eager=false
    cpu_offload_gb=0

    if [[ "$VALUE_MODEL_PATH" =~ "dummy" ]]; then
        gpu_memory_utilization=0.85 #增加到0.7，7bpolicy + 1.5b reward才能加载到gpu上
    else
        gpu_memory_utilization=0.85  # 0.8->0.65->0.50 降低以解决内存不足
    fi

    if [[ "$POLICY_MODEL_PATH" =~ "DeepSeek-R1" ]]; then
        max_model_length=32768
    fi

    command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python -m reason.llm_service.workers.vllm_worker --max_model_length ${max_model_length} --gpu_memory_utilization ${gpu_memory_utilization} --swap_space 16 --model-path $POLICY_MODEL_PATH --controller-address ${CONTROLLER_URL} --host $HOST_ADDR --port $WORKER_PORT --worker-address http://${REGISTER_HOST}:$WORKER_PORT"
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
    echo "Policy worker $i started on GPU ${GPU_LIST[$i]} with port $WORKER_PORT, model: $POLICY_MODEL_PATH"
done
