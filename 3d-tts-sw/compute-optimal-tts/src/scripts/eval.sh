#!/bin/bash
export VALUE_MODEL_PATH=/root/autodl-tmp/3d-tts-simulator/data/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_PATH=/root/autodl-tmp/3d-tts-simulator/data/models/Qwen2.5-Math-1.5B-Instruct
export LOGDIR=/root/autodl-tmp/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/logs
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081

bash scripts/serve_gpu1.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

echo "等待30秒..."
sleep 30
echo "等待完成"

bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 8 --num_seq 1 --method beam_search --temperature 0.7 --max_new_tokens 4096 --tree_max_depth 40  --num_seq 1  --bs 500 --mt 3 --n_gpus 1 --double_line_break 1 --local 0 --beam-log 1 --task AMC23

echo "等待30秒..."
sleep 30
echo "等待完成"

bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 8 --num_seq 1 --method beam_search --temperature 0.7 --max_new_tokens 4096 --tree_max_depth 40  --num_seq 1  --bs 40 --mt 3 --n_gpus 1 --double_line_break 1 --local 0 --beam-log 1 --task MATH

echo "等待30秒..."
sleep 30
echo "等待完成"

bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 8 --num_seq 1 --method beam_search --temperature 0.7 --max_new_tokens 4096 --tree_max_depth 40  --num_seq 1  --bs 30 --mt 3 --n_gpus 1 --double_line_break 1 --local 0 --beam-log 1 --task AIME24