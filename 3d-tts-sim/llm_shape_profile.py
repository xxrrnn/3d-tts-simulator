import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import argparse, os
from typing import Optional
import pickle
from pathlib import Path

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 获取项目根目录和模型目录路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
models_dir = project_root /  "data" / "models"

# 可用的本地模型映射
LOCAL_MODELS = {
    # Policy models
    "qwen2.5-1.5b-math": str(models_dir / "Qwen2.5-Math-1.5B"),
    "qwen2.5-3b": str(models_dir / "Qwen2.5-3B"),
    "qwen2.5-math-1.5b": str(models_dir / "Qwen2.5-Math-1.5B-Instruct"),
    "qwen2.5-math-7b": str(models_dir / "Qwen2.5-Math-7B-Instruct"),
    
    # Reward models
    "skywork-prm-1.5b": str(models_dir / "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"),
    "math-shepherd-7b": str(models_dir / "math-shepherd-mistral-7b-prm"),
}

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--model", "-m", type=str, default="qwen2.5-1.5b-math", 
    help=f"Model name. Available local models: {list(LOCAL_MODELS.keys())}\n"
         f"Or use HuggingFace model path directly"
)
args = parser.parse_args()
model_str = args.model

# 如果是本地模型别名，转换为实际路径
if model_str in LOCAL_MODELS:
    model_path = LOCAL_MODELS[model_str]
    print(f"Using local model: {model_str} -> {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found: {model_path}")
    model_str = model_path
else:
    print(f"Using HuggingFace model: {model_str}")

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
model_config = AutoConfig.from_pretrained(model_str).to_dict()

layer_config = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        layer_config[n] = list(m.weight.shape)
        print(f'Module name:  {n}')
        print(f'Module shape: {m.weight.shape}')
        print()
print('\n\n')

model_name_dict = {
    # 本地模型路径映射
    str(models_dir / "Qwen2.5-Math-1.5B"): "qwen_2_5_1_5b_math_policy",
    str(models_dir / "Qwen2.5-3B"): "qwen_2_5_3b_policy", 
    str(models_dir / "Qwen2.5-Math-1.5B-Instruct"): "qwen_2_5_math_1_5b_policy",
    str(models_dir / "Qwen2.5-Math-7B-Instruct"): "qwen_2_5_math_7b_policy",
    str(models_dir / "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"): "skywork_prm_1_5b_reward",
    str(models_dir / "math-shepherd-mistral-7b-prm"): "math_shepherd_7b_reward",
    str(models_dir / "Skywork-Reward-V2-Llama-3.2-1B"): "skywork_reward_v2_llama_3_2_1b_reward",
}
# 根据模型名称确定保存路径
if model_str in model_name_dict:
    model_file_name = model_name_dict[model_str]
else:
    # 如果模型不在字典中，生成一个默认名称
    model_file_name = os.path.basename(model_str).replace("/", "_").replace("-", "_").lower()
    print(f"Warning: Model not in predefined dict, using generated name: {model_file_name}")

if model_file_name.endswith('_policy'):
    subfolder = 'policy'
elif model_file_name.endswith('_reward'):
    subfolder = 'reward'
else:
    subfolder = 'other'
    
file_path = f'./model_shape_config/{subfolder}/{model_file_name}.pickle'
os.makedirs(os.path.dirname(file_path), exist_ok=True)

print(f"Saving model configuration to: {file_path}")
with open(file_path, 'wb') as f:
    pickle.dump((model_config, layer_config), f)

print(f"Model shape profile saved successfully!")
print(f"Model: {model_str}")
print(f"Config file: {model_file_name}")
print(f"Category: {subfolder}")
