import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import argparse, os
from typing import Optional
import pickle

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
     "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
args = parser.parse_args()
model_str = args.model

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(model_str, dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
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
    # Policy models
    "meta-llama/Llama-3.2-1B": "llama_3_2_1b_policy",
    "Qwen/Qwen2.5-1.5B": "qwen_2_5_1_5b_policy", 
    "Qwen/Qwen2.5-3B": "qwen_2_5_3b_policy",
    
    # Reward models  
    "Qwen/Qwen2.5-1.5B-reward": "qwen_2_5_1_5b_reward",
    "Qwen/Qwen2.5-7B": "qwen_2_5_7b_reward",
    "meta-llama/Llama-3.1-8B": "llama_3_1_8b_reward",
}
# 根据模型名称确定保存路径
model_file_name = model_name_dict[model_str]
if model_file_name.endswith('_policy'):
    subfolder = 'policy'
elif model_file_name.endswith('_reward'):
    subfolder = 'reward'
else:
    subfolder = 'other'
    
file_path = f'./model_shape_config/{subfolder}/{model_file_name}.pickle'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'wb') as f:
    pickle.dump((model_config, layer_config), f)
