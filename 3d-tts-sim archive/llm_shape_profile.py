import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import argparse, os
from typing import Optional
import pickle

# 禁用torch编译和triton相关功能，避免在CPU模式下出错
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
     "--model", "-m", type=str, required=True, 
     help="Model path or HuggingFace model name (e.g., './data/models/Qwen2.5-Math-1.5B-Instruct' or 'meta-llama/Llama-3.1-8B-Instruct')"
)
parser.add_argument(
     "--output-dir", "-o", type=str, default="./model_shape_config",
     help="Output directory for pickle files (default: ./model_shape_config)"
)
parser.add_argument(
     "--cpu-only", action="store_true",
     help="Load model on CPU only (no GPU)"
)
args = parser.parse_args()
model_str = args.model

torch.set_grad_enabled(False)

print(f"Loading model from: {model_str}")
if args.cpu_only:
    print("Using CPU only mode")
    model = AutoModelForCausalLM.from_pretrained(
        model_str, 
        torch_dtype=torch.float32,  # CPU通常使用float32
        low_cpu_mem_usage=True, 
        device_map='cpu'
    )
else:
    print("Using GPU mode")
    model = AutoModelForCausalLM.from_pretrained(
        model_str, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map='auto'
    )
model_config = AutoConfig.from_pretrained(model_str).to_dict()

layer_config = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        layer_config[n] = list(m.weight.shape)
        print(f'Module name:  {n}')
        print(f'Module shape: {m.weight.shape}')
        print()
print('\n\n')

# 计算参数统计
linear_params = sum(shape[0] * shape[1] for shape in layer_config.values())
total_params = sum(p.numel() for p in model.parameters())
param_stats = {
    'total_params': total_params,
    'linear_params': linear_params,
    'non_linear_params': total_params - linear_params
}

print(f'=== 参数统计 ===')
print(f'总参数量: {total_params:,} ({total_params/1e9:.2f}B)')
print(f'Linear层参数: {linear_params:,} ({linear_params/1e9:.2f}B)')
print()

model_basename = os.path.basename(model_str.rstrip('/'))
safe_name = model_basename.replace('/', '_').replace('.', '_')
file_path = os.path.join(args.output_dir, f'{safe_name}.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump((model_config, layer_config, param_stats), f)

print(f"Profile saved to: {file_path}")
print(f"\n使用pickle文件的示例代码:")
print(f"""
import pickle

# 读取pickle文件
with open('{file_path}', 'rb') as f:
    model_config, layer_config, param_stats = pickle.load(f)

# model_config: 包含模型的配置信息 (dict)
print("Model config keys:", model_config.keys())
print("Hidden size:", model_config.get('hidden_size'))
print("Num layers:", model_config.get('num_hidden_layers'))

# layer_config: 包含每个Linear层的shape信息 (dict)
print("\\nLayer shapes:")
for layer_name, shape in list(layer_config.items())[:5]:
    print(f"  {{layer_name}}: {{shape}}")

# param_stats: 参数统计信息 (dict)
print("\\n参数统计:")
print(f"  总参数量: {{param_stats['total_params']:,}} ({{param_stats['total_params']/1e9:.2f}}B)")
print(f"  Linear参数: {{param_stats['linear_params']:,}} ({{param_stats['linear_params']/1e9:.2f}}B)")
print(f"  Linear占比: {{param_stats['linear_ratio']*100:.2f}}%")
""")
