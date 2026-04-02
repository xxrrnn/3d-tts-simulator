#!/usr/bin/env python3
"""
最终简化工作负载生成器
保留必要信息：prefill的KV cache、decode的branch路径、每个branch的reward得分和源文件路径

输出格式:
{
  "question_id": "question_X",
  "prefill": {
    "kv_cache_count": int  # Prefill阶段的KV cache数量
  },
  "decode": {
    "steps": [
      {
        "step": int,  # 步骤编号
        "branch_count": int,  # branch数量
        "branch_tokens": [int, ...],  # 每个branch的token数量
        "branch_rewards": [float, ...],  # 每个branch的reward得分
        "selected_branch_index": int  # 选中的branch索引(-1表示未选中)
      },
      ...
      {
        "step": N,  # 最后一个步骤
        ...
        "source_jsonl": "path/to/record_0.jsonl"  # 源JSONL文件的绝对路径
      }
    ]
  }
}

usage:

cd /root/autodl-tmp/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload
# 为单个数据集生成
python gen_workload.py --input ../../src/output/AMC23_beam_search --verbose
# 或批量处理
for dataset in AMC23_beam_search AIME24_beam_search MATH_beam_search; do
    python gen_workload.py --input ../../src/output/$dataset --verbose
done
"""

import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_question_data(record_path: Path) -> Optional[Dict[str, Any]]:
    """加载单个问题的 JSONL 记录"""
    try:
        with open(record_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 尝试直接解析整个文件(如果是单个JSON对象)
            try:
                data = json.loads(content)
                if 'output' in data and len(data['output']) > 0:
                    return data
            except json.JSONDecodeError:
                # 如果不是单个JSON对象,尝试按行解析(JSONL格式)
                pass
            
            # 按行解析JSONL格式
            for line in content.split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if 'output' in data and len(data['output']) > 0:
                            return data
                    except json.JSONDecodeError:
                        continue
        return None
    except Exception as e:
        logger.error(f"Error loading {record_path}: {e}")
        return None

def extract_prefill_kv_cache(data: Dict[str, Any]) -> int:
    """提取 Prefill 阶段的 KV cache 数量"""
    kv_cache_count = 0
    
    if data.get('output') and len(data['output']) > 0:
        first_output = data['output'][0]
        detailed_log = first_output.get('detailed_beam_search_log', {})
        
        if detailed_log:
            step_details = detailed_log.get("step_details", [])
            if step_details and len(step_details) > 0:
                first_step = step_details[0]
                expansion_results = first_step.get("expansion_results", [])
                
                if expansion_results:
                    first_expansion = expansion_results[0]
                    initial_tokens = first_expansion.get("api_completion_tokens", 0)
                    question_tokens = len(data.get("question", "").split())
                    kv_cache_count = question_tokens + initial_tokens
    
    return kv_cache_count

def extract_decode_steps(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """提取 Decode 阶段的步骤信息"""
    steps = []
    
    if data.get('output') and len(data['output']) > 0:
        first_output = data['output'][0]
        detailed_log = first_output.get('detailed_beam_search_log', {})
        
        if detailed_log:
            step_details = detailed_log.get("step_details", [])
            
            for step_info in step_details:
                step = step_info.get("step", 0)
                selection = step_info.get("selection_process", {})
                selected_branches = selection.get("selected_branches", [])
                
                # 保留token数量和reward得分
                branch_tokens = []
                branch_rewards = []
                selected_index = -1
                
                for i, branch in enumerate(selected_branches):
                    branch_tokens.append(branch.get("num_tokens", 0))
                    branch_rewards.append(branch.get("reward_score", 0.0))
                    if branch.get("selected", False):
                        selected_index = i
                
                steps.append({
                    "step": step,
                    "branch_count": len(selected_branches),
                    "branch_tokens": branch_tokens,
                    "branch_rewards": branch_rewards,
                    "selected_branch_index": selected_index
                })
    
    return steps

def generate_workload_for_question(question_dir: Path, output_dir: Path) -> bool:
    """为单个问题生成最简化工作负载"""
    record_file = question_dir / "record_0.jsonl"
    if not record_file.exists():
        logger.warning(f"Record file not found: {record_file}")
        return False
    
    data = load_question_data(record_file)
    if not data:
        logger.error(f"Failed to load data from {record_file}")
        return False
    
    # 生成最简化的工作负载
    steps = extract_decode_steps(data)
    
    # 在最后一个step中添加jsonl文件路径
    if steps:
        steps[-1]["source_jsonl"] = str(record_file.absolute())
    
    workload = {
        "question_id": question_dir.name,
        "prefill": {
            "kv_cache_count": extract_prefill_kv_cache(data)
        },
        "decode": {
            "steps": steps
        }
    }
    
    output_file = output_dir / f"{question_dir.name}_workload.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(workload, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated workload: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving workload to {output_file}: {e}")
        return False

def generate_all_workloads(input_dir: Path, dataset_name: str) -> None:
    """生成所有问题的工作负载"""
    logger.info(f"Starting workload generation for dataset: {dataset_name}")
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    generated_count = 0
    total_questions = 0
    
    for policy_dir in input_dir.iterdir():
        if not policy_dir.is_dir():
            continue
            
        logger.info(f"Processing policy model: {policy_dir.name}")
        
        for reward_dir in policy_dir.iterdir():
            if not reward_dir.is_dir():
                continue
                
            logger.info(f"  Processing reward model: {reward_dir.name}")
            
            for config_or_question_dir in reward_dir.iterdir():
                if not config_or_question_dir.is_dir():
                    continue
                
                # 检查是配置目录(如16384_4_2)还是直接的question目录
                has_config_subdirs = any(
                    d.is_dir() and (d.name.startswith("question_") or d.name.startswith("16384_"))
                    for d in config_or_question_dir.iterdir()
                )
                
<<<<<<< HEAD
                # 创建输出目录
                output_base = Path("/root/autodl-tmp/3d-tts-simulator/3d-tts-sim/model_workloads") / dataset_name / policy_dir.name / reward_dir.name / config_dir.name
                output_base.mkdir(parents=True, exist_ok=True)
                
                # 处理所有问题
                for question_dir in config_dir.iterdir():
                    if not question_dir.is_dir() or not question_dir.name.startswith("question_"):
=======
                if has_config_subdirs and not config_or_question_dir.name.startswith("question_"):
                    # 这是配置目录，需要再深入一层
                    logger.info(f"    Processing config: {config_or_question_dir.name}")
                    
                    # 创建输出目录
                    output_base = Path("/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads") / dataset_name / policy_dir.name / reward_dir.name / config_or_question_dir.name
                    output_base.mkdir(parents=True, exist_ok=True)
                    
                    # 处理配置目录下的所有问题
                    for question_dir in config_or_question_dir.iterdir():
                        if not question_dir.is_dir() or not question_dir.name.startswith("question_"):
                            continue
                        
                        total_questions += 1
                        if generate_workload_for_question(question_dir, output_base):
                            generated_count += 1
                else:
                    # 这是直接的question目录（旧格式兼容）
                    if not config_or_question_dir.name.startswith("question_"):
>>>>>>> 207a1d0 (A6000 0401)
                        continue
                    
                    logger.info(f"    Processing direct question: {config_or_question_dir.name}")
                    
                    # 创建输出目录
                    output_base = Path("/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads") / dataset_name / policy_dir.name / reward_dir.name
                    output_base.mkdir(parents=True, exist_ok=True)
                    
                    total_questions += 1
                    if generate_workload_for_question(config_or_question_dir, output_base):
                        generated_count += 1
    
    logger.info(f"Generated {generated_count}/{total_questions} workloads")

def main():
    parser = argparse.ArgumentParser(description='最简化工作负载生成器')
    parser.add_argument('--input', required=True, help='输入目录路径')
    parser.add_argument('--dataset', help='数据集名称')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_path = Path(args.input)
    dataset_name = args.dataset or input_path.name
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Input: {input_path}")
    
    generate_all_workloads(input_path, dataset_name)

if __name__ == "__main__":
    main()