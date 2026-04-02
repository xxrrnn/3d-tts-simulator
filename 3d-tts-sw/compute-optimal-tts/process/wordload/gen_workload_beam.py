#!/usr/bin/env python3
"""
Beam Search 工作负载生成器
从detailed_beam_search_log中提取beam search的workload信息
注意：这是workload格式，只包含统计信息，不包含完整文本

输出格式:
{
  "question_id": "question_X",
  "prefill": {
    "kv_cache_count": int  # Prefill阶段的KV cache数量
  },
  "beam_search_config": {
    "beam_size": int,
    "max_step": int
  },
  "decode": {
    "steps": [
      {
        "step": int,  # 步骤编号
        "branch_count": int,  # 候选branch数量
        "branch_tokens": [int, ...],  # 每个branch的token数量
        "branch_rewards": [float, ...],  # 每个branch的reward得分
        "selected_branch_index": int,  # 选中的branch索引(-1表示未选中)
        "terminated_count": int  # 终止的分支数
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
cd /DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload
python gen_workload_beam.py --input ../../src/output/AIME24_beam_search --verbose
"""

import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_question_data(record_path: Path) -> Optional[Dict[str, Any]]:
    """加载单个问题的 JSONL 记录"""
    try:
        with open(record_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            try:
                data = json.loads(content)
                if 'output' in data and len(data['output']) > 0:
                    return data
            except json.JSONDecodeError:
                pass
            
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


def extract_beam_search_workload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """提取beam search的workload信息（只保留统计信息）"""
    
    if not data.get('output') or len(data['output']) == 0:
        return {}, False
    
    # 查找包含detailed_beam_search_log的output条目
    detailed_log = None
    for output in data['output']:
        if output.get('detailed_beam_search_log'):
            detailed_log = output['detailed_beam_search_log']
            break
    
    if not detailed_log:
        first_output = data['output'][0]
        detailed_log = first_output.get('detailed_beam_search_log', {})
    
    if not detailed_log:
        logger.warning("No detailed_beam_search_log found")
        return {}, False
    
    # 提取beam search配置
    beam_config = {
        "beam_size": detailed_log.get("beam_size", 0),
        "max_step": detailed_log.get("max_step", 0)
    }
    
    # 提取每一步的workload信息
    decode_steps = []
    step_details = detailed_log.get("step_details", [])
    
    # 用于追踪上一步selected分支的reward到索引的映射
    prev_step_reward_to_branch_idx = {}
    
    for step_info in step_details:
        step_num = step_info.get("step", 0)
        selection_process = step_info.get("selection_process", {})
        selected_branches = selection_process.get("selected_branches", [])
        
        # 只保留统计信息，不保留文本内容
        branch_tokens = []
        branch_rewards = []
        branch_parent_indices = []  # 新增：记录每个分支的父分支索引
        selected_indices = []  # 改为列表，记录所有被选中的分支
        
        for i, branch in enumerate(selected_branches):
            branch_tokens.append(branch.get("num_tokens", 0))
            reward = branch.get("reward_score", 0.0)
            branch_rewards.append(reward)
            
            # 通过parent_value关联上一步的分支索引
            parent_value = branch.get("parent_value")
            parent_index = -1  # -1表示根节点或无法追踪
            
            if parent_value is not None and step_num > 0 and prev_step_reward_to_branch_idx:
                # 在上一步的reward映射中查找（浮点数比较）
                for prev_reward, prev_idx in prev_step_reward_to_branch_idx.items():
                    if abs(prev_reward - parent_value) < 1e-9:
                        parent_index = prev_idx
                        break
            
            branch_parent_indices.append(parent_index)
            
            if branch.get("selected", False):
                selected_indices.append(i)
        
        # 更新prev_step_reward_to_branch_idx供下一步使用
        # 只保存当前步的所有分支（因为下一步的parent可能来自任何一个）
        current_step_reward_to_idx = {}
        for i, reward in enumerate(branch_rewards):
            current_step_reward_to_idx[reward] = i
        prev_step_reward_to_branch_idx = current_step_reward_to_idx
        
        step_data = {
            "step": step_num,
            "branch_count": len(selected_branches),
            "branch_tokens": branch_tokens,
            "branch_rewards": branch_rewards,
            "branch_parent_indices": branch_parent_indices,  # 新增字段
            "selected_branch_indices": selected_indices,  # 改为复数形式
            "terminated_count": selection_process.get("terminated_count", 0)
        }
        decode_steps.append(step_data)
    
    result = {
        "beam_config": beam_config,
        "decode_steps": decode_steps
    }
    
    return result, True


def extract_prefill_kv_cache(data: Dict[str, Any]) -> int:
    """提取Prefill阶段的KV cache数量"""
    kv_cache_count = 0
    question_tokens = len(data.get("question", "").split())
    kv_cache_count = question_tokens
    
    if data.get('output') and len(data['output']) > 0:
        # 查找包含detailed_beam_search_log的output条目
        detailed_log = None
        for output in data['output']:
            if output.get('detailed_beam_search_log'):
                detailed_log = output['detailed_beam_search_log']
                break
        
        if not detailed_log:
            first_output = data['output'][0]
            detailed_log = first_output.get('detailed_beam_search_log', {})
        
        if detailed_log:
            step_details = detailed_log.get("step_details", [])
            if step_details and len(step_details) > 0:
                first_step = step_details[0]
                expansion_results = first_step.get("expansion_results", [])
                
                if expansion_results and len(expansion_results) > 0:
                    first_expansion = expansion_results[0]
                    api_tokens = first_expansion.get("api_completion_tokens", 0)
                    if api_tokens > 0:
                        kv_cache_count += api_tokens
    
    return kv_cache_count


def generate_workload_for_question(question_dir: Path, output_dir: Path) -> bool:
    """为单个问题生成beam search workload"""
    record_file = question_dir / "record_0.jsonl"
    if not record_file.exists():
        logger.warning(f"Record file not found: {record_file}")
        return False
    
    data = load_question_data(record_file)
    if not data:
        logger.error(f"Failed to load data from {record_file}")
        return False
    
    # 提取beam search workload
    beam_result, has_detailed_log = extract_beam_search_workload(data)
    
    if not has_detailed_log:
        logger.warning(f"No detailed beam search log in {record_file}, skipping")
        return False
    
    # 提取prefill信息
    kv_cache_count = extract_prefill_kv_cache(data)
    
    # 生成workload
    decode_steps = beam_result.get("decode_steps", [])
    
    # 在最后一个step中添加jsonl文件路径
    if decode_steps:
        decode_steps[-1]["source_jsonl"] = str(record_file.absolute())
    
    workload = {
        "question_id": question_dir.name,
        "prefill": {
            "kv_cache_count": kv_cache_count
        },
        "beam_search_config": beam_result.get("beam_config", {}),
        "decode_steps": decode_steps  # 修正：直接使用decode_steps
    }
    
    output_file = output_dir / f"workload_0.json"  # 修正文件名
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(workload, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated beam workload: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving workload to {output_file}: {e}")
        return False


def generate_all_workloads(input_dir: Path, dataset_name: str) -> None:
    """生成所有问题的beam search workload"""
    logger.info(f"Starting beam search workload generation for dataset: {dataset_name}")
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    generated_count = 0
    skipped_count = 0
    total_questions = 0
    
    # 判断input_dir是否直接包含question_目录
    question_dirs = list(input_dir.glob("question_*"))
    
    if question_dirs:
        # 直接在当前目录处理
        logger.info(f"Processing directory: {input_dir.name}")
        output_base = Path(__file__).parent / "output_beam" / dataset_name / input_dir.name
        output_base.mkdir(parents=True, exist_ok=True)
        
        for question_dir in question_dirs:
            if question_dir.is_dir():
                total_questions += 1
                question_output = output_base / question_dir.name
                question_output.mkdir(parents=True, exist_ok=True)
                if generate_workload_for_question(question_dir, question_output):
                    generated_count += 1
                else:
                    skipped_count += 1
    else:
        # 遍历多层目录结构
        for policy_dir in input_dir.iterdir():
            if not policy_dir.is_dir():
                continue
                
            logger.info(f"Processing policy model: {policy_dir.name}")
            
            for reward_dir in policy_dir.iterdir():
                if not reward_dir.is_dir():
                    continue
                    
                logger.info(f"  Processing reward model: {reward_dir.name}")
                
                for config_dir in reward_dir.iterdir():
                    if not config_dir.is_dir():
                        continue
                    
                    logger.info(f"    Processing config: {config_dir.name}")
                    
                    # 创建输出目录
                    output_base = Path(__file__).parent / "output_beam" / dataset_name / policy_dir.name / reward_dir.name / config_dir.name
                    output_base.mkdir(parents=True, exist_ok=True)
                    
                    # 处理所有问题
                    for question_dir in config_dir.iterdir():
                        if not question_dir.is_dir() or not question_dir.name.startswith("question_"):
                            continue
                        
                        total_questions += 1
                        question_output = output_base / question_dir.name
                        question_output.mkdir(parents=True, exist_ok=True)
                        if generate_workload_for_question(question_dir, question_output):
                            generated_count += 1
                        else:
                            skipped_count += 1
    
    logger.info(f"Generated {generated_count}/{total_questions} beam workloads, skipped {skipped_count}")


def main():
    parser = argparse.ArgumentParser(description='Beam Search工作负载生成器（简化格式）')
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
