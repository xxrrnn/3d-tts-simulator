#!/usr/bin/env python3
"""
<<<<<<< HEAD
识别符合straggler定义的branch并计算注意力熵
=======
识别符合straggler定义的branch
>>>>>>> 207a1d0 (A6000 0401)

Straggler定义:
1. 超过80 token
2. 超过除它之外其他branch最大值的2倍
3. 只统计branch数量不为1的情况

<<<<<<< HEAD
输出完整的branch情况汇总，包括原始位置信息和注意力熵分析
=======
输出完整的branch情况汇总，包括原始位置信息
>>>>>>> 207a1d0 (A6000 0401)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 207a1d0 (A6000 0401)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_straggler(branch_tokens: List[int], branch_index: int) -> bool:
    """判断某个branch是否为straggler
    
    Args:
        branch_tokens: 所有branch的token数量列表
        branch_index: 要判断的branch索引
    
    Returns:
        是否为straggler
    """
    # 条件3: branch数量至少为2
    if len(branch_tokens) <= 1:
        return False
    
    target_tokens = branch_tokens[branch_index]
    
    # 条件1: 超过80 token
    if target_tokens <= 100:
        return False
    
    # 条件2: 超过除它之外其他branch最大值的2倍
    other_tokens = [t for i, t in enumerate(branch_tokens) if i != branch_index]
    if not other_tokens:
        return False
    
    max_other_tokens = max(other_tokens)
    
    if target_tokens > max_other_tokens * 2:
        return True
    
    return False


<<<<<<< HEAD
def calculate_attention_entropy(token_probs: List[float]) -> Dict[str, float]:
    """计算token概率序列的注意力熵
    
    Args:
        token_probs: token概率列表
    
    Returns:
        包含各种熵指标的字典
    """
    if not token_probs or len(token_probs) == 0:
        return {
            'mean_entropy': None,
            'min_entropy': None,
            'max_entropy': None,
            'std_entropy': None,
            'total_entropy': None,
            'mean_prob': None,
            'min_prob': None,
        }
    
    probs = np.array(token_probs)
    
    # 计算每个token的熵 H = -p*log(p)
    # 这里的熵衡量的是单个token生成的确定性
    # 接近1的概率 -> 低熵(确定)，接近0的概率 -> 高熵(不确定)
    entropies = -probs * np.log(probs + 1e-10)
    
    return {
        'mean_entropy': float(np.mean(entropies)),
        'min_entropy': float(np.min(entropies)),
        'max_entropy': float(np.max(entropies)),
        'std_entropy': float(np.std(entropies)),
        'total_entropy': float(np.sum(entropies)),
        'mean_prob': float(np.mean(probs)),
        'min_prob': float(np.min(probs)),
        'low_confidence_tokens': int(np.sum(probs < 0.5)),  # 概率<0.5的token数
        'high_confidence_tokens': int(np.sum(probs > 0.9)),  # 概率>0.9的token数
    }


def analyze_workload_file(workload_path: Path) -> List[Dict[str, Any]]:
    """分析单个workload文件，找出所有straggler情况并计算注意力熵
=======
def analyze_workload_file(workload_path: Path) -> List[Dict[str, Any]]:
    """分析单个workload文件，找出所有straggler情况
>>>>>>> 207a1d0 (A6000 0401)
    
    Args:
        workload_path: workload文件路径
    
    Returns:
        straggler情况列表
    """
    stragglers = []
    
    try:
        with open(workload_path, 'r') as f:
            workload = json.load(f)
        
        question_id = workload.get('question_id', 'unknown')
        decode_steps = workload.get('decode', {}).get('steps', [])
        
        for step_info in decode_steps:
            step_num = step_info.get('step', -1)
            branch_count = step_info.get('branch_count', 0)
            branch_tokens = step_info.get('branch_tokens', [])
            branch_rewards = step_info.get('branch_rewards', [])
<<<<<<< HEAD
            branch_token_probs = step_info.get('branch_token_probs', [])
=======
>>>>>>> 207a1d0 (A6000 0401)
            selected_branch_index = step_info.get('selected_branch_index', -1)
            
            # 检查每个branch是否为straggler
            for branch_idx, tokens in enumerate(branch_tokens):
                if is_straggler(branch_tokens, branch_idx):
<<<<<<< HEAD
                    # 计算注意力熵（如果有token_probs数据）
                    entropy_metrics = None
                    has_token_probs = False
                    
                    # 需要找到对应branch的token_probs
                    # 如果这个branch是selected_branch，则可能在branch_token_probs[0]中
                    if branch_token_probs and len(branch_token_probs) > 0:
                        # 检查是否为选中的分支
                        if branch_idx == selected_branch_index:
                            token_probs = branch_token_probs[0]
                            entropy_metrics = calculate_attention_entropy(token_probs)
                            has_token_probs = True
                        # 如果不是选中分支，检查branch_token_probs是否有多个
                        elif branch_idx < len(branch_token_probs):
                            token_probs = branch_token_probs[branch_idx]
                            if token_probs:  # 确保不是空列表
                                entropy_metrics = calculate_attention_entropy(token_probs)
                                has_token_probs = True
                    
=======
>>>>>>> 207a1d0 (A6000 0401)
                    straggler_info = {
                        # 位置信息
                        'source_file': str(workload_path),
                        'question_id': question_id,
                        'step': step_num,
                        
                        # Straggler信息
                        'straggler_branch_index': branch_idx,
                        'straggler_tokens': tokens,
                        'straggler_reward': branch_rewards[branch_idx] if branch_idx < len(branch_rewards) else None,
                        'is_selected': (branch_idx == selected_branch_index),
                        
<<<<<<< HEAD
                        # 注意力熵信息
                        'has_token_probs': has_token_probs,
                        'entropy_metrics': entropy_metrics,
                        
=======
>>>>>>> 207a1d0 (A6000 0401)
                        # 完整的step信息
                        'step_info': {
                            'step': step_num,
                            'branch_count': branch_count,
                            'branch_tokens': branch_tokens,
                            'branch_rewards': branch_rewards,
                            'selected_branch_index': selected_branch_index,
<<<<<<< HEAD
                            'has_branch_token_probs': len(branch_token_probs) > 0,
                            'num_branches_with_probs': len(branch_token_probs),
=======
>>>>>>> 207a1d0 (A6000 0401)
                        },
                        
                        # 统计信息
                        'stats': {
                            'max_other_tokens': max([t for i, t in enumerate(branch_tokens) if i != branch_idx]),
                            'ratio_to_max_other': tokens / max([t for i, t in enumerate(branch_tokens) if i != branch_idx]),
                            'total_branches': len(branch_tokens),
                        }
                    }
                    
                    stragglers.append(straggler_info)
                    
<<<<<<< HEAD
                    entropy_str = f", entropy={entropy_metrics['mean_entropy']:.4f}" if has_token_probs else ", no token_probs"
                    logger.debug(f"Found straggler in {question_id} step {step_num}: "
                               f"branch {branch_idx} has {tokens} tokens "
                               f"(ratio: {straggler_info['stats']['ratio_to_max_other']:.2f}x{entropy_str})")
    
    except Exception as e:
        logger.error(f"Error processing {workload_path}: {e}")
        import traceback
        traceback.print_exc()
=======
                    logger.debug(f"Found straggler in {question_id} step {step_num}: "
                               f"branch {branch_idx} has {tokens} tokens "
                               f"(ratio: {straggler_info['stats']['ratio_to_max_other']:.2f}x)")
    
    except Exception as e:
        logger.error(f"Error processing {workload_path}: {e}")
>>>>>>> 207a1d0 (A6000 0401)
    
    return stragglers


def analyze_all_workloads(workload_dir: Path, output_file: Path) -> None:
    """分析目录中的所有workload文件
    
    Args:
        workload_dir: workload目录路径
        output_file: 输出文件路径
    """
    all_stragglers = []
    total_files = 0
    processed_files = 0
    
    # 递归查找所有JSON文件
    workload_files = list(workload_dir.rglob("*_workload.json"))
    total_files = len(workload_files)
    
    logger.info(f"Found {total_files} workload files")
    
    for workload_path in workload_files:
        stragglers = analyze_workload_file(workload_path)
        all_stragglers.extend(stragglers)
        
        if stragglers:
            processed_files += 1
            logger.info(f"File {processed_files}/{total_files}: Found {len(stragglers)} stragglers in {workload_path.name}")
    
    # 保存结果
    output_data = {
        'summary': {
            'total_workload_files': total_files,
            'files_with_stragglers': processed_files,
            'total_straggler_branches': len(all_stragglers),
<<<<<<< HEAD
            'stragglers_with_token_probs': sum(1 for s in all_stragglers if s['has_token_probs']),
=======
>>>>>>> 207a1d0 (A6000 0401)
        },
        'stragglers': all_stragglers
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Analysis complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Total workload files: {total_files}")
    logger.info(f"Files with stragglers: {processed_files}")
    logger.info(f"Total straggler branches: {len(all_stragglers)}")
<<<<<<< HEAD
    logger.info(f"Stragglers with token_probs: {output_data['summary']['stragglers_with_token_probs']}")
=======
>>>>>>> 207a1d0 (A6000 0401)
    logger.info(f"Output saved to: {output_file}")
    
    # 统计信息
    if all_stragglers:
        token_counts = [s['straggler_tokens'] for s in all_stragglers]
        ratios = [s['stats']['ratio_to_max_other'] for s in all_stragglers]
        selected_count = sum(1 for s in all_stragglers if s['is_selected'])
        
        logger.info(f"\nStraggler Statistics:")
        logger.info(f"  Token count: min={min(token_counts)}, max={max(token_counts)}, "
                   f"avg={sum(token_counts)/len(token_counts):.1f}")
        logger.info(f"  Ratio to max_other: min={min(ratios):.2f}x, max={max(ratios):.2f}x, "
                   f"avg={sum(ratios)/len(ratios):.2f}x")
        logger.info(f"  Selected as final branch: {selected_count}/{len(all_stragglers)} "
                   f"({selected_count/len(all_stragglers)*100:.1f}%)")
<<<<<<< HEAD
        
        # 注意力熵统计
        stragglers_with_entropy = [s for s in all_stragglers if s['has_token_probs']]
        if stragglers_with_entropy:
            logger.info(f"\nAttention Entropy Statistics ({len(stragglers_with_entropy)} stragglers):")
            
            mean_entropies = [s['entropy_metrics']['mean_entropy'] for s in stragglers_with_entropy]
            mean_probs = [s['entropy_metrics']['mean_prob'] for s in stragglers_with_entropy]
            
            logger.info(f"  Mean entropy: min={min(mean_entropies):.4f}, max={max(mean_entropies):.4f}, "
                       f"avg={np.mean(mean_entropies):.4f}")
            logger.info(f"  Mean prob: min={min(mean_probs):.4f}, max={max(mean_probs):.4f}, "
                       f"avg={np.mean(mean_probs):.4f}")
            
            # 比较选中 vs 未选中的straggler的熵
            selected_entropies = [s['entropy_metrics']['mean_entropy'] 
                                 for s in stragglers_with_entropy if s['is_selected']]
            not_selected_entropies = [s['entropy_metrics']['mean_entropy'] 
                                     for s in stragglers_with_entropy if not s['is_selected']]
            
            if selected_entropies:
                logger.info(f"  Selected stragglers mean entropy: {np.mean(selected_entropies):.4f} "
                           f"(n={len(selected_entropies)})")
            if not_selected_entropies:
                logger.info(f"  Not selected stragglers mean entropy: {np.mean(not_selected_entropies):.4f} "
                           f"(n={len(not_selected_entropies)})")
            
            if selected_entropies and not_selected_entropies:
                diff = np.mean(selected_entropies) - np.mean(not_selected_entropies)
                logger.info(f"  Difference (selected - not_selected): {diff:.4f}")
                logger.info(f"  → {'Selected stragglers have HIGHER entropy' if diff > 0 else 'Selected stragglers have LOWER entropy'}")
=======
>>>>>>> 207a1d0 (A6000 0401)


def generate_summary_report(straggler_file: Path, output_report: Path) -> None:
    """生成易读的摘要报告
    
    Args:
        straggler_file: straggler数据文件
        output_report: 输出报告文件
    """
    with open(straggler_file, 'r') as f:
        data = json.load(f)
    
    stragglers = data['stragglers']
    
    lines = []
    lines.append("=" * 80)
<<<<<<< HEAD
    lines.append("STRAGGLER BRANCH ANALYSIS REPORT WITH ATTENTION ENTROPY")
=======
    lines.append("STRAGGLER BRANCH ANALYSIS REPORT")
>>>>>>> 207a1d0 (A6000 0401)
    lines.append("=" * 80)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 80)
    for key, value in data['summary'].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    
<<<<<<< HEAD
    # Entropy analysis summary
    stragglers_with_entropy = [s for s in stragglers if s['has_token_probs']]
    if stragglers_with_entropy:
        lines.append("ATTENTION ENTROPY SUMMARY")
        lines.append("-" * 80)
        
        mean_entropies = [s['entropy_metrics']['mean_entropy'] for s in stragglers_with_entropy]
        mean_probs = [s['entropy_metrics']['mean_prob'] for s in stragglers_with_entropy]
        
        lines.append(f"  Total stragglers with entropy data: {len(stragglers_with_entropy)}")
        lines.append(f"  Mean entropy: min={min(mean_entropies):.4f}, max={max(mean_entropies):.4f}, "
                    f"avg={np.mean(mean_entropies):.4f}")
        lines.append(f"  Mean prob: min={min(mean_probs):.4f}, max={max(mean_probs):.4f}, "
                    f"avg={np.mean(mean_probs):.4f}")
        
        # 比较选中 vs 未选中
        selected_entropies = [s['entropy_metrics']['mean_entropy'] 
                             for s in stragglers_with_entropy if s['is_selected']]
        not_selected_entropies = [s['entropy_metrics']['mean_entropy'] 
                                 for s in stragglers_with_entropy if not s['is_selected']]
        
        if selected_entropies and not_selected_entropies:
            lines.append(f"\n  Selected stragglers (n={len(selected_entropies)}):")
            lines.append(f"    Mean entropy: {np.mean(selected_entropies):.4f}")
            lines.append(f"    Mean prob: {np.mean([s['entropy_metrics']['mean_prob'] for s in stragglers_with_entropy if s['is_selected']]):.4f}")
            
            lines.append(f"\n  Not selected stragglers (n={len(not_selected_entropies)}):")
            lines.append(f"    Mean entropy: {np.mean(not_selected_entropies):.4f}")
            lines.append(f"    Mean prob: {np.mean([s['entropy_metrics']['mean_prob'] for s in stragglers_with_entropy if not s['is_selected']]):.4f}")
            
            diff = np.mean(selected_entropies) - np.mean(not_selected_entropies)
            lines.append(f"\n  Entropy difference (selected - not_selected): {diff:.4f}")
            lines.append(f"  → {'Selected stragglers have HIGHER entropy (less confident)' if diff > 0 else 'Selected stragglers have LOWER entropy (more confident)'}")
        
        lines.append("")
    
=======
>>>>>>> 207a1d0 (A6000 0401)
    # Group by question
    question_groups = {}
    for s in stragglers:
        qid = s['question_id']
        if qid not in question_groups:
            question_groups[qid] = []
        question_groups[qid].append(s)
    
    lines.append(f"STRAGGLERS BY QUESTION ({len(question_groups)} questions)")
    lines.append("-" * 80)
    
    for qid, straggler_list in sorted(question_groups.items()):
        lines.append(f"\n{qid} - {len(straggler_list)} straggler(s)")
        lines.append(f"  Source: {straggler_list[0]['source_file']}")
        
        for s in straggler_list:
            step_info = s['step_info']
            stats = s['stats']
            
            lines.append(f"\n  Step {s['step']}:")
            lines.append(f"    Straggler branch: {s['straggler_branch_index']} "
                        f"({'SELECTED' if s['is_selected'] else 'not selected'})")
            lines.append(f"    Tokens: {s['straggler_tokens']} "
                        f"(ratio: {stats['ratio_to_max_other']:.2f}x)")
            lines.append(f"    Reward: {s['straggler_reward']:.6f}" if s['straggler_reward'] else "    Reward: N/A")
<<<<<<< HEAD
            
            # 添加熵信息
            if s['has_token_probs'] and s['entropy_metrics']:
                ent = s['entropy_metrics']
                lines.append(f"    Attention Entropy:")
                lines.append(f"      Mean: {ent['mean_entropy']:.4f}, Std: {ent['std_entropy']:.4f}")
                lines.append(f"      Mean prob: {ent['mean_prob']:.4f}, Min prob: {ent['min_prob']:.4f}")
                lines.append(f"      High confidence tokens (>0.9): {ent['high_confidence_tokens']}/{s['straggler_tokens']} "
                            f"({ent['high_confidence_tokens']/s['straggler_tokens']*100:.1f}%)")
                lines.append(f"      Low confidence tokens (<0.5): {ent['low_confidence_tokens']}/{s['straggler_tokens']} "
                            f"({ent['low_confidence_tokens']/s['straggler_tokens']*100:.1f}%)")
            else:
                lines.append(f"    Attention Entropy: N/A (no token_probs data)")
            
=======
>>>>>>> 207a1d0 (A6000 0401)
            lines.append(f"    All branches ({step_info['branch_count']}):")
            
            for i, (tokens, reward) in enumerate(zip(step_info['branch_tokens'], 
                                                     step_info['branch_rewards'])):
                marker = " <-- STRAGGLER" if i == s['straggler_branch_index'] else ""
                selected = " [SELECTED]" if i == step_info['selected_branch_index'] else ""
                lines.append(f"      Branch {i}: {tokens} tokens, reward={reward:.6f}{selected}{marker}")
    
    lines.append("\n" + "=" * 80)
    
    # 保存报告
    with open(output_report, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Summary report saved to: {output_report}")


def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description='识别符合straggler定义的branch并计算注意力熵')
    parser.add_argument('--input', 
                       default='../wordload/model_workloads/16384_4_1',
=======
    parser = argparse.ArgumentParser(description='识别符合straggler定义的branch')
    parser.add_argument('--input', 
                       default='../wordload/model_workloads_need',
>>>>>>> 207a1d0 (A6000 0401)
                       help='Workload目录路径')
    parser.add_argument('--output',
                       default='straggler_analysis.json',
                       help='输出JSON文件')
    parser.add_argument('--report',
                       default='straggler_report.txt',
                       help='输出摘要报告文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    workload_dir = Path(args.input)
    output_file = Path(__file__).parent / args.output
    report_file = Path(__file__).parent / args.report
    
    if not workload_dir.exists():
        logger.error(f"Workload directory not found: {workload_dir}")
        return
    
    logger.info(f"Input directory: {workload_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Report file: {report_file}")
    
    # 分析workload文件
    analyze_all_workloads(workload_dir, output_file)
    
    # 生成摘要报告
    if output_file.exists():
        generate_summary_report(output_file, report_file)


if __name__ == '__main__':
    main()
