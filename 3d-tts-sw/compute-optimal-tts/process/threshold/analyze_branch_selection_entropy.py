#!/usr/bin/env python3
"""
分析16384_4_1数据集：
1. 被选中branch vs 未选中branch的熵特征对比
2. 做对题目 vs 做错题目的straggler表现
3. 基于熵的剪枝策略模拟
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def calculate_attention_entropy(token_probs: List[float]) -> Dict[str, float]:
    """计算token概率序列的注意力熵"""
    if not token_probs or len(token_probs) == 0:
        return {
            'mean_entropy': None,
            'min_entropy': None,
            'max_entropy': None,
            'std_entropy': None,
            'mean_prob': None,
            'min_prob': None,
            'low_conf_tokens': 0,
            'high_conf_tokens': 0,
        }
    
    probs = np.array(token_probs)
    entropies = -probs * np.log(probs + 1e-10)
    
    return {
        'mean_entropy': float(np.mean(entropies)),
        'min_entropy': float(np.min(entropies)),
        'max_entropy': float(np.max(entropies)),
        'std_entropy': float(np.std(entropies)),
        'mean_prob': float(np.mean(probs)),
        'min_prob': float(np.min(probs)),
        'low_conf_tokens': int(np.sum(probs < 0.5)),
        'high_conf_tokens': int(np.sum(probs > 0.9)),
    }


def is_straggler(branch_tokens: List[int], branch_index: int) -> bool:
    """判断是否为straggler"""
    if len(branch_tokens) <= 1:
        return False
    
    target_tokens = branch_tokens[branch_index]
    
    if target_tokens <= 100:
        return False
    
    other_tokens = [t for i, t in enumerate(branch_tokens) if i != branch_index]
    if not other_tokens:
        return False
    
    max_other_tokens = max(other_tokens)
    
    return target_tokens > max_other_tokens * 2


def analyze_workload_file(filepath: Path) -> Dict[str, Any]:
    """分析单个workload文件"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    question_id = data['question_id']
    steps = data['decode']['steps']
    
    selected_branches = []
    unselected_branches = []
    straggler_info = []
    all_step_info = []
    
    for step_idx, step in enumerate(steps):
        branch_count = step['branch_count']
        
        if branch_count <= 1:
            continue
        
        branch_tokens = step['branch_tokens']
        branch_rewards = step.get('branch_rewards', [])
        selected_idx = step.get('selected_branch_index', -1)
        branch_token_probs = step.get('branch_token_probs', [])
        
        step_branches = []
        
        for br_idx in range(branch_count):
            token_probs = branch_token_probs[br_idx] if br_idx < len(branch_token_probs) else []
            entropy_metrics = calculate_attention_entropy(token_probs)
            
            is_strag = is_straggler(branch_tokens, br_idx)
            is_selected = (br_idx == selected_idx)
            
            branch_info = {
                'question_id': question_id,
                'step': step_idx,
                'branch_idx': br_idx,
                'branch_count': branch_count,
                'token_count': branch_tokens[br_idx],
                'reward': branch_rewards[br_idx] if br_idx < len(branch_rewards) else None,
                'is_selected': is_selected,
                'is_straggler': is_strag,
                **entropy_metrics
            }
            
            step_branches.append(branch_info)
            
            if is_selected:
                selected_branches.append(branch_info)
            else:
                unselected_branches.append(branch_info)
            
            if is_strag:
                straggler_info.append(branch_info)
        
        all_step_info.append({
            'step': step_idx,
            'branch_count': branch_count,
            'branches': step_branches,
            'selected_idx': selected_idx,
            'has_straggler': any(b['is_straggler'] for b in step_branches)
        })
    
    return {
        'question_id': question_id,
        'selected_branches': selected_branches,
        'unselected_branches': unselected_branches,
        'straggler_info': straggler_info,
        'all_step_info': all_step_info
    }


def compare_entropy_distributions(selected: List[Dict], unselected: List[Dict], metric: str) -> Dict:
    """比较选中和未选中branch的某个熵指标分布"""
    selected_values = [b[metric] for b in selected if b[metric] is not None]
    unselected_values = [b[metric] for b in unselected if b[metric] is not None]
    
    if not selected_values or not unselected_values:
        return None
    
    return {
        'selected_mean': np.mean(selected_values),
        'selected_median': np.median(selected_values),
        'selected_std': np.std(selected_values),
        'unselected_mean': np.mean(unselected_values),
        'unselected_median': np.median(unselected_values),
        'unselected_std': np.std(unselected_values),
        'mean_diff': np.mean(selected_values) - np.mean(unselected_values),
        'median_diff': np.median(selected_values) - np.median(unselected_values),
    }


def simulate_pruning(all_results: List[Dict], pruning_strategy: Dict) -> Dict:
    """模拟剪枝策略"""
    total_branches = 0
    pruned_branches = 0
    pruned_selected = 0
    pruned_stragglers = 0
    total_stragglers = 0
    
    for result in all_results:
        for step_info in result['all_step_info']:
            for branch in step_info['branches']:
                total_branches += 1
                
                if branch['is_straggler']:
                    total_stragglers += 1
                
                # 检查是否应该剪枝
                should_prune = False
                
                if pruning_strategy.get('max_entropy_threshold'):
                    if branch['max_entropy'] and branch['max_entropy'] > pruning_strategy['max_entropy_threshold']:
                        should_prune = True
                
                if pruning_strategy.get('min_prob_threshold'):
                    if branch['min_prob'] and branch['min_prob'] < pruning_strategy['min_prob_threshold']:
                        should_prune = True
                
                if pruning_strategy.get('mean_entropy_threshold'):
                    if branch['mean_entropy'] and branch['mean_entropy'] > pruning_strategy['mean_entropy_threshold']:
                        should_prune = True
                
                if pruning_strategy.get('low_conf_token_threshold'):
                    if branch['low_conf_tokens'] > pruning_strategy['low_conf_token_threshold']:
                        should_prune = True
                
                if should_prune:
                    pruned_branches += 1
                    if branch['is_selected']:
                        pruned_selected += 1
                    if branch['is_straggler']:
                        pruned_stragglers += 1
    
    return {
        'total_branches': total_branches,
        'pruned_branches': pruned_branches,
        'pruned_ratio': pruned_branches / total_branches if total_branches > 0 else 0,
        'pruned_selected': pruned_selected,
        'total_stragglers': total_stragglers,
        'pruned_stragglers': pruned_stragglers,
        'straggler_recall': pruned_stragglers / total_stragglers if total_stragglers > 0 else 0,
    }


def main():
    source_dir = Path('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads/16384_4_1')
    
    # 做对的题目
    correct_questions = [1, 3, 6, 15, 19]
    
    workload_files = sorted(source_dir.glob('question_*_workload.json'))
    
    print("=" * 100)
    print("分析16384_4_1数据集 - Branch选择与熵的关系")
    print("=" * 100)
    print(f"\n找到 {len(workload_files)} 个workload文件")
    print(f"做对的题目: {correct_questions}\n")
    
    # 分析所有文件
    all_results = []
    correct_results = []
    incorrect_results = []
    
    for filepath in workload_files:
        print(f"处理: {filepath.name}")
        result = analyze_workload_file(filepath)
        all_results.append(result)
        
        q_num = int(result['question_id'].replace('question_', ''))
        if q_num in correct_questions:
            correct_results.append(result)
        else:
            incorrect_results.append(result)
    
    # 汇总统计
    all_selected = []
    all_unselected = []
    
    for result in all_results:
        all_selected.extend(result['selected_branches'])
        all_unselected.extend(result['unselected_branches'])
    
    print(f"\n处理完成！")
    print(f"  总选中branches: {len(all_selected)}")
    print(f"  总未选中branches: {len(all_unselected)}")
    
    # ===== 分析1: 被选中 vs 未选中的熵特征 =====
    print("\n" + "=" * 100)
    print("分析1: 被选中 vs 未选中 Branch的熵特征对比")
    print("=" * 100)
    
    metrics = ['mean_entropy', 'min_entropy', 'max_entropy', 'std_entropy', 
               'mean_prob', 'min_prob', 'low_conf_tokens', 'high_conf_tokens']
    
    for metric in metrics:
        comparison = compare_entropy_distributions(all_selected, all_unselected, metric)
        if comparison:
            print(f"\n【{metric}】")
            print(f"  选中branches:   均值={comparison['selected_mean']:.4f}, 中位数={comparison['selected_median']:.4f}")
            print(f"  未选中branches: 均值={comparison['unselected_mean']:.4f}, 中位数={comparison['unselected_median']:.4f}")
            print(f"  差异:           均值差={comparison['mean_diff']:.4f}, 中位数差={comparison['median_diff']:.4f}")
            
            if abs(comparison['mean_diff']) > 0.01:
                direction = "更低" if comparison['mean_diff'] < 0 else "更高"
                print(f"  💡 选中的branch在{metric}上{direction}！")
    
    # ===== 分析2: 做对 vs 做错题目的straggler表现 =====
    print("\n\n" + "=" * 100)
    print("分析2: 做对题目 vs 做错题目的Straggler表现")
    print("=" * 100)
    
    correct_stragglers = []
    incorrect_stragglers = []
    
    for result in correct_results:
        correct_stragglers.extend(result['straggler_info'])
    
    for result in incorrect_results:
        incorrect_stragglers.extend(result['straggler_info'])
    
    print(f"\n做对题目 ({len(correct_questions)}题):")
    print(f"  总steps: {sum(len(r['all_step_info']) for r in correct_results)}")
    print(f"  Straggler数量: {len(correct_stragglers)}")
    print(f"  平均每题straggler: {len(correct_stragglers)/len(correct_questions):.2f}")
    
    if correct_stragglers:
        print(f"  Straggler熵特征:")
        print(f"    mean_entropy: {np.mean([s['mean_entropy'] for s in correct_stragglers if s['mean_entropy']]):.4f}")
        print(f"    max_entropy: {np.mean([s['max_entropy'] for s in correct_stragglers if s['max_entropy']]):.4f}")
        print(f"    min_prob: {np.mean([s['min_prob'] for s in correct_stragglers if s['min_prob']]):.4f}")
    
    print(f"\n做错题目 ({len(workload_files) - len(correct_questions)}题):")
    print(f"  总steps: {sum(len(r['all_step_info']) for r in incorrect_results)}")
    print(f"  Straggler数量: {len(incorrect_stragglers)}")
    print(f"  平均每题straggler: {len(incorrect_stragglers)/(len(workload_files)-len(correct_questions)):.2f}")
    
    if incorrect_stragglers:
        print(f"  Straggler熵特征:")
        print(f"    mean_entropy: {np.mean([s['mean_entropy'] for s in incorrect_stragglers if s['mean_entropy']]):.4f}")
        print(f"    max_entropy: {np.mean([s['max_entropy'] for s in incorrect_stragglers if s['max_entropy']]):.4f}")
        print(f"    min_prob: {np.mean([s['min_prob'] for s in incorrect_stragglers if s['min_prob']]):.4f}")
    
    # 做对题目的选中branch特征
    correct_selected = []
    incorrect_selected = []
    
    for result in correct_results:
        correct_selected.extend(result['selected_branches'])
    
    for result in incorrect_results:
        incorrect_selected.extend(result['selected_branches'])
    
    print(f"\n做对题目的选中branch熵特征:")
    if correct_selected:
        print(f"  mean_entropy: {np.mean([s['mean_entropy'] for s in correct_selected if s['mean_entropy']]):.4f}")
        print(f"  max_entropy: {np.mean([s['max_entropy'] for s in correct_selected if s['max_entropy']]):.4f}")
        print(f"  min_prob: {np.mean([s['min_prob'] for s in correct_selected if s['min_prob']]):.4f}")
    
    print(f"\n做错题目的选中branch熵特征:")
    if incorrect_selected:
        print(f"  mean_entropy: {np.mean([s['mean_entropy'] for s in incorrect_selected if s['mean_entropy']]):.4f}")
        print(f"  max_entropy: {np.mean([s['max_entropy'] for s in incorrect_selected if s['max_entropy']]):.4f}")
        print(f"  min_prob: {np.mean([s['min_prob'] for s in incorrect_selected if s['min_prob']]):.4f}")
    
    # ===== 分析3: 剪枝策略模拟 =====
    print("\n\n" + "=" * 100)
    print("分析3: 基于熵的剪枝策略模拟")
    print("=" * 100)
    
    # 测试不同的剪枝策略
    strategies = [
        {'name': '高max_entropy剪枝', 'max_entropy_threshold': 1.5},
        {'name': '极高max_entropy剪枝', 'max_entropy_threshold': 2.0},
        {'name': '低min_prob剪枝', 'min_prob_threshold': 0.1},
        {'name': '极低min_prob剪枝', 'min_prob_threshold': 0.05},
        {'name': '高mean_entropy剪枝', 'mean_entropy_threshold': 0.5},
        {'name': '多低置信度token剪枝', 'low_conf_token_threshold': 10},
        {'name': '组合策略1', 'max_entropy_threshold': 2.0, 'min_prob_threshold': 0.05},
    ]
    
    for strategy in strategies:
        strategy_config = {k: v for k, v in strategy.items() if k != 'name'}
        result = simulate_pruning(all_results, strategy_config)
        
        print(f"\n【{strategy['name']}】")
        print(f"  剪枝配置: {strategy_config}")
        print(f"  总branches: {result['total_branches']}")
        print(f"  剪枝branches: {result['pruned_branches']} ({result['pruned_ratio']*100:.1f}%)")
        print(f"  剪掉的选中branches: {result['pruned_selected']} ⚠️")
        print(f"  总stragglers: {result['total_stragglers']}")
        print(f"  剪掉的stragglers: {result['pruned_stragglers']} ({result['straggler_recall']*100:.1f}%)")
        
        if result['pruned_selected'] > 0:
            print(f"  ⚠️  警告: 此策略会剪掉 {result['pruned_selected']} 个被选中的branch！")
        
        if result['straggler_recall'] > 0.8 and result['pruned_selected'] == 0:
            print(f"  ✅ 这个策略不错：能剪掉80%+的stragglers且不误伤选中branch！")
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)


if __name__ == '__main__':
    main()
