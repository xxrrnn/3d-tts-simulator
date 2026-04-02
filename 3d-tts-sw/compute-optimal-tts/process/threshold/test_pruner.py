#!/usr/bin/env python3
"""
测试动态剪枝器的效果

在测试集(16384_4_2)上评估剪枝器,看看:
1. 是否能保留正确题目的正确分支
2. 是否能剪枝错误题目的错误分支
3. 剪枝率和准确率
"""

import os
import json
import sys
from dynamic_pruner import DynamicPruner
from typing import Dict, List

def load_question_data(question_dir: str) -> tuple:
    """加载单个题目的数据"""
    record_file = os.path.join(question_dir, "record_0.jsonl")
    
    if not os.path.exists(record_file):
        return None, None
    
    try:
        with open(record_file, 'r') as f:
            data = json.loads(f.read())
        
        if 'output' not in data or 'result' not in data:
            return None, None
        
        outputs = data['output'] if isinstance(data['output'], list) else [data['output']]
        result = data['result']
        
        return outputs, result
    
    except Exception as e:
        print(f"Error loading {question_dir}: {e}")
        return None, None

def test_on_dataset(base_dir: str, 
                   correct_indices: List[int],
                   pruner: DynamicPruner,
                   metric: str = 'prm_avg_max') -> Dict:
    """在数据集上测试剪枝器"""
    
    results = {
        'correct_questions': {
            'total': 0,
            'branches_kept': 0,
            'branches_pruned': 0,
            'best_branch_kept': 0,
            'best_branch_pruned': 0
        },
        'incorrect_questions': {
            'total': 0,
            'branches_kept': 0,
            'branches_pruned': 0,
        },
        'details': []
    }
    
    question_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("question_")],
                          key=lambda x: int(x.split('_')[1]))
    
    for question_dir_name in question_dirs:
        question_idx = int(question_dir_name.split('_')[1])
        question_dir = os.path.join(base_dir, question_dir_name)
        
        outputs, result = load_question_data(question_dir)
        
        if outputs is None or result is None:
            continue
        
        is_correct = question_idx in correct_indices
        
        # 确定哪个分支是最好的(根据metric)
        best_branch_idx = -1
        best_metric_value = -1
        
        # 简化:假设最高reward的分支是最好的
        for i, output in enumerate(outputs):
            if 'reward_history' in output and len(output['reward_history']) > 0:
                final_reward = output['reward_history'][-1]
                if final_reward > best_metric_value:
                    best_metric_value = final_reward
                    best_branch_idx = i
        
        # 对每个分支进行剪枝测试
        question_detail = {
            'question_idx': question_idx,
            'is_correct': is_correct,
            'num_branches': len(outputs),
            'branches': []
        }
        
        # 模拟运行时剪枝决策
        # 我们在多个步骤测试
        max_steps = max(len(output.get('reward_history', [])) for output in outputs)
        
        for branch_idx, branch_output in enumerate(outputs):
            branch_info = {
                'branch_idx': branch_idx,
                'is_best': (branch_idx == best_branch_idx),
                'pruning_decisions': []
            }
            
            # 在不同步骤测试剪枝决策
            steps_to_test = [max_steps // 4, max_steps // 2, 3 * max_steps // 4, max_steps - 1]
            steps_to_test = [s for s in steps_to_test if s > 0 and s < len(branch_output.get('reward_history', []))]
            
            if not steps_to_test and len(branch_output.get('reward_history', [])) > 0:
                steps_to_test = [len(branch_output.get('reward_history', [])) - 1]
            
            for step_idx in steps_to_test:
                # 截取到当前步的数据
                branch_data_at_step = {
                    'reward_history': branch_output.get('reward_history', [])[:step_idx+1],
                    'token_prob_history': branch_output.get('token_prob_history', [])[:step_idx+1],
                    'prob_history': branch_output.get('prob_history', [])[:step_idx+1]
                }
                
                all_branches_at_step = []
                for other_output in outputs:
                    all_branches_at_step.append({
                        'reward_history': other_output.get('reward_history', [])[:step_idx+1],
                        'token_prob_history': other_output.get('token_prob_history', [])[:step_idx+1],
                        'prob_history': other_output.get('prob_history', [])[:step_idx+1]
                    })
                
                should_prune, reason = pruner.should_prune_branch(
                    branch_data_at_step,
                    all_branches_at_step,
                    current_step=step_idx,
                    max_steps=max_steps
                )
                
                branch_info['pruning_decisions'].append({
                    'step': step_idx,
                    'should_prune': should_prune,
                    'reason': reason
                })
            
            # 最终决策:如果任何一步决定剪枝,就认为被剪枝了
            final_pruned = any(d['should_prune'] for d in branch_info['pruning_decisions'])
            branch_info['final_pruned'] = final_pruned
            
            question_detail['branches'].append(branch_info)
            
            # 统计
            if is_correct:
                if final_pruned:
                    results['correct_questions']['branches_pruned'] += 1
                    if branch_idx == best_branch_idx:
                        results['correct_questions']['best_branch_pruned'] += 1
                else:
                    results['correct_questions']['branches_kept'] += 1
                    if branch_idx == best_branch_idx:
                        results['correct_questions']['best_branch_kept'] += 1
            else:
                if final_pruned:
                    results['incorrect_questions']['branches_pruned'] += 1
                else:
                    results['incorrect_questions']['branches_kept'] += 1
        
        if is_correct:
            results['correct_questions']['total'] += 1
        else:
            results['incorrect_questions']['total'] += 1
        
        results['details'].append(question_detail)
    
    return results

def print_results(results: Dict):
    """打印测试结果"""
    
    print("\n" + "="*80)
    print("剪枝器测试结果")
    print("="*80)
    
    print("\n【正确题目】")
    correct = results['correct_questions']
    print(f"  题目数量: {correct['total']}")
    print(f"  分支总数: {correct['branches_kept'] + correct['branches_pruned']}")
    print(f"  保留分支: {correct['branches_kept']}")
    print(f"  剪枝分支: {correct['branches_pruned']}")
    
    if correct['branches_kept'] + correct['branches_pruned'] > 0:
        keep_rate = correct['branches_kept'] / (correct['branches_kept'] + correct['branches_pruned'])
        print(f"  保留率: {keep_rate:.2%}")
    
    print(f"\n  最优分支保留: {correct['best_branch_kept']}")
    print(f"  最优分支剪枝: {correct['best_branch_pruned']}")
    
    if correct['total'] > 0:
        best_keep_rate = correct['best_branch_kept'] / correct['total']
        print(f"  最优分支保留率: {best_keep_rate:.2%}")
    
    print("\n【错误题目】")
    incorrect = results['incorrect_questions']
    print(f"  题目数量: {incorrect['total']}")
    print(f"  分支总数: {incorrect['branches_kept'] + incorrect['branches_pruned']}")
    print(f"  保留分支: {incorrect['branches_kept']}")
    print(f"  剪枝分支: {incorrect['branches_pruned']}")
    
    if incorrect['branches_kept'] + incorrect['branches_pruned'] > 0:
        prune_rate = incorrect['branches_pruned'] / (incorrect['branches_kept'] + incorrect['branches_pruned'])
        print(f"  剪枝率: {prune_rate:.2%}")
    
    print("\n【总体】")
    total_branches = (correct['branches_kept'] + correct['branches_pruned'] + 
                     incorrect['branches_kept'] + incorrect['branches_pruned'])
    total_pruned = correct['branches_pruned'] + incorrect['branches_pruned']
    
    if total_branches > 0:
        overall_prune_rate = total_pruned / total_branches
        print(f"  总分支数: {total_branches}")
        print(f"  总剪枝数: {total_pruned}")
        print(f"  总剪枝率: {overall_prune_rate:.2%}")
    
    # 详细分析几个案例
    print("\n" + "="*80)
    print("案例分析 (前5个正确题目)")
    print("="*80)
    
    correct_cases = [d for d in results['details'] if d['is_correct']][:5]
    for case in correct_cases:
        print(f"\n题目 {case['question_idx']}:")
        for branch in case['branches']:
            status = "被剪枝" if branch['final_pruned'] else "保留"
            best_mark = " [最优]" if branch['is_best'] else ""
            print(f"  分支 {branch['branch_idx']}{best_mark}: {status}")
            if branch['pruning_decisions']:
                last_decision = branch['pruning_decisions'][-1]
                print(f"    最后决策: {last_decision['reason']}")

def main():
    # 初始化剪枝器
    model_path = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/threshold_model.json"
    pruner = DynamicPruner(model_path)
    
    print("动态剪枝器测试")
    print(f"模型: {model_path}")
    
    # 测试数据集(使用训练集进行测试,看效果)
    base_dir = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/output/AIME24_beam_search/Qwen2.5-Math-7B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/16384_8_2"
    correct_indices = [0, 1, 3, 6, 18]
    
    print(f"测试数据集: {base_dir}")
    print(f"正确题目索引: {correct_indices}")
    
    # 运行测试
    print("\n开始测试...")
    results = test_on_dataset(base_dir, correct_indices, pruner)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    output_file = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 打印剪枝器统计
    print("\n剪枝器统计:")
    stats = pruner.get_stats()
    print(f"  总决策次数: {stats['total_decisions']}")
    print(f"  剪枝次数: {stats['pruned_count']}")
    print(f"  剪枝率: {stats['prune_rate']:.2%}")
    
    print("\n剪枝原因分布:")
    for reason, count in sorted(stats['prune_reasons'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")

if __name__ == '__main__':
    main()
