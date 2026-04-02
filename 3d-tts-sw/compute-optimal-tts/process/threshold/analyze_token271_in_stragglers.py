#!/usr/bin/env python3
"""
分析straggler中token 271 ('\n\n') 的出现情况
对比选中和未选中的straggler
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import Counter

def is_straggler(branch_tokens: List[int], branch_index: int) -> bool:
    """判断某个branch是否为straggler（与identify_stragglers.py保持一致）"""
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


def analyze_token271_in_sequence(branch_topk_logprobs: List[Dict[str, float]]) -> Dict[str, Any]:
    """分析token序列中token 271的出现情况
    
    Args:
        branch_topk_logprobs: branch的topk logprobs列表
    
    Returns:
        包含token 271出现情况的字典
    """
    if not branch_topk_logprobs:
        return {
            'has_271': False,
            'count_271': 0,
            'positions_271': [],
            'avg_prob_271': None,
            'max_prob_271': None,
            'in_top1': 0,
            'in_last_position': False,
        }
    
    positions = []
    probs = []
    top1_count = 0
    
    for pos, token_dict in enumerate(branch_topk_logprobs):
        if '271' in token_dict:
            logprob = token_dict['271']
            prob = math.exp(logprob)
            positions.append(pos)
            probs.append(prob)
            
            # 检查是否是top1
            sorted_tokens = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)
            if sorted_tokens[0][0] == '271':
                top1_count += 1
    
    return {
        'has_271': len(positions) > 0,
        'count_271': len(positions),
        'positions_271': positions,
        'avg_prob_271': float(np.mean(probs)) if probs else None,
        'max_prob_271': float(max(probs)) if probs else None,
        'min_prob_271': float(min(probs)) if probs else None,
        'in_top1': top1_count,
        'in_last_position': (len(positions) > 0 and positions[-1] == len(branch_topk_logprobs) - 1),
        'last_position_is_271_top1': False,  # 后面会更新
    }


def analyze_workload_file(workload_path: Path) -> Dict[str, Any]:
    """分析单个workload文件
    
    Returns:
        包含straggler和非straggler的token 271分析
    """
    results = {
        'stragglers_selected': [],
        'stragglers_not_selected': [],
        'non_stragglers': []
    }
    
    try:
        with open(workload_path, 'r') as f:
            workload = json.load(f)
        
        question_id = workload.get('question_id', 'unknown')
        decode_steps = workload.get('decode', {}).get('steps', [])
        
        for step_info in decode_steps:
            step_num = step_info.get('step', -1)
            branch_count = step_info.get('branch_count', 0)
            branch_tokens = step_info.get('branch_tokens', [])
            selected_branch_index = step_info.get('selected_branch_index', -1)
            branch_token_topk_logprobs = step_info.get('branch_token_topk_logprobs', [])
            
            if not branch_token_topk_logprobs:
                continue
            
            # 分析每个branch
            for branch_idx in range(min(len(branch_tokens), len(branch_token_topk_logprobs))):
                branch_topk = branch_token_topk_logprobs[branch_idx]
                
                if not branch_topk:
                    continue
                
                # 分析token 271
                token271_info = analyze_token271_in_sequence(branch_topk)
                
                # 检查最后位置是否是271且为top1
                if branch_topk:
                    last_token_dict = branch_topk[-1]
                    if '271' in last_token_dict:
                        sorted_tokens = sorted(last_token_dict.items(), key=lambda x: x[1], reverse=True)
                        token271_info['last_position_is_271_top1'] = (sorted_tokens[0][0] == '271')
                
                branch_info = {
                    'question_id': question_id,
                    'step': step_num,
                    'branch_idx': branch_idx,
                    'branch_tokens': branch_tokens[branch_idx],
                    'is_selected': (branch_idx == selected_branch_index),
                    'token271_info': token271_info,
                    'total_tokens_in_topk': len(branch_topk),
                }
                
                # 分类
                if is_straggler(branch_tokens, branch_idx):
                    if branch_idx == selected_branch_index:
                        results['stragglers_selected'].append(branch_info)
                    else:
                        results['stragglers_not_selected'].append(branch_info)
                else:
                    results['non_stragglers'].append(branch_info)
    
    except Exception as e:
        print(f"Error processing {workload_path}: {e}")
    
    return results


def main():
    workload_dir = Path('../wordload/model_workloads/16384_4_1')
    
    print("分析straggler中token 271 ('\\n\\n') 的出现情况")
    print("="*80)
    
    all_stragglers_selected = []
    all_stragglers_not_selected = []
    all_non_stragglers = []
    
    # 处理所有文件
    workload_files = sorted(workload_dir.glob('question_*_workload.json'))
    
    for idx, workload_file in enumerate(workload_files):
        print(f"处理 {idx+1}/{len(workload_files)}: {workload_file.name}")
        results = analyze_workload_file(workload_file)
        all_stragglers_selected.extend(results['stragglers_selected'])
        all_stragglers_not_selected.extend(results['stragglers_not_selected'])
        all_non_stragglers.extend(results['non_stragglers'])
    
    print("\n" + "="*80)
    print("分析结果")
    print("="*80)
    
    # 统计函数
    def print_stats(branches: List[Dict], title: str):
        if not branches:
            print(f"\n{title}: 无数据")
            return
        
        print(f"\n{title} (n={len(branches)}):")
        print("-"*80)
        
        # Token 271出现率
        has_271 = sum(1 for b in branches if b['token271_info']['has_271'])
        print(f"  包含token 271: {has_271}/{len(branches)} ({has_271/len(branches)*100:.1f}%)")
        
        # 在有271的branch中的统计
        branches_with_271 = [b for b in branches if b['token271_info']['has_271']]
        
        if branches_with_271:
            # 出现次数
            counts = [b['token271_info']['count_271'] for b in branches_with_271]
            print(f"  出现次数: min={min(counts)}, max={max(counts)}, "
                  f"avg={np.mean(counts):.1f}, median={np.median(counts):.1f}")
            
            # 平均概率
            avg_probs = [b['token271_info']['avg_prob_271'] for b in branches_with_271 
                         if b['token271_info']['avg_prob_271'] is not None]
            if avg_probs:
                print(f"  平均概率: min={min(avg_probs):.4f}, max={max(avg_probs):.4f}, "
                      f"avg={np.mean(avg_probs):.4f}")
            
            # Top1次数
            top1_counts = [b['token271_info']['in_top1'] for b in branches_with_271]
            total_top1 = sum(top1_counts)
            print(f"  作为top-1的次数: total={total_top1}, "
                  f"avg_per_branch={np.mean(top1_counts):.1f}")
            
            # 最后位置
            in_last = sum(1 for b in branches_with_271 if b['token271_info']['in_last_position'])
            print(f"  出现在最后位置: {in_last}/{len(branches_with_271)} ({in_last/len(branches_with_271)*100:.1f}%)")
            
            # 最后位置且为top1
            last_top1 = sum(1 for b in branches_with_271 if b['token271_info']['last_position_is_271_top1'])
            if in_last > 0:
                print(f"  最后位置且为top-1: {last_top1}/{in_last} ({last_top1/in_last*100:.1f}%)")
            
            # Token分布率（相对于总token数）
            ratios = [b['token271_info']['count_271'] / b['branch_tokens'] 
                     for b in branches_with_271 if b['branch_tokens'] > 0]
            if ratios:
                print(f"  Token 271占比: min={min(ratios)*100:.1f}%, max={max(ratios)*100:.1f}%, "
                      f"avg={np.mean(ratios)*100:.1f}%")
    
    # 输出各组统计
    print_stats(all_stragglers_selected, "【选中的Straggler】")
    print_stats(all_stragglers_not_selected, "【未选中的Straggler】")
    print_stats(all_non_stragglers, "【非Straggler】")
    
    # 对比分析
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    
    # 计算出现率差异
    if all_stragglers_selected and all_stragglers_not_selected:
        rate_selected = sum(1 for b in all_stragglers_selected if b['token271_info']['has_271']) / len(all_stragglers_selected)
        rate_not_selected = sum(1 for b in all_stragglers_not_selected if b['token271_info']['has_271']) / len(all_stragglers_not_selected)
        
        print(f"\n选中的straggler中token 271出现率: {rate_selected*100:.1f}%")
        print(f"未选中的straggler中token 271出现率: {rate_not_selected*100:.1f}%")
        print(f"差异: {(rate_selected - rate_not_selected)*100:.1f}个百分点")
        
        if rate_selected > rate_not_selected:
            print("→ 选中的straggler更频繁包含token 271 (step分隔符)")
        else:
            print("→ 未选中的straggler更频繁包含token 271 (step分隔符)")
    
    # 对比straggler vs 非straggler
    if all_stragglers_selected and all_non_stragglers:
        rate_straggler = sum(1 for b in all_stragglers_selected if b['token271_info']['has_271']) / len(all_stragglers_selected)
        rate_normal = sum(1 for b in all_non_stragglers if b['token271_info']['has_271']) / len(all_non_stragglers)
        
        print(f"\nStraggler中token 271出现率: {rate_straggler*100:.1f}%")
        print(f"非Straggler中token 271出现率: {rate_normal*100:.1f}%")
        print(f"差异: {(rate_straggler - rate_normal)*100:.1f}个百分点")
        
        if rate_straggler > rate_normal:
            print("→ Straggler更频繁包含token 271")
        else:
            print("→ 非Straggler更频繁包含token 271")
    
    # 保存详细结果
    output_file = Path(__file__).parent / 'token271_analysis.json'
    output_data = {
        'summary': {
            'total_stragglers_selected': len(all_stragglers_selected),
            'total_stragglers_not_selected': len(all_stragglers_not_selected),
            'total_non_stragglers': len(all_non_stragglers),
        },
        'stragglers_selected': all_stragglers_selected[:100],  # 只保存前100个
        'stragglers_not_selected': all_stragglers_not_selected[:100],
        'non_stragglers': all_non_stragglers[:100],
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
