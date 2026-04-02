#!/usr/bin/env python3
"""
深入分析straggler的注意力熵与选择关系

检查注意力熵是否能够提前指示某个straggler branch是否会被选中
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

def load_straggler_data(json_file: Path) -> Dict[str, Any]:
    """加载straggler分析数据"""
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_entropy_selection_relationship(data: Dict[str, Any]) -> None:
    """分析熵与选择之间的关系"""
    stragglers = data['stragglers']
    
    # 只分析有entropy数据的straggler
    stragglers_with_entropy = [s for s in stragglers if s['has_token_probs']]
    
    print("=" * 80)
    print("ATTENTION ENTROPY vs SELECTION ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. 基本统计
    selected = [s for s in stragglers_with_entropy if s['is_selected']]
    not_selected = [s for s in stragglers_with_entropy if not s['is_selected']]
    
    print(f"Total stragglers with entropy: {len(stragglers_with_entropy)}")
    print(f"  Selected: {len(selected)} ({len(selected)/len(stragglers_with_entropy)*100:.1f}%)")
    print(f"  Not selected: {len(not_selected)} ({len(not_selected)/len(stragglers_with_entropy)*100:.1f}%)")
    print()
    
    # 2. 比较各种熵指标
    metrics = ['mean_entropy', 'std_entropy', 'mean_prob', 'min_prob']
    
    print("Entropy Metrics Comparison:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Selected (μ)':<15} {'Not Selected (μ)':<15} {'Diff':<10} {'Conclusion'}")
    print("-" * 80)
    
    for metric in metrics:
        selected_values = [s['entropy_metrics'][metric] for s in selected]
        not_selected_values = [s['entropy_metrics'][metric] for s in not_selected]
        
        selected_mean = np.mean(selected_values)
        not_selected_mean = np.mean(not_selected_values)
        diff = selected_mean - not_selected_mean
        
        # 判断趋势
        if metric in ['mean_entropy', 'std_entropy']:
            # 熵越高越不确定
            conclusion = "Selected MORE uncertain" if diff > 0.001 else "Selected LESS uncertain" if diff < -0.001 else "No difference"
        else:
            # 概率越高越确定
            conclusion = "Selected MORE confident" if diff > 0.001 else "Selected LESS confident" if diff < -0.001 else "No difference"
        
        print(f"{metric:<20} {selected_mean:<15.4f} {not_selected_mean:<15.4f} {diff:<10.4f} {conclusion}")
    
    print()
    
    # 3. 高/低置信度token比例
    print("Token Confidence Distribution:")
    print("-" * 80)
    
    selected_high_conf = np.mean([s['entropy_metrics']['high_confidence_tokens'] / s['straggler_tokens'] 
                                  for s in selected])
    not_selected_high_conf = np.mean([s['entropy_metrics']['high_confidence_tokens'] / s['straggler_tokens'] 
                                      for s in not_selected])
    
    selected_low_conf = np.mean([s['entropy_metrics']['low_confidence_tokens'] / s['straggler_tokens'] 
                                 for s in selected])
    not_selected_low_conf = np.mean([s['entropy_metrics']['low_confidence_tokens'] / s['straggler_tokens'] 
                                     for s in not_selected])
    
    print(f"High confidence tokens (prob>0.9):")
    print(f"  Selected:     {selected_high_conf*100:.1f}%")
    print(f"  Not selected: {not_selected_high_conf*100:.1f}%")
    print(f"  Difference:   {(selected_high_conf - not_selected_high_conf)*100:.1f}%")
    print()
    
    print(f"Low confidence tokens (prob<0.5):")
    print(f"  Selected:     {selected_low_conf*100:.1f}%")
    print(f"  Not selected: {not_selected_low_conf*100:.1f}%")
    print(f"  Difference:   {(selected_low_conf - not_selected_low_conf)*100:.1f}%")
    print()
    
    # 4. 与reward的关系
    print("Relationship with Reward:")
    print("-" * 80)
    
    selected_rewards = [s['straggler_reward'] for s in selected if s['straggler_reward'] is not None]
    not_selected_rewards = [s['straggler_reward'] for s in not_selected if s['straggler_reward'] is not None]
    
    if selected_rewards and not_selected_rewards:
        print(f"Mean reward:")
        print(f"  Selected:     {np.mean(selected_rewards):.4f}")
        print(f"  Not selected: {np.mean(not_selected_rewards):.4f}")
        print(f"  Difference:   {np.mean(selected_rewards) - np.mean(not_selected_rewards):.4f}")
    print()
    
    # 5. 与token长度的关系
    print("Relationship with Token Length:")
    print("-" * 80)
    
    selected_tokens = [s['straggler_tokens'] for s in selected]
    not_selected_tokens = [s['straggler_tokens'] for s in not_selected]
    
    print(f"Mean tokens:")
    print(f"  Selected:     {np.mean(selected_tokens):.1f}")
    print(f"  Not selected: {np.mean(not_selected_tokens):.1f}")
    print(f"  Difference:   {np.mean(selected_tokens) - np.mean(not_selected_tokens):.1f}")
    print()
    
    # 6. 结论
    print("=" * 80)
    print("CONCLUSIONS:")
    print("=" * 80)
    
    # 检查熵的差异是否显著
    selected_entropies = [s['entropy_metrics']['mean_entropy'] for s in selected]
    not_selected_entropies = [s['entropy_metrics']['mean_entropy'] for s in not_selected]
    
    entropy_diff = np.mean(selected_entropies) - np.mean(not_selected_entropies)
    entropy_diff_pct = entropy_diff / np.mean(not_selected_entropies) * 100
    
    print(f"1. Selected stragglers have {entropy_diff_pct:+.1f}% different mean entropy")
    print(f"   ({'HIGHER' if entropy_diff > 0 else 'LOWER'} = {'LESS' if entropy_diff > 0 else 'MORE'} confident)")
    print()
    
    if abs(entropy_diff_pct) < 5:
        print("2. The entropy difference is SMALL (<5%), suggesting that:")
        print("   - Attention entropy alone may NOT be a strong predictor")
        print("   - Reward score is likely more important for selection")
    elif abs(entropy_diff_pct) < 15:
        print("2. The entropy difference is MODERATE (5-15%), suggesting that:")
        print("   - Attention entropy may provide WEAK signals")
        print("   - Should be combined with other features (reward, length)")
    else:
        print("2. The entropy difference is LARGE (>15%), suggesting that:")
        print("   - Attention entropy is a STRONG indicator")
        print("   - Can be used as an early warning signal for straggler selection")
    print()
    
    # 检查reward差异
    if selected_rewards and not_selected_rewards:
        reward_diff_pct = (np.mean(selected_rewards) - np.mean(not_selected_rewards)) / np.mean(not_selected_rewards) * 100
        print(f"3. Selected stragglers have {reward_diff_pct:+.1f}% different reward")
        if abs(reward_diff_pct) > abs(entropy_diff_pct):
            print("   → Reward is a STRONGER predictor than entropy")
        else:
            print("   → Entropy is a STRONGER predictor than reward")
    
    print()
    print("=" * 80)


def generate_detailed_case_study(data: Dict[str, Any], output_file: Path) -> None:
    """生成详细的案例研究"""
    stragglers = [s for s in data['stragglers'] if s['has_token_probs']]
    
    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED CASE STUDIES: Entropy vs Selection")
    lines.append("=" * 80)
    lines.append("")
    
    # 按entropy排序
    sorted_by_entropy = sorted(stragglers, key=lambda x: x['entropy_metrics']['mean_entropy'])
    
    lines.append("LOWEST ENTROPY STRAGGLERS (Most Confident):")
    lines.append("-" * 80)
    for s in sorted_by_entropy[:5]:
        ent = s['entropy_metrics']
        lines.append(f"{s['question_id']} Step {s['step']} Branch {s['straggler_branch_index']}:")
        lines.append(f"  {'✓ SELECTED' if s['is_selected'] else '✗ NOT SELECTED'}")
        lines.append(f"  Tokens: {s['straggler_tokens']}, Reward: {s['straggler_reward']:.4f}")
        lines.append(f"  Mean entropy: {ent['mean_entropy']:.4f}, Mean prob: {ent['mean_prob']:.4f}")
        lines.append(f"  High conf: {ent['high_confidence_tokens']}/{s['straggler_tokens']} "
                    f"({ent['high_confidence_tokens']/s['straggler_tokens']*100:.1f}%)")
        lines.append("")
    
    lines.append("")
    lines.append("HIGHEST ENTROPY STRAGGLERS (Least Confident):")
    lines.append("-" * 80)
    for s in sorted_by_entropy[-5:]:
        ent = s['entropy_metrics']
        lines.append(f"{s['question_id']} Step {s['step']} Branch {s['straggler_branch_index']}:")
        lines.append(f"  {'✓ SELECTED' if s['is_selected'] else '✗ NOT SELECTED'}")
        lines.append(f"  Tokens: {s['straggler_tokens']}, Reward: {s['straggler_reward']:.4f}")
        lines.append(f"  Mean entropy: {ent['mean_entropy']:.4f}, Mean prob: {ent['mean_prob']:.4f}")
        lines.append(f"  High conf: {ent['high_confidence_tokens']}/{s['straggler_tokens']} "
                    f"({ent['high_confidence_tokens']/s['straggler_tokens']*100:.1f}%)")
        lines.append("")
    
    lines.append("=" * 80)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Detailed case study saved to: {output_file}")


def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / 'straggler_analysis.json'
    case_study_file = script_dir / 'straggler_entropy_case_study.txt'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run identify_stragglers.py first.")
        return
    
    # 加载数据
    data = load_straggler_data(input_file)
    
    # 分析熵与选择关系
    analyze_entropy_selection_relationship(data)
    
    # 生成案例研究
    generate_detailed_case_study(data, case_study_file)


if __name__ == '__main__':
    main()
