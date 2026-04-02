#!/usr/bin/env python3
"""
分析RUNTIME时可用的指标来预测straggler是否会被选中

Runtime可用的指标:
- Token数量
- Token长度比例
- Token概率（注意力熵）

Runtime不可用的指标:
- Reward (还没计算)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_straggler_data(json_file: Path) -> Dict[str, Any]:
    """加载straggler分析数据"""
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_runtime_features(data: Dict[str, Any]) -> None:
    """分析runtime时可用的特征"""
    stragglers = [s for s in data['stragglers'] if s['has_token_probs']]
    
    selected = [s for s in stragglers if s['is_selected']]
    not_selected = [s for s in stragglers if not s['is_selected']]
    
    print("=" * 80)
    print("RUNTIME FEATURE ANALYSIS (No Reward Available)")
    print("=" * 80)
    print()
    
    print(f"Total stragglers: {len(stragglers)}")
    print(f"  Selected: {len(selected)} ({len(selected)/len(stragglers)*100:.1f}%)")
    print(f"  Not selected: {len(not_selected)} ({len(not_selected)/len(stragglers)*100:.1f}%)")
    print()
    
    # 1. Token数量特征
    print("1. TOKEN LENGTH FEATURES")
    print("-" * 80)
    
    selected_tokens = [s['straggler_tokens'] for s in selected]
    not_selected_tokens = [s['straggler_tokens'] for s in not_selected]
    
    print(f"Absolute token count:")
    print(f"  Selected:     {np.mean(selected_tokens):.1f} ± {np.std(selected_tokens):.1f}")
    print(f"  Not selected: {np.mean(not_selected_tokens):.1f} ± {np.std(not_selected_tokens):.1f}")
    
    diff = np.mean(selected_tokens) - np.mean(not_selected_tokens)
    diff_pct = diff / np.mean(not_selected_tokens) * 100
    print(f"  Difference: {diff:.1f} tokens ({diff_pct:+.1f}%)")
    print()
    
    selected_ratios = [s['stats']['ratio_to_max_other'] for s in selected]
    not_selected_ratios = [s['stats']['ratio_to_max_other'] for s in not_selected]
    
    print(f"Length ratio (vs max other branch):")
    print(f"  Selected:     {np.mean(selected_ratios):.2f}x ± {np.std(selected_ratios):.2f}x")
    print(f"  Not selected: {np.mean(not_selected_ratios):.2f}x ± {np.std(not_selected_ratios):.2f}x")
    
    ratio_diff = np.mean(selected_ratios) - np.mean(not_selected_ratios)
    ratio_diff_pct = ratio_diff / np.mean(not_selected_ratios) * 100
    print(f"  Difference: {ratio_diff:.2f}x ({ratio_diff_pct:+.1f}%)")
    print()
    
    # 2. 注意力熵特征
    print("2. ATTENTION ENTROPY FEATURES")
    print("-" * 80)
    
    metrics = {
        'mean_entropy': 'Mean entropy',
        'std_entropy': 'Std entropy',
        'mean_prob': 'Mean probability',
        'min_prob': 'Min probability',
    }
    
    best_predictor = None
    best_diff_pct = 0
    
    for key, name in metrics.items():
        selected_vals = [s['entropy_metrics'][key] for s in selected]
        not_selected_vals = [s['entropy_metrics'][key] for s in not_selected]
        
        sel_mean = np.mean(selected_vals)
        notsel_mean = np.mean(not_selected_vals)
        diff = sel_mean - notsel_mean
        diff_pct = abs(diff / notsel_mean * 100) if notsel_mean != 0 else 0
        
        print(f"{name}:")
        print(f"  Selected:     {sel_mean:.4f}")
        print(f"  Not selected: {notsel_mean:.4f}")
        print(f"  Difference: {diff:+.4f} ({diff_pct:+.1f}%)")
        
        if diff_pct > best_diff_pct:
            best_diff_pct = diff_pct
            best_predictor = (key, name, diff_pct)
        
        print()
    
    # 3. Token置信度分布
    print("3. TOKEN CONFIDENCE DISTRIBUTION")
    print("-" * 80)
    
    selected_high = [s['entropy_metrics']['high_confidence_tokens'] / s['straggler_tokens'] 
                     for s in selected]
    not_selected_high = [s['entropy_metrics']['high_confidence_tokens'] / s['straggler_tokens'] 
                         for s in not_selected]
    
    high_conf_diff = (np.mean(selected_high) - np.mean(not_selected_high)) * 100
    high_conf_diff_pct = high_conf_diff / (np.mean(not_selected_high) * 100) * 100
    
    print(f"High confidence tokens (prob > 0.9):")
    print(f"  Selected:     {np.mean(selected_high)*100:.1f}%")
    print(f"  Not selected: {np.mean(not_selected_high)*100:.1f}%")
    print(f"  Difference: {high_conf_diff:+.1f}% ({high_conf_diff_pct:+.1f}%)")
    print()
    
    selected_low = [s['entropy_metrics']['low_confidence_tokens'] / s['straggler_tokens'] 
                    for s in selected]
    not_selected_low = [s['entropy_metrics']['low_confidence_tokens'] / s['straggler_tokens'] 
                        for s in not_selected]
    
    low_conf_diff = (np.mean(selected_low) - np.mean(not_selected_low)) * 100
    
    print(f"Low confidence tokens (prob < 0.5):")
    print(f"  Selected:     {np.mean(selected_low)*100:.1f}%")
    print(f"  Not selected: {np.mean(not_selected_low)*100:.1f}%")
    print(f"  Difference: {low_conf_diff:+.1f}%")
    print()
    
    # 4. 组合特征分析
    print("4. FEATURE IMPORTANCE RANKING")
    print("-" * 80)
    
    features = [
        ('Token length (absolute)', abs(diff_pct)),
        ('Token ratio', abs(ratio_diff_pct)),
        ('High conf % (runtime)', abs(high_conf_diff_pct)),
        (best_predictor[1] if best_predictor else 'N/A', best_diff_pct),
    ]
    
    features.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(features, 1):
        print(f"{i}. {feature}: {importance:.1f}% difference")
    
    print()
    
    # 5. 实际应用建议
    print("=" * 80)
    print("RUNTIME PRUNING STRATEGY (Without Reward)")
    print("=" * 80)
    print()
    
    # 找阈值
    print("SUGGESTED THRESHOLDS:")
    print("-" * 80)
    
    # Token length threshold
    token_threshold = np.percentile(selected_tokens, 75)  # 75%分位数
    print(f"1. Token length threshold: {token_threshold:.0f} tokens")
    print(f"   If tokens > {token_threshold:.0f}: Likely NOT selected")
    false_positive = sum(1 for t in selected_tokens if t > token_threshold)
    print(f"   False positives: {false_positive}/{len(selected)} ({false_positive/len(selected)*100:.1f}%)")
    print()
    
    # Ratio threshold
    ratio_threshold = np.percentile(selected_ratios, 75)
    print(f"2. Length ratio threshold: {ratio_threshold:.1f}x")
    print(f"   If ratio > {ratio_threshold:.1f}x: Likely NOT selected")
    false_positive = sum(1 for r in selected_ratios if r > ratio_threshold)
    print(f"   False positives: {false_positive}/{len(selected)} ({false_positive/len(selected)*100:.1f}%)")
    print()
    
    # High confidence threshold
    high_conf_threshold = np.percentile(selected_high, 25)  # 25%分位数 (selected通常更低)
    print(f"3. High confidence threshold: {high_conf_threshold*100:.1f}%")
    print(f"   If high_conf > {high_conf_threshold*100:.1f}%: Might NOT be selected")
    false_positive = sum(1 for h in selected_high if h > high_conf_threshold)
    print(f"   False positives: {false_positive}/{len(selected)} ({false_positive/len(selected)*100:.1f}%)")
    print()
    
    # 组合规则
    print("RECOMMENDED COMBINED RULE:")
    print("-" * 80)
    print(f"Prune if:")
    print(f"  (tokens > {token_threshold:.0f} AND ratio > {ratio_threshold:.1f}x)")
    print(f"  OR")
    print(f"  (tokens > {np.mean(selected_tokens)+np.std(selected_tokens):.0f} AND high_conf > {high_conf_threshold*100:.0f}%)")
    print()
    
    # 验证组合规则
    rule1_fn = 0  # False negatives (误剪枝被选中的)
    rule1_tp = 0  # True positives (正确剪枝未被选中的)
    
    for s in selected:
        if (s['straggler_tokens'] > token_threshold and 
            s['stats']['ratio_to_max_other'] > ratio_threshold):
            rule1_fn += 1
        elif (s['straggler_tokens'] > np.mean(selected_tokens)+np.std(selected_tokens) and
              s['entropy_metrics']['high_confidence_tokens']/s['straggler_tokens'] > high_conf_threshold):
            rule1_fn += 1
    
    for s in not_selected:
        if (s['straggler_tokens'] > token_threshold and 
            s['stats']['ratio_to_max_other'] > ratio_threshold):
            rule1_tp += 1
        elif (s['straggler_tokens'] > np.mean(selected_tokens)+np.std(selected_tokens) and
              s['entropy_metrics']['high_confidence_tokens']/s['straggler_tokens'] > high_conf_threshold):
            rule1_tp += 1
    
    print(f"Performance:")
    print(f"  Would prune: {rule1_tp}/{len(not_selected)} not-selected stragglers ({rule1_tp/len(not_selected)*100:.1f}%)")
    print(f"  False prune: {rule1_fn}/{len(selected)} selected stragglers ({rule1_fn/len(selected)*100:.1f}%)")
    print(f"  Precision: {rule1_tp/(rule1_tp+rule1_fn)*100:.1f}%" if (rule1_tp+rule1_fn) > 0 else "  Precision: N/A")
    print()
    
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    
    if abs(diff_pct) > 20:
        print(f"✓ Token length is a STRONG indicator ({abs(diff_pct):.0f}% difference)")
        print(f"  → Selected stragglers are {'SHORTER' if diff < 0 else 'LONGER'}")
    else:
        print(f"⚠ Token length is a WEAK indicator ({abs(diff_pct):.0f}% difference)")
    
    print()
    
    if abs(high_conf_diff_pct) > 20:
        print(f"✓ High confidence % is a STRONG indicator ({abs(high_conf_diff_pct):.0f}% difference)")
        print(f"  → Selected stragglers have {'FEWER' if high_conf_diff < 0 else 'MORE'} high-conf tokens")
    else:
        print(f"⚠ High confidence % is a WEAK indicator ({abs(high_conf_diff_pct):.0f}% difference)")
    
    print()
    print("CONCLUSION:")
    if abs(diff_pct) > 30 or abs(high_conf_diff_pct) > 30:
        print("→ Runtime features CAN provide useful signals for pruning!")
    else:
        print("→ Runtime features provide WEAK signals. Pruning is risky without reward.")
    
    print("=" * 80)


def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / 'straggler_analysis.json'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run identify_stragglers.py first.")
        return
    
    data = load_straggler_data(input_file)
    analyze_runtime_features(data)


if __name__ == '__main__':
    main()
