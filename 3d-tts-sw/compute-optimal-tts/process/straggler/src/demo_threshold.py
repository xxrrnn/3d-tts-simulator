#!/usr/bin/env python3
"""
测试动态threshold功能
演示：在不同配置和branch数下，如何动态判断一个分支是否为straggler
"""

import json
import numpy as np


# 模拟训练得到的参数（示例）
EXAMPLE_PARAMS = {
    # 全局默认参数
    "global": {
        "base_ratio": 3.2,
        "base_length": 110,
        "step_factor": 0.02,
        "branch_factor": -0.05,
        "std_factor": 0.15,
        "avg_factor": 0.08,
        "history_factor": 0.03
    },
    # 配置级参数
    "config_256_8_1": {
        "base_ratio": 3.5,
        "base_length": 100,
        "step_factor": 0.03,
        "branch_factor": -0.08,
        "std_factor": 0.18,
        "avg_factor": 0.10,
        "history_factor": 0.04
    },
    "config_16384_4_1": {
        "base_ratio": 2.8,
        "base_length": 120,
        "step_factor": 0.01,
        "branch_factor": -0.03,
        "std_factor": 0.12,
        "avg_factor": 0.06,
        "history_factor": 0.02
    },
    # Branch级参数
    "branch_2": {
        "base_ratio": 3.8,
        "base_length": 130,
        "step_factor": 0.025,
        "branch_factor": -0.10,
        "std_factor": 0.20,
        "avg_factor": 0.12,
        "history_factor": 0.05
    },
    "branch_4": {
        "base_ratio": 3.0,
        "base_length": 105,
        "step_factor": 0.02,
        "branch_factor": -0.06,
        "std_factor": 0.15,
        "avg_factor": 0.08,
        "history_factor": 0.03
    }
}


def calculate_dynamic_threshold(params, step, branch_count, branch_tokens, history_max=0):
    """
    计算动态threshold
    
    这就是您需要的功能！根据：
    - step: 当前步骤
    - branch_count: 分支数
    - branch_tokens: 各分支的token数
    - history_max: 历史最大token数
    
    返回动态的ratio_threshold和length_threshold
    """
    avg_token = np.mean(branch_tokens)
    std_token = np.std(branch_tokens) if len(branch_tokens) > 1 else 0
    
    # 动态ratio阈值（比值要求）
    ratio_threshold = params['base_ratio']
    ratio_threshold += params['step_factor'] * step
    ratio_threshold += params['branch_factor'] * branch_count
    if avg_token > 0:
        ratio_threshold += params['std_factor'] * std_token / avg_token
    
    # 动态length阈值（长度要求）
    length_threshold = params['base_length']
    length_threshold += params['avg_factor'] * avg_token
    length_threshold += params['history_factor'] * history_max
    
    # 限制在合理范围
    ratio_threshold = max(2.0, min(5.0, ratio_threshold))
    length_threshold = max(50, min(300, length_threshold))
    
    return ratio_threshold, length_threshold


def predict_straggler(params, step, branch_count, branch_tokens, history_max=0):
    """
    预测是否有straggler
    
    返回: (is_straggler, straggler_indices, thresholds)
    """
    ratio_th, length_th = calculate_dynamic_threshold(
        params, step, branch_count, branch_tokens, history_max
    )
    
    straggler_indices = []
    for i, token in enumerate(branch_tokens):
        if token <= length_threshold:
            continue
        
        # 找除自己外的最大值
        other_tokens = branch_tokens[:i] + branch_tokens[i+1:]
        if not other_tokens:
            continue
        
        max_other = max(other_tokens)
        if max_other > 0 and token >= max_other * ratio_th:
            straggler_indices.append(i)
    
    return len(straggler_indices) > 0, straggler_indices, {
        'ratio_threshold': ratio_th,
        'length_threshold': length_th
    }


def demo():
    """演示动态threshold功能"""
    print("="*80)
    print("动态Threshold演示 - 这就是您需要的功能！")
    print("="*80)
    print()
    
    # 场景1: Branch=4, 配置256_8_1, step=5
    print("场景1: 配置=256_8_1, Branch=4, Step=5")
    print("-" * 60)
    branch_tokens = [120, 130, 650, 125]
    params = EXAMPLE_PARAMS["config_256_8_1"]
    
    is_strag, indices, thresholds = predict_straggler(
        params, step=5, branch_count=4, branch_tokens=branch_tokens
    )
    
    print(f"  分支tokens: {branch_tokens}")
    print(f"  动态ratio阈值: {thresholds['ratio_threshold']:.2f}")
    print(f"  动态length阈值: {thresholds['length_threshold']:.1f}")
    print(f"  → 检测到straggler: {is_strag}")
    if is_strag:
        print(f"  → Straggler分支: {indices}")
        for idx in indices:
            other_max = max([t for i, t in enumerate(branch_tokens) if i != idx])
            print(f"     分支{idx}: {branch_tokens[idx]} > {other_max} × {thresholds['ratio_threshold']:.2f} = {other_max * thresholds['ratio_threshold']:.1f} ✓")
    print()
    
    # 场景2: 同样的tokens，但step=20 (threshold会变化!)
    print("场景2: 同样的分支，但step=20 (threshold动态变化!)")
    print("-" * 60)
    is_strag, indices, thresholds = predict_straggler(
        params, step=20, branch_count=4, branch_tokens=branch_tokens
    )
    
    print(f"  分支tokens: {branch_tokens}")
    print(f"  动态ratio阈值: {thresholds['ratio_threshold']:.2f} (注意变化!)")
    print(f"  动态length阈值: {thresholds['length_threshold']:.1f}")
    print(f"  → 检测到straggler: {is_strag}")
    print()
    
    # 场景3: Branch=2, 配置16384_4_1
    print("场景3: 配置=16384_4_1, Branch=2, Step=3")
    print("-" * 60)
    branch_tokens = [120, 450]
    params = EXAMPLE_PARAMS["config_16384_4_1"]
    
    is_strag, indices, thresholds = predict_straggler(
        params, step=3, branch_count=2, branch_tokens=branch_tokens
    )
    
    print(f"  分支tokens: {branch_tokens}")
    print(f"  动态ratio阈值: {thresholds['ratio_threshold']:.2f}")
    print(f"  动态length阈值: {thresholds['length_threshold']:.1f}")
    print(f"  → 检测到straggler: {is_strag}")
    if is_strag:
        print(f"  → Straggler分支: {indices}")
    print()
    
    # 场景4: 展示不同配置的区别
    print("场景4: 相同场景，不同配置的threshold对比")
    print("-" * 60)
    branch_tokens = [100, 110, 400]
    
    for config_name in ["global", "config_256_8_1", "config_16384_4_1", "branch_4"]:
        params = EXAMPLE_PARAMS[config_name]
        _, _, thresholds = predict_straggler(
            params, step=5, branch_count=3, branch_tokens=branch_tokens
        )
        print(f"  {config_name:20s}: ratio={thresholds['ratio_threshold']:.2f}, "
              f"length={thresholds['length_threshold']:.1f}")
    print()
    
    # 关键总结
    print("="*80)
    print("关键功能总结:")
    print("="*80)
    print("1. ✅ 动态threshold: 根据step/branch/tokens动态计算")
    print("2. ✅ 配置级参数: 不同配置(256_8_1等)有不同的threshold")
    print("3. ✅ Branch级参数: 不同branch数有不同的threshold")
    print("4. ✅ 自适应: threshold随着step增加而变化")
    print("5. ✅ 精确: 考虑tokens的统计特征(均值、标准差)")
    print()
    print("使用方式:")
    print("  params = 加载训练好的参数(按config/branch选择)")
    print("  ratio_th, length_th = calculate_dynamic_threshold(params, step, branch_count, tokens)")
    print("  → 判断: 如果 token > length_th 且 token > other_max * ratio_th")
    print("  → 则该分支是straggler")
    print("="*80)


if __name__ == "__main__":
    demo()
