#!/usr/bin/env python3
"""
动态Runtime剪枝策略

在每个step实时决策是否剪枝某个branch，基于：
1. 当前branch已生成的token数量
2. 其他branch的长度分布（已完成或当前状态）
3. 当前step索引
4. 已生成token的统计信息（概率、熵等）

关键：不需要知道branch的最终长度，只使用已有信息
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BranchState:
    """Branch的当前状态"""
    branch_id: int
    current_tokens: int  # 已生成的token数
    token_probs: List[float]  # 已生成token的概率列表
    is_finished: bool  # 是否已经完成生成
    

class DynamicStraggler剪枝器:
    """动态straggler剪枝器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置参数
                - base_ratio_threshold: 基础长度比例阈值 (默认2.0)
                - min_tokens_to_check: 开始检查的最小token数 (默认50)
                - step_factor: step因子，越晚越严格 (默认1.0)
                - high_conf_boost: 高置信度时的阈值加成 (默认1.5)
                - early_warning_ratio: 早期警告比例 (默认1.5)
        """
        self.config = config or {}
        self.base_ratio_threshold = self.config.get('base_ratio_threshold', 2.0)
        self.min_tokens_to_check = self.config.get('min_tokens_to_check', 50)
        self.step_factor = self.config.get('step_factor', 1.0)
        self.high_conf_boost = self.config.get('high_conf_boost', 1.5)
        self.early_warning_ratio = self.config.get('early_warning_ratio', 1.5)
        
        # 统计信息
        self.pruned_count = 0
        self.checked_count = 0
    
    def compute_dynamic_threshold(
        self,
        current_step: int,
        other_branches: List[BranchState],
        total_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """计算动态阈值
        
        Args:
            current_step: 当前step索引
            other_branches: 其他branch的状态列表
            total_steps: 总step数（如果已知）
        
        Returns:
            动态阈值字典
        """
        # 1. 计算其他branch的长度统计
        other_lengths = [b.current_tokens for b in other_branches if b.current_tokens > 0]
        
        if not other_lengths:
            # 没有其他branch，使用保守的默认值
            return {
                'max_length_threshold': 200,
                'ratio_threshold': self.base_ratio_threshold,
                'reason': 'no_other_branches'
            }
        
        max_other = max(other_lengths)
        mean_other = np.mean(other_lengths)
        std_other = np.std(other_lengths) if len(other_lengths) > 1 else 0
        
        # 2. 根据step调整阈值（越往后越严格）
        if total_steps and total_steps > 0:
            step_progress = current_step / total_steps
            # 早期(0-30%): 宽松，中期(30-70%): 标准，后期(70-100%): 严格
            if step_progress < 0.3:
                step_multiplier = 1.5  # 早期宽松
            elif step_progress < 0.7:
                step_multiplier = 1.0  # 中期标准
            else:
                step_multiplier = 0.8  # 后期严格
        else:
            # 不知道总step数，使用固定step因子
            step_multiplier = max(1.0 - current_step * 0.02, 0.7)  # 逐渐变严
        
        # 3. 计算动态比例阈值
        base_ratio = self.base_ratio_threshold * step_multiplier
        
        # 如果其他branch长度差异大，提高阈值（更宽容）
        if std_other > mean_other * 0.3:
            ratio_adjustment = 1.2
        else:
            ratio_adjustment = 1.0
        
        dynamic_ratio = base_ratio * ratio_adjustment
        
        # 4. 计算绝对长度阈值
        # 方式1: 基于最大值
        max_length_threshold = max_other * dynamic_ratio
        
        # 方式2: 基于均值+标准差
        statistical_threshold = mean_other + 2 * std_other
        
        # 取较大值（更保守）
        final_length_threshold = max(max_length_threshold, statistical_threshold)
        
        return {
            'max_length_threshold': final_length_threshold,
            'ratio_threshold': dynamic_ratio,
            'max_other_length': max_other,
            'mean_other_length': mean_other,
            'std_other_length': std_other,
            'step_multiplier': step_multiplier,
            'reason': 'computed'
        }
    
    def should_prune_branch(
        self,
        branch: BranchState,
        other_branches: List[BranchState],
        current_step: int,
        total_steps: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """判断是否应该剪枝某个branch
        
        Args:
            branch: 当前branch的状态
            other_branches: 其他branch的状态
            current_step: 当前step
            total_steps: 总step数（可选）
            verbose: 是否输出详细信息
        
        Returns:
            (should_prune, reason, details)
        """
        self.checked_count += 1
        
        # 0. 如果branch已经完成，不剪枝
        if branch.is_finished:
            return False, "already_finished", {}
        
        # 1. 如果token数太少，不检查
        if branch.current_tokens < self.min_tokens_to_check:
            return False, "too_short_to_check", {'current_tokens': branch.current_tokens}
        
        # 2. 如果没有其他branch可比较，不剪枝
        if not other_branches or len(other_branches) == 0:
            return False, "no_comparison", {}
        
        # 3. 计算动态阈值
        thresholds = self.compute_dynamic_threshold(current_step, other_branches, total_steps)
        
        # 4. 计算token统计（如果有概率数据）
        token_stats = self._compute_token_stats(branch.token_probs) if branch.token_probs else None
        
        # 5. 计算当前比例
        current_ratio = branch.current_tokens / thresholds['max_other_length']
        
        # 6. 综合决策
        prune_decision = False
        prune_reason = "within_normal_range"
        
        # 规则1: 检查绝对长度
        if branch.current_tokens > thresholds['max_length_threshold']:
            # 超过绝对阈值，进一步检查比例
            if current_ratio > thresholds['ratio_threshold']:
                # 如果高置信度，给一次机会（提高阈值）
                if token_stats and token_stats['high_conf_ratio'] > 0.8:
                    adjusted_threshold = thresholds['ratio_threshold'] * self.high_conf_boost
                    if current_ratio > adjusted_threshold:
                        prune_decision = True
                        prune_reason = f"exceeds_ratio_even_with_high_conf"
                    else:
                        prune_reason = f"saved_by_high_confidence"
                else:
                    prune_decision = True
                    prune_reason = f"exceeds_ratio_threshold"
            else:
                prune_reason = f"exceeds_length_but_ratio_ok"
        
        # 规则2: 增长趋势检查（如果没有被规则1剪枝）
        if not prune_decision and len(branch.token_probs) > 10:
            recent_growth = self._estimate_growth_rate(branch, other_branches)
            if recent_growth > 3.0:  # 增长率超过3倍
                prune_decision = True
                prune_reason = f"excessive_growth_rate"
        
        # 详细信息
        details = {
            'current_tokens': branch.current_tokens,
            'current_ratio': current_ratio,
            'thresholds': thresholds,
            'token_stats': token_stats,
        }
        
        if prune_decision:
            self.pruned_count += 1
            
        if verbose:
            print(f"Branch {branch.branch_id} at step {current_step}:")
            print(f"  Tokens: {branch.current_tokens} / {thresholds['max_length_threshold']:.0f} (threshold)")
            print(f"  Ratio: {current_ratio:.2f}x / {thresholds['ratio_threshold']:.2f}x (threshold)")
            print(f"  Max other: {thresholds['max_other_length']:.0f}, Mean: {thresholds['mean_other_length']:.1f}")
            if token_stats:
                print(f"  High conf: {token_stats['high_conf_ratio']*100:.1f}%, Mean prob: {token_stats['mean_prob']:.3f}")
            print(f"  Decision: {'🔴 PRUNE' if prune_decision else '✓ KEEP'} ({prune_reason})")
        
        return prune_decision, prune_reason, details
    
    def _compute_token_stats(self, token_probs: List[float]) -> Dict[str, float]:
        """计算token统计信息"""
        if not token_probs:
            return {}
        
        probs = np.array(token_probs)
        
        return {
            'mean_prob': float(np.mean(probs)),
            'min_prob': float(np.min(probs)),
            'max_prob': float(np.max(probs)),
            'std_prob': float(np.std(probs)),
            'high_conf_ratio': float(np.sum(probs > 0.9) / len(probs)),
            'low_conf_ratio': float(np.sum(probs < 0.5) / len(probs)),
            'mean_entropy': float(-np.mean(probs * np.log(probs + 1e-10))),
        }
    
    def _estimate_growth_rate(
        self,
        branch: BranchState,
        other_branches: List[BranchState]
    ) -> float:
        """估计branch的增长率（相对于其他branch）"""
        # 简单估计：当前长度 / 平均其他branch长度
        other_lengths = [b.current_tokens for b in other_branches if b.current_tokens > 0]
        if not other_lengths:
            return 1.0
        
        mean_other = np.mean(other_lengths)
        if mean_other == 0:
            return float('inf')
        
        return branch.current_tokens / mean_other
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            'checked_count': self.checked_count,
            'pruned_count': self.pruned_count,
            'prune_rate': self.pruned_count / self.checked_count if self.checked_count > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.checked_count = 0
        self.pruned_count = 0


def example_usage():
    """使用示例"""
    print("=" * 80)
    print("动态Runtime剪枝示例")
    print("=" * 80)
    print()
    
    # 创建剪枝器
    pruner = DynamicStraggler剪枝器({
        'base_ratio_threshold': 2.0,
        'min_tokens_to_check': 50,
    })
    
    # 模拟beam search场景
    print("场景1: Step 5, 一个branch明显超长但高置信度")
    print("-" * 80)
    
    # 其他branch状态
    other_branches = [
        BranchState(0, 45, [0.95] * 45, False),
        BranchState(1, 52, [0.92] * 52, False),
        BranchState(2, 48, [0.88] * 48, False),
    ]
    
    # 当前检查的branch（超长但高置信度）
    current_branch = BranchState(
        3, 
        120,  # 已经生成了120个token
        [0.98] * 120,  # 高置信度
        False
    )
    
    should_prune, reason, details = pruner.should_prune_branch(
        current_branch,
        other_branches,
        current_step=5,
        total_steps=10,
        verbose=True
    )
    
    print()
    print("场景2: Step 5, 一个branch极度超长且低置信度")
    print("-" * 80)
    
    # 当前检查的branch（极度超长且低置信度）
    current_branch = BranchState(
        3, 
        250,  # 极度超长
        [0.75] * 250,  # 低置信度
        False
    )
    
    should_prune, reason, details = pruner.should_prune_branch(
        current_branch,
        other_branches,
        current_step=5,
        total_steps=10,
        verbose=True
    )
    
    print()
    print("场景3: Step 8, branch长度在合理范围内")
    print("-" * 80)
    
    other_branches = [
        BranchState(0, 150, [0.95] * 150, True),  # 已完成
        BranchState(1, 180, [0.92] * 180, True),  # 已完成
    ]
    
    current_branch = BranchState(
        2,
        200,  # 稍长但还可以
        [0.90] * 200,
        False
    )
    
    should_prune, reason, details = pruner.should_prune_branch(
        current_branch,
        other_branches,
        current_step=8,
        total_steps=10,
        verbose=True
    )
    
    print()
    print("场景4: Step 2, 早期阶段（更宽容）")
    print("-" * 80)
    
    other_branches = [
        BranchState(0, 30, [0.95] * 30, False),
        BranchState(1, 35, [0.92] * 35, False),
    ]
    
    current_branch = BranchState(
        2,
        80,  # 早期就很长
        [0.85] * 80,
        False
    )
    
    should_prune, reason, details = pruner.should_prune_branch(
        current_branch,
        other_branches,
        current_step=2,
        total_steps=10,
        verbose=True
    )
    
    print()
    print("场景5: Step 9, 后期阶段（更严格），branch稍长")
    print("-" * 80)
    
    other_branches = [
        BranchState(0, 300, [0.95] * 300, True),
        BranchState(1, 320, [0.92] * 320, True),
    ]
    
    current_branch = BranchState(
        2,
        500,  # 后期阶段，比其他branch长50%
        [0.88] * 500,
        False
    )
    
    should_prune, reason, details = pruner.should_prune_branch(
        current_branch,
        other_branches,
        current_step=9,
        total_steps=10,
        verbose=True
    )
    
    print()
    print("=" * 80)
    print("统计信息:")
    stats = pruner.get_stats()
    print(f"  检查次数: {stats['checked_count']}")
    print(f"  剪枝次数: {stats['pruned_count']}")
    print(f"  剪枝率: {stats['prune_rate']*100:.1f}%")


if __name__ == '__main__':
    example_usage()
