#!/usr/bin/env python3
"""
动态分支剪枝器 - 运行时使用

在beam search过程中动态决定是否剪枝某个分支
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple

class DynamicPruner:
    """动态剪枝器
    
    使用学习到的threshold策略,在运行时决定是否剪枝分支
    """
    
    def __init__(self, model_path: str):
        """初始化剪枝器
        
        Args:
            model_path: threshold模型文件路径
        """
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        
        self.strategy = self.model['adaptive_strategy']
        self.decision_rules = self.model['decision_rules']
        self.score_thresholds = self.model['score_thresholds']
        
        # 统计信息
        self.stats = {
            'total_decisions': 0,
            'pruned_count': 0,
            'kept_count': 0,
            'prune_reasons': {}
        }
    
    def _compute_stage(self, current_step: int, max_steps: Optional[int] = None) -> str:
        """计算当前所处阶段
        
        Args:
            current_step: 当前步数
            max_steps: 预估最大步数(如果不知道,使用启发式)
        
        Returns:
            'early', 'mid', or 'late'
        """
        if max_steps is None:
            # 启发式:假设最多100步
            max_steps = 100
        
        progress = current_step / max_steps
        
        if progress < 0.3:
            return 'early'
        elif progress < 0.7:
            return 'mid'
        else:
            return 'late'
    
    def _extract_runtime_features(self, 
                                   branch_data: Dict,
                                   all_branches: List[Dict],
                                   current_step: int) -> Dict:
        """从运行时数据中提取特征
        
        **重要**: 只使用历史数据(不包括当前step),因为当前step的reward还未计算
        
        Args:
            branch_data: 当前分支的数据
                - reward_history: List[float] - 历史reward(不包括当前step)
                - token_prob_history: List[List[float]] - 历史token概率
                - prob_history: List[float] - 历史累积概率
            all_branches: 所有分支的数据
            current_step: 当前步数(即将生成的步数,此时该步的reward未知)
        
        Returns:
            特征字典
        """
        features = {}
        
        # Reward特征 - 只使用历史reward(已经计算出的)
        # 注意: reward_history应该只包含已完成步骤的reward,不包括当前步
        if 'reward_history' in branch_data and len(branch_data['reward_history']) > 0:
            rewards = branch_data['reward_history']
            # rewards[-1]是最近一步已完成的reward,不是当前步
            features['reward_last'] = rewards[-1] if rewards else 0.0
            features['reward_mean'] = np.mean(rewards) if rewards else 0.0
            features['reward_max'] = np.max(rewards) if rewards else 0.0
            features['reward_min'] = np.min(rewards) if rewards else 0.0
            
            if len(rewards) > 1:
                features['reward_trend'] = rewards[-1] - rewards[0]
                features['reward_volatility'] = np.std([rewards[i+1] - rewards[i] for i in range(len(rewards)-1)])
            else:
                features['reward_trend'] = 0.0
                features['reward_volatility'] = 0.0
        else:
            features['reward_last'] = 0.0
            features['reward_mean'] = 0.0
            features['reward_max'] = 0.0
            features['reward_min'] = 0.0
            features['reward_trend'] = 0.0
            features['reward_volatility'] = 0.0
        
        # Token概率特征 - 使用历史最后一步的token概率
        if 'token_prob_history' in branch_data and len(branch_data['token_prob_history']) > 0:
            last_step_probs = branch_data['token_prob_history'][-1]
            if isinstance(last_step_probs, list) and len(last_step_probs) > 0:
                features['token_prob_mean'] = np.mean(last_step_probs)
                features['token_prob_min'] = np.min(last_step_probs)
            else:
                features['token_prob_mean'] = 0.5
                features['token_prob_min'] = 0.5
        else:
            features['token_prob_mean'] = 0.5
            features['token_prob_min'] = 0.5
        
        # 累积概率 - 使用历史累积概率
        if 'prob_history' in branch_data and len(branch_data['prob_history']) > 0:
            features['cum_prob'] = branch_data['prob_history'][-1]
        else:
            features['cum_prob'] = 0.5
        
        # 相对特征 - 基于所有分支的历史最后reward
        all_last_rewards = []
        for branch in all_branches:
            if 'reward_history' in branch and len(branch['reward_history']) > 0:
                all_last_rewards.append(branch['reward_history'][-1])
        
        if all_last_rewards:
            features['max_reward'] = max(all_last_rewards)
            features['mean_reward'] = np.mean(all_last_rewards)
            
            # 计算排名(基于最后一步的reward)
            sorted_rewards = sorted(all_last_rewards, reverse=True)
            current_reward = features['reward_last']
            if current_reward in sorted_rewards:
                rank_idx = sorted_rewards.index(current_reward)
                features['reward_rank'] = rank_idx / max(len(all_last_rewards) - 1, 1)
            else:
                features['reward_rank'] = 1.0
        else:
            features['max_reward'] = 0.0
            features['mean_reward'] = 0.0
            features['reward_rank'] = 0.5
        
        features['num_branches'] = len(all_branches)
        features['current_step'] = current_step
        features['history_length'] = len(branch_data.get('reward_history', []))
        
        return features
    
    def _compute_weighted_score(self, features: Dict) -> float:
        """计算综合得分
        
        Args:
            features: 特征字典
        
        Returns:
            加权得分
        """
        weights = self.strategy['feature_weights']
        
        score = 0.0
        total_weight = 0.0
        
        # 使用reward_last代替reward_current
        feature_mapping = {
            'reward_current': 'reward_last',  # 映射到历史最后reward
            'reward_mean': 'reward_mean',
            'token_prob_mean': 'token_prob_mean',
            'cum_prob': 'cum_prob',
            'reward_trend': 'reward_trend'
        }
        
        for weight_key, weight in weights.items():
            feat_key = feature_mapping.get(weight_key, weight_key)
            if feat_key in features:
                score += weight * features[feat_key]
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def should_prune_branch(self,
                           branch_data: Dict,
                           all_branches: List[Dict],
                           current_step: int,
                           max_steps: Optional[int] = None,
                           verbose: bool = False) -> Tuple[bool, str]:
        """判断是否应该剪枝某个分支
        
        Args:
            branch_data: 当前分支的数据
            all_branches: 所有分支的数据
            current_step: 当前步数
            max_steps: 预估最大步数
            verbose: 是否打印调试信息
        
        Returns:
            (should_prune, reason)
        """
        self.stats['total_decisions'] += 1
        
        # 提取特征
        features = self._extract_runtime_features(branch_data, all_branches, current_step)
        
        # 获取阈值
        stage = self._compute_stage(current_step, max_steps)
        reward_relative_thresholds = self.strategy['base_thresholds']['reward_relative']
        reward_absolute = self.strategy['base_thresholds']['reward_absolute']
        rank_threshold = self.strategy['base_thresholds']['rank_threshold']
        
        # 决策逻辑
        
        # 规则1: 绝对reward太低,直接剪枝 (使用历史最后reward)
        if features['reward_last'] < reward_absolute['min_threshold']:
            reason = f"low_absolute_reward (last_reward={features['reward_last']:.4f} < {reward_absolute['min_threshold']:.4f})"
            self.stats['pruned_count'] += 1
            self.stats['prune_reasons'][reason] = self.stats['prune_reasons'].get(reason, 0) + 1
            if verbose:
                print(f"  [PRUNE] {reason}")
            return True, reason
        
        # 规则2: 绝对reward很高,一定保留
        if features['reward_last'] > reward_absolute['safe_threshold']:
            reason = f"high_absolute_reward (last_reward={features['reward_last']:.4f} > {reward_absolute['safe_threshold']:.4f})"
            self.stats['kept_count'] += 1
            if verbose:
                print(f"  [KEEP] {reason}")
            return False, reason
        
        # 规则3: 基于相对reward和排名
        stage_key = f"{stage}_stage"
        relative_threshold = reward_relative_thresholds.get(stage_key, 0.7)
        
        if (features['reward_rank'] > rank_threshold['max_rank'] and 
            features['reward_last'] < features['max_reward'] * relative_threshold):
            reason = f"low_relative_reward (rank={features['reward_rank']:.2f}, last_reward={features['reward_last']:.4f} < {features['max_reward']:.4f}*{relative_threshold})"
            self.stats['pruned_count'] += 1
            self.stats['prune_reasons'][reason] = self.stats['prune_reasons'].get(reason, 0) + 1
            if verbose:
                print(f"  [PRUNE] {reason}")
            return True, reason
        
        # 规则4: 基于综合得分
        weighted_score = self._compute_weighted_score(features)
        score_threshold = self.score_thresholds['score_threshold']
        
        if weighted_score < score_threshold:
            reason = f"low_weighted_score (score={weighted_score:.4f} < {score_threshold:.4f})"
            self.stats['pruned_count'] += 1
            self.stats['prune_reasons'][reason] = self.stats['prune_reasons'].get(reason, 0) + 1
            if verbose:
                print(f"  [PRUNE] {reason}")
            return True, reason
        
        # 默认保留
        reason = "passed_all_checks"
        self.stats['kept_count'] += 1
        if verbose:
            print(f"  [KEEP] {reason} (score={weighted_score:.4f}, last_reward={features['reward_last']:.4f}, history_len={features['history_length']})")
        return False, reason
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_decisions'] > 0:
            stats['prune_rate'] = stats['pruned_count'] / stats['total_decisions']
        else:
            stats['prune_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_decisions': 0,
            'pruned_count': 0,
            'kept_count': 0,
            'prune_reasons': {}
        }


# 示例使用
if __name__ == '__main__':
    # 创建剪枝器
    model_path = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/threshold_model.json"
    pruner = DynamicPruner(model_path)
    
    print("动态剪枝器已初始化")
    print(f"模型版本: {pruner.model['metadata']['version']}")
    print(f"训练数据: {pruner.model['metadata']['training_data']}")
    
    # 模拟示例
    print("\n" + "="*80)
    print("示例: 模拟剪枝决策")
    print("="*80)
    
    # 分支1: 高reward (正确分支)
    branch1 = {
        'reward_history': [0.5, 0.6, 0.7, 0.8],
        'token_prob_history': [
            [0.9] * 100,
            [0.95] * 100,
            [0.96] * 100,
            [0.97] * 100
        ],
        'prob_history': [0.9, 0.92, 0.94, 0.96]
    }
    
    # 分支2: 低reward (错误分支)
    branch2 = {
        'reward_history': [0.2, 0.25, 0.22, 0.24],
        'token_prob_history': [
            [0.7] * 100,
            [0.75] * 100,
            [0.72] * 100,
            [0.74] * 100
        ],
        'prob_history': [0.7, 0.72, 0.71, 0.73]
    }
    
    all_branches = [branch1, branch2]
    
    print("\n测试分支1 (高reward):")
    should_prune1, reason1 = pruner.should_prune_branch(branch1, all_branches, current_step=3, verbose=True)
    print(f"结果: {'剪枝' if should_prune1 else '保留'}")
    
    print("\n测试分支2 (低reward):")
    should_prune2, reason2 = pruner.should_prune_branch(branch2, all_branches, current_step=3, verbose=True)
    print(f"结果: {'剪枝' if should_prune2 else '保留'}")
    
    print("\n统计信息:")
    stats = pruner.get_stats()
    print(f"总决策次数: {stats['total_decisions']}")
    print(f"剪枝次数: {stats['pruned_count']}")
    print(f"保留次数: {stats['kept_count']}")
    print(f"剪枝率: {stats['prune_rate']:.2%}")
