#!/usr/bin/env python3
"""
学习动态threshold策略

基于特征分析结果,学习一个threshold函数,使得:
1. 正确题目的正确分支不被剪枝
2. 错误题目的错误分支被剪枝
3. 不显式使用题目标签
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_features(feature_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载特征数据"""
    with open(feature_file, 'r') as f:
        data = json.load(f)
    
    # 我们需要从原始数据重新提取,因为之前只保存了统计信息
    # 这里简化处理:基于统计信息生成决策规则
    return data

def extract_decision_rules(feature_analysis: Dict) -> Dict:
    """从特征分析中提取决策规则
    
    基于特征差异,设计一个简单但有效的threshold策略
    """
    
    feature_importance = feature_analysis['feature_importance']
    correct_stats = feature_analysis['correct_stats']
    incorrect_stats = feature_analysis['incorrect_stats']
    
    # 提取top K最重要的特征
    top_features = feature_importance[:10]
    
    rules = {
        'strategy': 'multi_threshold',
        'features': []
    }
    
    for feat_info in top_features:
        feature_name = feat_info['feature']
        cohens_d = feat_info['cohens_d']
        correct_mean = feat_info['correct_mean']
        incorrect_mean = feat_info['incorrect_mean']
        
        if feature_name in correct_stats and feature_name in incorrect_stats:
            correct_q25 = correct_stats[feature_name]['q25']
            correct_median = correct_stats[feature_name]['median']
            incorrect_q75 = incorrect_stats[feature_name]['q75']
            incorrect_median = incorrect_stats[feature_name]['median']
            
            # 设计threshold:在正确和错误分布之间
            if correct_mean > incorrect_mean:
                # 正确分支的特征值更高
                # threshold设在错误分支的75%分位点附近
                threshold = incorrect_q75 + 0.3 * (correct_q25 - incorrect_q75)
                direction = 'greater'  # 大于threshold才保留
            else:
                # 正确分支的特征值更低
                threshold = incorrect_q75 - 0.3 * (incorrect_q75 - correct_q25)
                direction = 'less'  # 小于threshold才保留
            
            rules['features'].append({
                'name': feature_name,
                'threshold': float(threshold),
                'direction': direction,
                'weight': float(cohens_d),
                'correct_mean': float(correct_mean),
                'incorrect_mean': float(incorrect_mean)
            })
    
    return rules

def design_adaptive_threshold(feature_analysis: Dict) -> Dict:
    """设计自适应threshold策略
    
    核心思想:
    1. 使用reward作为主要指标
    2. 使用相对排名而非绝对值
    3. 早期宽松,后期严格
    4. 考虑多个维度的综合评分
    """
    
    correct_stats = feature_analysis['correct_stats']
    incorrect_stats = feature_analysis['incorrect_stats']
    
    # 提取关键统计
    reward_correct_q25 = correct_stats['reward_current']['q25']
    reward_incorrect_q75 = incorrect_stats['reward_current']['q75']
    
    strategy = {
        'version': '1.0',
        'description': '基于多维度特征的自适应剪枝策略',
        
        'base_thresholds': {
            # Reward阈值:相对于分支中的最大reward
            'reward_relative': {
                'early_stage': 0.5,    # 前30%步骤:保留reward > max_reward * 0.5的分支
                'mid_stage': 0.7,      # 中间40%步骤:保留reward > max_reward * 0.7的分支  
                'late_stage': 0.85,    # 后30%步骤:保留reward > max_reward * 0.85的分支
            },
            
            # 绝对reward阈值
            'reward_absolute': {
                'min_threshold': float(reward_incorrect_q75),  # 低于此值直接剪枝
                'safe_threshold': float(reward_correct_q25),   # 高于此值一定保留
            },
            
            # 排名阈值
            'rank_threshold': {
                'max_rank': 0.7,  # 排名后30%的分支可能被剪枝
            }
        },
        
        'feature_weights': {
            'reward_current': 0.4,
            'reward_mean': 0.2,
            'token_prob_mean': 0.15,
            'cum_prob': 0.15,
            'reward_trend': 0.1
        },
        
        'pruning_rules': [
            {
                'name': 'low_absolute_reward',
                'condition': 'reward_current < reward_absolute.min_threshold',
                'action': 'prune',
                'priority': 1
            },
            {
                'name': 'high_absolute_reward',
                'condition': 'reward_current > reward_absolute.safe_threshold',
                'action': 'keep',
                'priority': 1
            },
            {
                'name': 'relative_reward_based',
                'condition': 'reward_rank > rank_threshold.max_rank AND reward_current < max_reward * reward_relative[stage]',
                'action': 'prune',
                'priority': 2
            },
            {
                'name': '综合得分低',
                'condition': 'weighted_score < threshold',
                'action': 'prune',
                'priority': 3
            }
        ],
        
        'stage_definition': {
            'early': [0, 0.3],   # 0-30%
            'mid': [0.3, 0.7],   # 30-70%
            'late': [0.7, 1.0]   # 70-100%
        }
    }
    
    return strategy

def compute_score_thresholds(feature_analysis: Dict) -> Dict:
    """计算综合得分的阈值
    
    基于正确/错误样本的综合得分分布,确定剪枝阈值
    """
    
    correct_stats = feature_analysis['correct_stats']
    incorrect_stats = feature_analysis['incorrect_stats']
    
    # 模拟计算综合得分(加权平均)
    # 这里使用统计值来估算
    
    weights = {
        'reward_current': 0.4,
        'reward_mean': 0.2,
        'token_prob_mean': 0.15,
        'cum_prob': 0.15,
        'reward_trend': 0.1
    }
    
    # 标准化并计算得分范围
    correct_score_est = sum(
        weights[feat] * correct_stats[feat]['median'] 
        for feat in weights.keys() if feat in correct_stats
    )
    
    incorrect_score_est = sum(
        weights[feat] * incorrect_stats[feat]['median']
        for feat in weights.keys() if feat in incorrect_stats
    )
    
    # threshold设在两者之间,偏向保留
    score_threshold = incorrect_score_est + 0.4 * (correct_score_est - incorrect_score_est)
    
    return {
        'score_threshold': float(score_threshold),
        'correct_score_estimate': float(correct_score_est),
        'incorrect_score_estimate': float(incorrect_score_est)
    }

def main():
    feature_file = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/feature_analysis.json"
    
    print("加载特征分析结果...")
    feature_analysis = load_features(feature_file)
    
    print("\n设计自适应threshold策略...")
    threshold_strategy = design_adaptive_threshold(feature_analysis)
    
    print("\n提取决策规则...")
    decision_rules = extract_decision_rules(feature_analysis)
    
    print("\n计算综合得分阈值...")
    score_thresholds = compute_score_thresholds(feature_analysis)
    
    # 合并到最终模型
    threshold_model = {
        'adaptive_strategy': threshold_strategy,
        'decision_rules': decision_rules,
        'score_thresholds': score_thresholds,
        'metadata': {
            'training_data': '16384_8_2',
            'num_correct_samples': feature_analysis['summary']['num_correct_samples'],
            'num_incorrect_samples': feature_analysis['summary']['num_incorrect_samples'],
            'version': '1.0'
        }
    }
    
    # 保存模型
    output_file = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold/threshold_model.json"
    with open(output_file, 'w') as f:
        json.dump(threshold_model, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Threshold模型已保存!")
    print(f"文件: {output_file}")
    print(f"{'='*80}")
    
    # 打印关键信息
    print("\n主要策略:")
    print(f"- Reward相对阈值 (早期/中期/晚期): {threshold_strategy['base_thresholds']['reward_relative']}")
    print(f"- Reward绝对阈值: {threshold_strategy['base_thresholds']['reward_absolute']}")
    print(f"- 排名阈值: {threshold_strategy['base_thresholds']['rank_threshold']}")
    print(f"\n综合得分阈值: {score_thresholds['score_threshold']:.6f}")
    print(f"  正确样本估计得分: {score_thresholds['correct_score_estimate']:.6f}")
    print(f"  错误样本估计得分: {score_thresholds['incorrect_score_estimate']:.6f}")
    
    print(f"\nTop 5 决策特征:")
    for i, feat in enumerate(decision_rules['features'][:5], 1):
        print(f"{i}. {feat['name']}: threshold={feat['threshold']:.6f}, direction={feat['direction']}, weight={feat['weight']:.4f}")

if __name__ == '__main__':
    main()
