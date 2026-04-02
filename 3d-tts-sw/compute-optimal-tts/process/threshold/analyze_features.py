#!/usr/bin/env python3
"""
动态剪枝特征提取和分析

从beam search的输出中提取特征,分析正确分支和错误分支的差异
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

def safe_mean(lst):
    """安全计算平均值"""
    if not lst or len(lst) == 0:
        return 0.0
    return float(np.mean(lst))

def safe_std(lst):
    """安全计算标准差"""
    if not lst or len(lst) <= 1:
        return 0.0
    return float(np.std(lst))

def extract_step_features(path_data: dict, step_idx: int, all_paths: List[dict]) -> Dict:
    """从单个步骤提取特征
    
    Args:
        path_data: 单个路径的数据
        step_idx: 当前步骤索引(模拟运行时)
        all_paths: 所有路径数据(用于计算相对特征)
    
    Returns:
        特征字典
    """
    features = {}
    
    # 基础信息
    features['path_idx'] = path_data.get('path_idx', -1)
    features['step_idx'] = step_idx
    
    # Reward特征
    if 'reward_history' in path_data and len(path_data['reward_history']) > step_idx:
        rewards = path_data['reward_history'][:step_idx+1]
        features['reward_current'] = rewards[-1] if rewards else 0.0
        features['reward_mean'] = safe_mean(rewards)
        features['reward_std'] = safe_std(rewards)
        features['reward_min'] = min(rewards) if rewards else 0.0
        features['reward_max'] = max(rewards) if rewards else 0.0
        
        if len(rewards) > 1:
            features['reward_trend'] = rewards[-1] - rewards[0]
            features['reward_volatility'] = safe_std([rewards[i+1] - rewards[i] for i in range(len(rewards)-1)])
        else:
            features['reward_trend'] = 0.0
            features['reward_volatility'] = 0.0
    
    # Token概率特征
    if 'token_prob_history' in path_data and len(path_data['token_prob_history']) > step_idx:
        # token_prob_history[step_idx]是一个list,包含该步所有token的概率
        step_token_probs = path_data['token_prob_history'][step_idx]
        if isinstance(step_token_probs, list):
            features['token_prob_mean'] = safe_mean(step_token_probs)
            features['token_prob_min'] = min(step_token_probs) if step_token_probs else 0.0
            features['token_prob_max'] = max(step_token_probs) if step_token_probs else 0.0
            features['token_prob_std'] = safe_std(step_token_probs)
            
            # 计算所有步骤的平均token概率
            all_step_probs = []
            for i in range(min(step_idx+1, len(path_data['token_prob_history']))):
                step_probs = path_data['token_prob_history'][i]
                if isinstance(step_probs, list):
                    all_step_probs.append(safe_mean(step_probs))
            
            features['token_prob_history_mean'] = safe_mean(all_step_probs)
            features['token_prob_history_min'] = min(all_step_probs) if all_step_probs else 0.0
    
    # 累积概率特征
    if 'prob_history' in path_data and len(path_data['prob_history']) > step_idx:
        probs = path_data['prob_history'][:step_idx+1]
        features['cum_prob'] = probs[-1] if probs else 0.0
        if len(probs) > 1:
            features['cum_prob_trend'] = probs[-1] - probs[0]
        else:
            features['cum_prob_trend'] = 0.0
    
    # 相对特征(相对于其他分支)
    if 'reward_current' in features:
        current_reward = features['reward_current']
        all_rewards = []
        for other_path in all_paths:
            if 'reward_history' in other_path and len(other_path['reward_history']) > step_idx:
                all_rewards.append(other_path['reward_history'][step_idx])
        
        if all_rewards:
            # 排名(0=最好, 1=最差)
            sorted_rewards = sorted(all_rewards, reverse=True)
            rank = sorted_rewards.index(current_reward) if current_reward in sorted_rewards else len(sorted_rewards)
            features['reward_rank'] = rank / max(len(all_rewards) - 1, 1)
            
            # 与最好分支的差距
            features['reward_gap_to_best'] = max(all_rewards) - current_reward
            features['reward_gap_to_mean'] = current_reward - safe_mean(all_rewards)
    
    # 结构特征
    features['num_branches'] = len(all_paths)
    features['current_length'] = step_idx + 1
    
    return features

def analyze_question(question_dir: str, is_correct: bool) -> List[Dict]:
    """分析单个题目的所有分支在各个步骤的特征"""
    record_file = os.path.join(question_dir, "record_0.jsonl")
    
    all_features = []
    
    if not os.path.exists(record_file):
        return all_features
    
    try:
        with open(record_file, 'r') as f:
            data = json.loads(f.read())
        
        if 'output' not in data:
            return all_features
        
        outputs = data['output'] if isinstance(data['output'], list) else [data['output']]
        
        # 确定最大步数
        max_steps = 0
        for output in outputs:
            if 'reward_history' in output:
                max_steps = max(max_steps, len(output['reward_history']))
        
        # 对每个步骤提取特征
        question_idx = int(os.path.basename(question_dir).split('_')[1])
        
        for step_idx in range(max_steps):
            for path_idx, output in enumerate(outputs):
                # 只提取该分支存在的步骤
                if 'reward_history' in output and len(output['reward_history']) > step_idx:
                    features = extract_step_features(output, step_idx, outputs)
                    features['is_correct_question'] = is_correct
                    features['question_idx'] = question_idx
                    features['max_steps'] = max_steps
                    
                    all_features.append(features)
    
    except Exception as e:
        print(f"Error processing {question_dir}: {e}")
        import traceback
        traceback.print_exc()
    
    return all_features

def analyze_dataset(base_dir: str, correct_indices: List[int]) -> Tuple[List[Dict], List[Dict]]:
    """分析整个数据集"""
    
    correct_features = []
    incorrect_features = []
    
    question_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("question_")],
                          key=lambda x: int(x.split('_')[1]))
    
    for question_dir_name in question_dirs:
        question_idx = int(question_dir_name.split('_')[1])
        question_dir = os.path.join(base_dir, question_dir_name)
        
        is_correct = question_idx in correct_indices
        
        features = analyze_question(question_dir, is_correct)
        
        if is_correct:
            correct_features.extend(features)
        else:
            incorrect_features.extend(features)
        
        if len(features) > 0:
            status = "正确" if is_correct else "错误"
            print(f"题目 {question_idx} ({status}): 提取了 {len(features)} 个特征点")
    
    return correct_features, incorrect_features

def compute_feature_stats(features_list: List[Dict]) -> Dict:
    """计算特征统计"""
    stats = defaultdict(list)
    
    exclude_keys = {'path_idx', 'question_idx', 'is_correct_question', 'step_idx'}
    
    for features in features_list:
        for key, value in features.items():
            if key not in exclude_keys and isinstance(value, (int, float)):
                stats[key].append(value)
    
    result = {}
    for key, values in stats.items():
        if values:
            result[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'count': len(values)
            }
    
    return result

def compare_features(correct_stats: Dict, incorrect_stats: Dict) -> List[Tuple]:
    """比较特征,计算效应量"""
    
    feature_diffs = []
    
    for key in correct_stats.keys():
        if key in incorrect_stats:
            correct_mean = correct_stats[key]['mean']
            incorrect_mean = incorrect_stats[key]['mean']
            
            correct_std = correct_stats[key]['std']
            incorrect_std = incorrect_stats[key]['std']
            
            pooled_std = np.sqrt((correct_std**2 + incorrect_std**2) / 2)
            
            if pooled_std > 1e-10:
                cohens_d = abs(correct_mean - incorrect_mean) / pooled_std
                feature_diffs.append((key, cohens_d, correct_mean, incorrect_mean))
    
    feature_diffs.sort(key=lambda x: x[1], reverse=True)
    
    return feature_diffs

def main():
    base_dir = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/output/AIME24_beam_search/Qwen2.5-Math-7B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/16384_8_2"
    
    # 做对的题目索引(使用prm_avg_max指标)
    correct_indices = [0, 1, 3, 6, 18]
    
    print("开始分析数据集特征...")
    print("="*80)
    
    correct_features, incorrect_features = analyze_dataset(base_dir, correct_indices)
    
    print("\n" + "="*80)
    print(f"正确题目的特征点数: {len(correct_features)}")
    print(f"错误题目的特征点数: {len(incorrect_features)}")
    
    # 计算统计
    correct_stats = compute_feature_stats(correct_features)
    incorrect_stats = compute_feature_stats(incorrect_features)
    
    print("\n" + "="*80)
    print("正确题目的特征统计 (Top 10):")
    print("="*80)
    for i, key in enumerate(sorted(correct_stats.keys())[:10]):
        stats = correct_stats[key]
        print(f"\n{key}:")
        print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Median: {stats['median']:.6f}")
    
    print("\n" + "="*80)
    print("错误题目的特征统计 (Top 10):")
    print("="*80)
    for i, key in enumerate(sorted(incorrect_stats.keys())[:10]):
        stats = incorrect_stats[key]
        print(f"\n{key}:")
        print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Median: {stats['median']:.6f}")
    
    # 特征差异分析
    print("\n" + "="*80)
    print("最显著的特征差异 (Cohen's d):")
    print("="*80)
    
    feature_diffs = compare_features(correct_stats, incorrect_stats)
    
    for i, (feature, cohens_d, correct_mean, incorrect_mean) in enumerate(feature_diffs[:20], 1):
        direction = "正确 > 错误" if correct_mean > incorrect_mean else "正确 < 错误"
        print(f"\n{i}. {feature}:")
        print(f"   效应量(Cohen's d): {cohens_d:.4f}")
        print(f"   方向: {direction}")
        print(f"   正确题均值: {correct_mean:.6f}, 错误题均值: {incorrect_mean:.6f}")
    
    # 保存结果
    output_dir = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/threshold"
    
    output_data = {
        'correct_stats': correct_stats,
        'incorrect_stats': incorrect_stats,
        'feature_importance': [
            {
                'feature': f[0],
                'cohens_d': f[1],
                'correct_mean': f[2],
                'incorrect_mean': f[3]
            }
            for f in feature_diffs
        ],
        'summary': {
            'num_correct_samples': len(correct_features),
            'num_incorrect_samples': len(incorrect_features),
            'num_features': len(correct_stats)
        }
    }
    
    output_file = os.path.join(output_dir, 'feature_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"特征分析结果已保存到: {output_file}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
