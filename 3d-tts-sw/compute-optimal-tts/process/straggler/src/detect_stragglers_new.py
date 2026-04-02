#!/usr/bin/env python3
"""
重新检测Stragglers - 使用新定义
新定义：
1. token数量 > 100 (基础阈值)
2. 该分支 > (除自己外的最大token数) × 3
3. 允许top2或更多的情况 (多个分支都可以是straggler)
4. 支持branch_count >= 2 (不再限制 > 3)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def detect_straggler_new(branch_tokens: List[int]) -> Tuple[bool, List[int]]:
    """
    新的straggler检测逻辑
    
    Args:
        branch_tokens: 各分支的token数列表
    
    Returns:
        (has_straggler, straggler_indices)
    """
    if len(branch_tokens) < 2:
        return False, []
    
    # 基础阈值
    BASE_LENGTH = 100
    BASE_RATIO = 3.0
    
    straggler_indices = []
    
    # 对每个分支检查
    for i, token in enumerate(branch_tokens):
        if token <= BASE_LENGTH:
            continue
        
        # 找到除当前分支外的最大值
        other_tokens = branch_tokens[:i] + branch_tokens[i+1:]
        if not other_tokens:
            continue
        
        max_other = max(other_tokens)
        
        # 检查是否满足straggler条件
        if max_other > 0 and token >= max_other * BASE_RATIO:
            straggler_indices.append(i)
    
    return len(straggler_indices) > 0, straggler_indices


def scan_workloads(model_workloads_dir: str) -> Tuple[List[Dict], Dict, List[Dict]]:
    """扫描所有workload文件"""
    print(f"扫描目录: {model_workloads_dir}")
    
    workloads_dir = Path(model_workloads_dir)
    all_files = list(workloads_dir.rglob("*_workload.json"))  # 修改匹配模式
    
    print(f"找到 {len(all_files)} 个workload文件")
    
    stragglers = []
    all_steps = []
    
    # 统计信息
    stats = {
        'total_files': len(all_files),
        'total_steps': 0,
        'straggler_count': 0,
        'by_dataset': defaultdict(lambda: {'files': 0, 'stragglers': 0, 'steps': 0}),
        'by_model': defaultdict(lambda: {'files': 0, 'stragglers': 0, 'steps': 0}),
        'by_config': defaultdict(lambda: {'files': 0, 'stragglers': 0, 'steps': 0}),
        'by_branch_count': defaultdict(lambda: {'steps': 0, 'stragglers': 0})
    }
    
    for idx, file_path in enumerate(all_files):
        if idx % 1000 == 0:
            print(f"处理进度: {idx}/{len(all_files)}")
        
        try:
            with open(file_path, 'r') as f:
                workload = json.load(f)
        except Exception as e:
            print(f"读取文件失败: {file_path}, 错误: {e}")
            continue
        
        # 提取元数据
        parts = file_path.parts
        dataset_idx = -1
        for i, part in enumerate(parts):
            if 'beam_search' in part:
                dataset_idx = i
                break
        
        if dataset_idx == -1:
            continue
        
        dataset = parts[dataset_idx]
        model = parts[dataset_idx + 1] if dataset_idx + 1 < len(parts) else "unknown"
        prm = parts[dataset_idx + 2] if dataset_idx + 2 < len(parts) else "unknown"
        config = parts[dataset_idx + 3] if dataset_idx + 3 < len(parts) else "unknown"
        
        model_key = f"{model}/{prm}"
        
        # 更新统计
        stats['by_dataset'][dataset]['files'] += 1
        stats['by_model'][model_key]['files'] += 1
        stats['by_config'][config]['files'] += 1
        
        question_id = workload.get('question_id', 'unknown')
        steps = workload.get('decode', {}).get('steps', [])  # 修复: 从decode中获取steps
        
        # 处理每个step
        for step_data in steps:
            step = step_data.get('step', -1)
            branch_count = step_data.get('branch_count', 0)
            branch_tokens = step_data.get('branch_tokens', [])
            
            stats['total_steps'] += 1
            stats['by_dataset'][dataset]['steps'] += 1
            stats['by_model'][model_key]['steps'] += 1
            stats['by_config'][config]['steps'] += 1
            stats['by_branch_count'][branch_count]['steps'] += 1
            
            # 使用新定义检测straggler
            has_straggler, straggler_indices = detect_straggler_new(branch_tokens)
            
            # 记录所有步骤信息
            step_info = {
                'file': str(file_path),
                'question_id': question_id,
                'step': step,
                'branch_count': branch_count,
                'branch_tokens': branch_tokens,
                'is_straggler': has_straggler
            }
            
            if branch_tokens and len(branch_tokens) >= 2:
                sorted_tokens = sorted(branch_tokens, reverse=True)
                step_info['max_token'] = sorted_tokens[0]
                step_info['second_max_token'] = sorted_tokens[1] if len(sorted_tokens) > 1 else 0
                if step_info['second_max_token'] > 0:
                    step_info['ratio'] = step_info['max_token'] / step_info['second_max_token']
                else:
                    step_info['ratio'] = 0
            
            all_steps.append(step_info)
            
            # 如果检测到straggler，记录详细信息
            if has_straggler:
                straggler_info = {
                    'file_path': str(file_path),
                    'question_id': question_id,
                    'step_idx': step,
                    'branch_count': branch_count,
                    'branch_tokens': branch_tokens,
                    'straggler_indices': straggler_indices,
                    'dataset': dataset,
                    'model': model,
                    'prm': prm,
                    'config': config
                }
                
                if branch_tokens:
                    sorted_tokens = sorted(branch_tokens, reverse=True)
                    straggler_info['max_token'] = sorted_tokens[0]
                    straggler_info['second_max_token'] = sorted_tokens[1] if len(sorted_tokens) > 1 else 0
                    if straggler_info['second_max_token'] > 0:
                        straggler_info['ratio'] = straggler_info['max_token'] / straggler_info['second_max_token']
                
                stragglers.append(straggler_info)
                
                stats['straggler_count'] += 1
                stats['by_dataset'][dataset]['stragglers'] += 1
                stats['by_model'][model_key]['stragglers'] += 1
                stats['by_config'][config]['stragglers'] += 1
                stats['by_branch_count'][branch_count]['stragglers'] += 1
    
    print(f"\n检测完成!")
    print(f"总步骤数: {stats['total_steps']}")
    print(f"检测到straggler: {stats['straggler_count']} ({100*stats['straggler_count']/max(stats['total_steps'],1):.2f}%)")
    
    return stragglers, dict(stats), all_steps


def main():
    # 路径配置
    model_workloads_dir = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    straggler_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(straggler_dir, "data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("重新检测Stragglers - 使用新定义")
    print("="*80)
    print("新定义:")
    print("  1. token数量 > 100")
    print("  2. 该分支 > (除自己外的最大token数) × 3")
    print("  3. 允许top2或更多的情况")
    print("  4. 支持branch_count >= 2")
    print("="*80)
    print()
    
    # 扫描并检测
    stragglers, stats, all_steps = scan_workloads(model_workloads_dir)
    
    # 保存结果
    print("\n保存结果...")
    
    # 保存stragglers
    straggler_file = os.path.join(output_dir, "stragglers_new.json")
    with open(straggler_file, 'w') as f:
        json.dump(stragglers, f, indent=2)
    print(f"已保存: {straggler_file} ({len(stragglers)} 个stragglers)")
    
    # 保存元数据
    metadata_file = os.path.join(output_dir, "metadata_new.json")
    # 转换defaultdict为普通dict
    stats_serializable = {
        'total_files': stats['total_files'],
        'total_steps': stats['total_steps'],
        'straggler_count': stats['straggler_count'],
        'by_dataset': dict(stats['by_dataset']),
        'by_model': dict(stats['by_model']),
        'by_config': dict(stats['by_config']),
        'by_branch_count': dict(stats['by_branch_count'])
    }
    with open(metadata_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"已保存: {metadata_file}")
    
    # 保存所有步骤信息 (可能很大)
    all_steps_file = os.path.join(output_dir, "all_steps_new.json")
    print(f"保存all_steps.json (这可能需要一些时间...)")
    with open(all_steps_file, 'w') as f:
        json.dump(all_steps, f)
    print(f"已保存: {all_steps_file}")
    
    # 生成简单报告
    report_file = os.path.join(straggler_dir, "results", "detection_report_new.txt")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Straggler检测报告 (新定义)\n")
        f.write("="*80 + "\n\n")
        
        f.write("## 总体统计\n\n")
        f.write(f"总文件数: {stats['total_files']:,}\n")
        f.write(f"总步骤数: {stats['total_steps']:,}\n")
        f.write(f"Straggler数: {stats['straggler_count']:,}\n")
        f.write(f"Straggler率: {100*stats['straggler_count']/max(stats['total_steps'],1):.2f}%\n\n")
        
        f.write("## 按Branch数统计\n\n")
        for branch, data in sorted(stats['by_branch_count'].items()):
            if data['steps'] > 0:
                rate = 100 * data['stragglers'] / data['steps']
                f.write(f"Branch {branch}: {data['stragglers']}/{data['steps']} ({rate:.2f}%)\n")
        
        f.write("\n## 按配置统计\n\n")
        for config, data in sorted(stats['by_config'].items(), 
                                   key=lambda x: x[1]['stragglers'], reverse=True):
            if data['steps'] > 0:
                rate = 100 * data['stragglers'] / data['steps']
                f.write(f"{config}: {data['stragglers']}/{data['steps']} ({rate:.2f}%)\n")
        
        f.write("\n## 按数据集统计\n\n")
        for dataset, data in sorted(stats['by_dataset'].items()):
            if data['steps'] > 0:
                rate = 100 * data['stragglers'] / data['steps']
                f.write(f"{dataset}: {data['stragglers']}/{data['steps']} ({rate:.2f}%)\n")
    
    print(f"已保存: {report_file}")
    
    print("\n" + "="*80)
    print("检测完成!")
    print("="*80)
    print(f"\n检测结果对比:")
    print(f"  新定义: {stats['straggler_count']} 个stragglers ({100*stats['straggler_count']/max(stats['total_steps'],1):.2f}%)")
    print(f"  (旧定义约为1.08%)")
    print(f"\n新定义特点:")
    print(f"  - 允许branch_count=2,3的情况")
    print(f"  - 允许top2或更多个straggler")
    print(f"  - 更灵活的检测逻辑")


if __name__ == "__main__":
    main()
