#!/usr/bin/env python3
"""
分析workload文件中的branch分布情况
汇总8, 7, 6, 5 branch的统计信息
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def analyze_workload_file(filepath: str) -> Dict[str, Any]:
    """分析单个workload文件"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    question_id = data['question_id']
    steps = data['decode']['steps']
    
    # 统计各个branch_count的出现次数
    branch_count_stats = defaultdict(int)
    step_details = []
    
    for step_idx, step in enumerate(steps):
        branch_count = step['branch_count']
        if branch_count > 0:
            branch_count_stats[branch_count] += 1
            
            # 收集详细信息
            step_info = {
                'step': step_idx,
                'branch_count': branch_count,
                'selected_branch': step.get('selected_branch_index', -1),
                'branch_tokens': step.get('branch_tokens', []),
                'branch_rewards': step.get('branch_rewards', [])
            }
            step_details.append(step_info)
    
    return {
        'question_id': question_id,
        'branch_count_stats': dict(branch_count_stats),
        'step_details': step_details,
        'total_steps': len([s for s in steps if s['branch_count'] > 0])
    }


def main():
    # 源目录
    source_dir = Path('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/math-shepherd-mistral-7b-prm/16384_8_1')
    
    # 目标目录
    output_dir = Path('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/branch')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 收集所有workload文件
    workload_files = sorted(source_dir.glob('question_*_workload.json'))
    
    print(f"找到 {len(workload_files)} 个workload文件")
    
    # 全局统计
    global_branch_stats = {
        8: {'total_occurrences': 0, 'questions': []},
        7: {'total_occurrences': 0, 'questions': []},
        6: {'total_occurrences': 0, 'questions': []},
        5: {'total_occurrences': 0, 'questions': []}
    }
    
    # 每个问题的详细分析
    all_questions_analysis = []
    
    # 分析每个文件
    for filepath in workload_files:
        print(f"处理: {filepath.name}")
        analysis = analyze_workload_file(str(filepath))
        all_questions_analysis.append(analysis)
        
        # 更新全局统计
        for branch_count, occurrences in analysis['branch_count_stats'].items():
            if branch_count in [8, 7, 6, 5]:
                global_branch_stats[branch_count]['total_occurrences'] += occurrences
                global_branch_stats[branch_count]['questions'].append({
                    'question_id': analysis['question_id'],
                    'occurrences': occurrences,
                    'total_steps': analysis['total_steps']
                })
    
    # 生成汇总报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Branch分布汇总报告 - 16384_8_1配置")
    report_lines.append("=" * 80)
    report_lines.append(f"\n数据源: {source_dir}")
    report_lines.append(f"分析问题数: {len(workload_files)}")
    report_lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 总体统计
    report_lines.append("\n" + "=" * 80)
    report_lines.append("1. 总体统计")
    report_lines.append("=" * 80)
    
    for branch_count in [8, 7, 6, 5]:
        stats = global_branch_stats[branch_count]
        report_lines.append(f"\n{branch_count} Branches:")
        report_lines.append(f"  - 总出现次数: {stats['total_occurrences']}")
        report_lines.append(f"  - 涉及问题数: {len(stats['questions'])}")
        if stats['questions']:
            avg_occurrences = stats['total_occurrences'] / len(stats['questions'])
            report_lines.append(f"  - 平均每题出现: {avg_occurrences:.2f} 次")
    
    # 2. 按branch_count分类详细信息
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("2. 按Branch Count分类详细信息")
    report_lines.append("=" * 80)
    
    for branch_count in [8, 7, 6, 5]:
        stats = global_branch_stats[branch_count]
        report_lines.append(f"\n\n{'#' * 80}")
        report_lines.append(f"## {branch_count} Branches 详细信息")
        report_lines.append(f"{'#' * 80}")
        
        if stats['questions']:
            report_lines.append(f"\n总出现次数: {stats['total_occurrences']}")
            report_lines.append(f"涉及的问题:")
            
            # 按出现次数排序
            sorted_questions = sorted(stats['questions'], key=lambda x: x['occurrences'], reverse=True)
            
            for q_info in sorted_questions:
                report_lines.append(f"\n  {q_info['question_id']}:")
                report_lines.append(f"    - 出现次数: {q_info['occurrences']}")
                report_lines.append(f"    - 总步数: {q_info['total_steps']}")
                report_lines.append(f"    - 占比: {q_info['occurrences']/q_info['total_steps']*100:.1f}%")
        else:
            report_lines.append(f"\n未发现 {branch_count} branches 的情况")
    
    # 3. 每个问题的详细分析
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("3. 每个问题的详细分析")
    report_lines.append("=" * 80)
    
    for analysis in all_questions_analysis:
        report_lines.append(f"\n\n{'#' * 80}")
        report_lines.append(f"## {analysis['question_id']}")
        report_lines.append(f"{'#' * 80}")
        
        report_lines.append(f"\n总步数: {analysis['total_steps']}")
        report_lines.append(f"\nBranch Count分布:")
        for bc, count in sorted(analysis['branch_count_stats'].items(), reverse=True):
            report_lines.append(f"  {bc} branches: {count} 次 ({count/analysis['total_steps']*100:.1f}%)")
        
        # 只显示8, 7, 6, 5 branch的详细信息
        target_branches = [8, 7, 6, 5]
        filtered_steps = [s for s in analysis['step_details'] if s['branch_count'] in target_branches]
        
        if filtered_steps:
            report_lines.append(f"\n目标Branch (8/7/6/5) 详细信息:")
            for step_info in filtered_steps:
                report_lines.append(f"\n  Step {step_info['step']} - {step_info['branch_count']} branches:")
                report_lines.append(f"    选中: Branch {step_info['selected_branch']}")
                report_lines.append(f"    Tokens: {step_info['branch_tokens']}")
                
                if step_info['branch_rewards']:
                    rewards = step_info['branch_rewards']
                    report_lines.append(f"    Rewards: [min={min(rewards):.4f}, max={max(rewards):.4f}, avg={sum(rewards)/len(rewards):.4f}]")
                    report_lines.append(f"    Selected reward: {rewards[step_info['selected_branch']]:.4f}")
    
    # 4. 汇总统计表格
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("4. 汇总统计表格")
    report_lines.append("=" * 80)
    
    report_lines.append("\n问题ID      | 总步数 | 8-br | 7-br | 6-br | 5-br | 其他")
    report_lines.append("-" * 80)
    
    for analysis in all_questions_analysis:
        q_id = analysis['question_id'].replace('question_', 'Q')
        total = analysis['total_steps']
        bc_8 = analysis['branch_count_stats'].get(8, 0)
        bc_7 = analysis['branch_count_stats'].get(7, 0)
        bc_6 = analysis['branch_count_stats'].get(6, 0)
        bc_5 = analysis['branch_count_stats'].get(5, 0)
        bc_other = total - (bc_8 + bc_7 + bc_6 + bc_5)
        
        report_lines.append(f"{q_id:11s} | {total:6d} | {bc_8:4d} | {bc_7:4d} | {bc_6:4d} | {bc_5:4d} | {bc_other:4d}")
    
    # 总计行
    total_steps = sum(a['total_steps'] for a in all_questions_analysis)
    total_8 = sum(a['branch_count_stats'].get(8, 0) for a in all_questions_analysis)
    total_7 = sum(a['branch_count_stats'].get(7, 0) for a in all_questions_analysis)
    total_6 = sum(a['branch_count_stats'].get(6, 0) for a in all_questions_analysis)
    total_5 = sum(a['branch_count_stats'].get(5, 0) for a in all_questions_analysis)
    total_other = total_steps - (total_8 + total_7 + total_6 + total_5)
    
    report_lines.append("-" * 80)
    report_lines.append(f"{'总计':11s} | {total_steps:6d} | {total_8:4d} | {total_7:4d} | {total_6:4d} | {total_5:4d} | {total_other:4d}")
    report_lines.append(f"{'百分比':11s} | {'100%':6s} | {total_8/total_steps*100:3.1f}% | {total_7/total_steps*100:3.1f}% | {total_6/total_steps*100:3.1f}% | {total_5/total_steps*100:3.1f}% | {total_other/total_steps*100:3.1f}%")
    
    # 写入报告文件
    report_path = output_dir / 'branch_distribution_summary_16384_8_1.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n报告已生成: {report_path}")
    
    # 保存JSON格式的详细数据
    json_output = {
        'metadata': {
            'source_dir': str(source_dir),
            'total_questions': len(workload_files),
            'generated_at': __import__('datetime').datetime.now().isoformat()
        },
        'global_statistics': {
            str(k): {
                'total_occurrences': v['total_occurrences'],
                'question_count': len(v['questions']),
                'questions': v['questions']
            }
            for k, v in global_branch_stats.items()
        },
        'question_analyses': all_questions_analysis
    }
    
    json_path = output_dir / 'branch_distribution_data_16384_8_1.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"JSON数据已保存: {json_path}")
    
    # 输出简要统计
    print("\n" + "=" * 80)
    print("分析完成! 简要统计:")
    print("=" * 80)
    for branch_count in [8, 7, 6, 5]:
        stats = global_branch_stats[branch_count]
        print(f"{branch_count} Branches: {stats['total_occurrences']} 次出现, {len(stats['questions'])} 个问题")


if __name__ == '__main__':
    main()
