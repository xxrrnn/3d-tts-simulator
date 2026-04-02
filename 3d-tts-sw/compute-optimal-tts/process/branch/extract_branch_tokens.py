#!/usr/bin/env python3
"""
提取并汇总所有branch_count=8,7,6,5的step信息
重点关注每个branch的token数量
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def extract_target_branches(filepath: str) -> List[Dict[str, Any]]:
    """提取目标branch count的step信息"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    question_id = data['question_id']
    steps = data['decode']['steps']
    
    target_steps = []
    
    for step_idx, step in enumerate(steps):
        branch_count = step['branch_count']
        
        # 只提取8, 7, 6, 5 branch的步骤
        if branch_count in [8, 7, 6, 5]:
            step_info = {
                'question_id': question_id,
                'step_index': step_idx,
                'branch_count': branch_count,
                'selected_branch_index': step.get('selected_branch_index', -1),
                'branch_tokens': step.get('branch_tokens', []),
                'branch_rewards': step.get('branch_rewards', []),
                'source_file': filepath
            }
            target_steps.append(step_info)
    
    return target_steps


def main():
    # 源目录
    source_dir = Path('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/wordload/model_workloads/AIME24_beam_search/Qwen2.5-Math-1.5B-Instruct/math-shepherd-mistral-7b-prm/16384_8_1')
    
    # 目标目录
    output_dir = Path('/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/branch')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 收集所有workload文件
    workload_files = sorted(source_dir.glob('question_*_workload.json'))
    
    print(f"找到 {len(workload_files)} 个workload文件")
    
    # 按branch_count分类收集所有steps
    all_steps_by_branch = {
        8: [],
        7: [],
        6: [],
        5: []
    }
    
    total_steps = 0
    
    # 处理每个文件
    for filepath in workload_files:
        print(f"处理: {filepath.name}")
        target_steps = extract_target_branches(str(filepath))
        total_steps += len(target_steps)
        
        for step_info in target_steps:
            bc = step_info['branch_count']
            all_steps_by_branch[bc].append(step_info)
    
    print(f"\n总共提取了 {total_steps} 个目标步骤")
    
    # 生成详细汇总报告
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("Branch Token详细信息汇总 - 16384_8_1配置")
    report_lines.append("=" * 100)
    report_lines.append(f"\n数据源: {source_dir}")
    report_lines.append(f"汇总时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"总步骤数: {total_steps}")
    report_lines.append(f"  - 8 Branches: {len(all_steps_by_branch[8])} 步")
    report_lines.append(f"  - 7 Branches: {len(all_steps_by_branch[7])} 步")
    report_lines.append(f"  - 6 Branches: {len(all_steps_by_branch[6])} 步")
    report_lines.append(f"  - 5 Branches: {len(all_steps_by_branch[5])} 步")
    
    # 为每种branch count生成详细信息
    for branch_count in [8, 7, 6, 5]:
        steps = all_steps_by_branch[branch_count]
        
        report_lines.append("\n\n" + "=" * 100)
        report_lines.append(f"Branch Count = {branch_count} (共 {len(steps)} 个步骤)")
        report_lines.append("=" * 100)
        
        if not steps:
            report_lines.append("\n无数据")
            continue
        
        # 统计token信息
        all_tokens = []
        for step in steps:
            all_tokens.extend(step['branch_tokens'])
        
        if all_tokens:
            report_lines.append(f"\nToken统计:")
            report_lines.append(f"  - 总Token数: {sum(all_tokens)}")
            report_lines.append(f"  - 平均Token: {sum(all_tokens)/len(all_tokens):.2f}")
            report_lines.append(f"  - 最小Token: {min(all_tokens)}")
            report_lines.append(f"  - 最大Token: {max(all_tokens)}")
            report_lines.append(f"  - 中位数Token: {sorted(all_tokens)[len(all_tokens)//2]}")
        
        # 按问题分组展示
        steps_by_question = defaultdict(list)
        for step in steps:
            steps_by_question[step['question_id']].append(step)
        
        report_lines.append(f"\n详细信息 (按问题分组):")
        report_lines.append("-" * 100)
        
        for question_id in sorted(steps_by_question.keys()):
            question_steps = steps_by_question[question_id]
            report_lines.append(f"\n【{question_id}】共 {len(question_steps)} 个{branch_count}-branch步骤")
            
            for step in sorted(question_steps, key=lambda x: x['step_index']):
                report_lines.append(f"\n  Step {step['step_index']}:")
                
                # Token信息
                tokens = step['branch_tokens']
                report_lines.append(f"    Branch Tokens: {tokens}")
                report_lines.append(f"      - 总计: {sum(tokens)} tokens")
                report_lines.append(f"      - 平均: {sum(tokens)/len(tokens):.2f}")
                report_lines.append(f"      - 范围: [{min(tokens)}, {max(tokens)}]")
                
                # 选中的branch
                selected_idx = step['selected_branch_index']
                if selected_idx >= 0 and selected_idx < len(tokens):
                    selected_token = tokens[selected_idx]
                    report_lines.append(f"      - 选中: Branch {selected_idx} ({selected_token} tokens)")
                
                # Reward信息
                rewards = step['branch_rewards']
                if rewards:
                    report_lines.append(f"    Branch Rewards: [{', '.join([f'{r:.4f}' for r in rewards])}]")
                    if selected_idx >= 0 and selected_idx < len(rewards):
                        selected_reward = rewards[selected_idx]
                        report_lines.append(f"      - 选中reward: {selected_reward:.4f}")
    
    # 生成跨branch count的对比分析
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append("对比分析")
    report_lines.append("=" * 100)
    
    report_lines.append("\nToken统计对比:")
    report_lines.append(f"{'Branch Count':<15} | {'步数':<8} | {'总Tokens':<12} | {'平均Token':<12} | {'最小':<8} | {'最大':<8} | {'中位数':<8}")
    report_lines.append("-" * 100)
    
    for branch_count in [8, 7, 6, 5]:
        steps = all_steps_by_branch[branch_count]
        if steps:
            all_tokens = []
            for step in steps:
                all_tokens.extend(step['branch_tokens'])
            
            avg_token = sum(all_tokens) / len(all_tokens)
            median_token = sorted(all_tokens)[len(all_tokens)//2]
            
            report_lines.append(
                f"{branch_count:<15} | {len(steps):<8} | {sum(all_tokens):<12} | "
                f"{avg_token:<12.2f} | {min(all_tokens):<8} | {max(all_tokens):<8} | {median_token:<8}"
            )
    
    # 按问题汇总统计表
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append("按问题汇总的Token统计")
    report_lines.append("=" * 100)
    
    # 收集所有问题
    all_question_ids = set()
    for bc in [8, 7, 6, 5]:
        for step in all_steps_by_branch[bc]:
            all_question_ids.add(step['question_id'])
    
    report_lines.append(f"\n{'问题ID':<15} | {'8-br步数':<10} | {'8-br总tokens':<15} | {'7-br步数':<10} | {'7-br总tokens':<15} | {'6-br步数':<10} | {'6-br总tokens':<15} | {'5-br步数':<10} | {'5-br总tokens':<15}")
    report_lines.append("-" * 150)
    
    for question_id in sorted(all_question_ids):
        row = [question_id]
        
        for bc in [8, 7, 6, 5]:
            q_steps = [s for s in all_steps_by_branch[bc] if s['question_id'] == question_id]
            step_count = len(q_steps)
            
            if q_steps:
                total_tokens = sum(sum(s['branch_tokens']) for s in q_steps)
            else:
                total_tokens = 0
            
            row.append(f"{step_count:<10}")
            row.append(f"{total_tokens:<15}")
        
        report_lines.append(" | ".join(row))
    
    # 写入报告文件
    report_path = output_dir / 'branch_tokens_summary_16384_8_1.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n详细Token报告已生成: {report_path}")
    
    # 保存JSON格式数据
    json_output = {
        'metadata': {
            'source_dir': str(source_dir),
            'total_steps': total_steps,
            'generated_at': __import__('datetime').datetime.now().isoformat()
        },
        'steps_by_branch_count': {
            str(bc): all_steps_by_branch[bc] for bc in [8, 7, 6, 5]
        }
    }
    
    json_path = output_dir / 'branch_tokens_data_16384_8_1.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"JSON数据已保存: {json_path}")
    
    # 生成CSV格式的简化版（便于Excel分析）
    csv_lines = []
    csv_lines.append("question_id,step_index,branch_count,selected_branch_index,branch_0_tokens,branch_1_tokens,branch_2_tokens,branch_3_tokens,branch_4_tokens,branch_5_tokens,branch_6_tokens,branch_7_tokens,selected_reward,avg_reward")
    
    for bc in [8, 7, 6, 5]:
        for step in sorted(all_steps_by_branch[bc], key=lambda x: (x['question_id'], x['step_index'])):
            tokens = step['branch_tokens']
            rewards = step['branch_rewards']
            
            # 填充token数据（不足的用空字符串）
            tokens_padded = tokens + [''] * (8 - len(tokens))
            
            selected_reward = rewards[step['selected_branch_index']] if step['selected_branch_index'] >= 0 and rewards else ''
            avg_reward = sum(rewards) / len(rewards) if rewards else ''
            
            csv_line = [
                step['question_id'],
                str(step['step_index']),
                str(bc),
                str(step['selected_branch_index']),
            ] + [str(t) for t in tokens_padded] + [
                f"{selected_reward:.4f}" if selected_reward != '' else '',
                f"{avg_reward:.4f}" if avg_reward != '' else ''
            ]
            
            csv_lines.append(','.join(csv_line))
    
    csv_path = output_dir / 'branch_tokens_16384_8_1.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"CSV数据已保存: {csv_path}")
    
    print("\n" + "=" * 100)
    print("完成!")
    print("=" * 100)


if __name__ == '__main__':
    main()
