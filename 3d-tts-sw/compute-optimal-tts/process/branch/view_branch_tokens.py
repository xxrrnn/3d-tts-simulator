#!/usr/bin/env python3
"""
快速查看工具 - 查询特定问题或branch count的token信息
"""

import json
import sys
from pathlib import Path


def load_data():
    """加载JSON数据"""
    json_path = Path(__file__).parent / 'branch_tokens_data_16384_8_1.json'
    with open(json_path, 'r') as f:
        return json.load(f)


def display_step(step, show_detail=True):
    """显示单个步骤的信息"""
    print(f"  Step {step['step_index']} ({step['branch_count']} branches):")
    
    tokens = step['branch_tokens']
    print(f"    Tokens: {tokens}")
    print(f"      - 总计: {sum(tokens)} tokens")
    print(f"      - 平均: {sum(tokens)/len(tokens):.2f}")
    print(f"      - 范围: [{min(tokens)}, {max(tokens)}]")
    print(f"      - 选中: Branch {step['selected_branch_index']} ({tokens[step['selected_branch_index']]} tokens)")
    
    if show_detail and step['branch_rewards']:
        rewards = step['branch_rewards']
        print(f"    Rewards: [{', '.join([f'{r:.4f}' for r in rewards])}]")
        print(f"      - 选中reward: {rewards[step['selected_branch_index']]:.4f}")
    print()


def query_by_question(data, question_id):
    """查询特定问题的所有步骤"""
    print(f"\n{'='*80}")
    print(f"问题: {question_id}")
    print(f"{'='*80}\n")
    
    found = False
    for bc in ['8', '7', '6', '5']:
        steps = [s for s in data['steps_by_branch_count'][bc] if s['question_id'] == question_id]
        if steps:
            found = True
            print(f"--- {bc} Branches ({len(steps)} 个步骤) ---\n")
            for step in sorted(steps, key=lambda x: x['step_index']):
                display_step(step)
    
    if not found:
        print(f"未找到 {question_id} 的数据")


def query_by_branch_count(data, branch_count):
    """查询特定branch count的统计信息"""
    bc_str = str(branch_count)
    steps = data['steps_by_branch_count'][bc_str]
    
    print(f"\n{'='*80}")
    print(f"Branch Count = {branch_count}")
    print(f"{'='*80}\n")
    
    print(f"总步骤数: {len(steps)}")
    
    all_tokens = []
    for step in steps:
        all_tokens.extend(step['branch_tokens'])
    
    print(f"\nToken统计:")
    print(f"  - 总Token数: {sum(all_tokens)}")
    print(f"  - 平均Token: {sum(all_tokens)/len(all_tokens):.2f}")
    print(f"  - 最小Token: {min(all_tokens)}")
    print(f"  - 最大Token: {max(all_tokens)}")
    print(f"  - 中位数Token: {sorted(all_tokens)[len(all_tokens)//2]}")
    
    # 按问题分组
    from collections import defaultdict
    steps_by_q = defaultdict(list)
    for step in steps:
        steps_by_q[step['question_id']].append(step)
    
    print(f"\n问题分布: {len(steps_by_q)} 个问题")
    for qid in sorted(steps_by_q.keys()):
        print(f"  - {qid}: {len(steps_by_q[qid])} 步")


def list_all_questions(data):
    """列出所有问题"""
    questions = set()
    for bc in ['8', '7', '6', '5']:
        for step in data['steps_by_branch_count'][bc]:
            questions.add(step['question_id'])
    
    print(f"\n所有问题列表 (共 {len(questions)} 个):\n")
    for qid in sorted(questions):
        # 统计该问题的步数
        counts = {}
        for bc in ['8', '7', '6', '5']:
            count = len([s for s in data['steps_by_branch_count'][bc] if s['question_id'] == qid])
            if count > 0:
                counts[bc] = count
        
        counts_str = ', '.join([f"{bc}-br:{cnt}" for bc, cnt in counts.items()])
        print(f"  {qid:15s} [{counts_str}]")


def show_statistics(data):
    """显示总体统计"""
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}\n")
    
    print(f"配置: 16384_8_1")
    print(f"生成时间: {data['metadata']['generated_at']}")
    print(f"总步骤数: {data['metadata']['total_steps']}\n")
    
    print(f"{'Branch Count':<15} | {'步数':<8} | {'总Tokens':<12} | {'平均Token':<12}")
    print("-" * 60)
    
    for bc in ['8', '7', '6', '5']:
        steps = data['steps_by_branch_count'][bc]
        all_tokens = []
        for step in steps:
            all_tokens.extend(step['branch_tokens'])
        
        avg_token = sum(all_tokens) / len(all_tokens) if all_tokens else 0
        print(f"{bc:<15} | {len(steps):<8} | {sum(all_tokens):<12} | {avg_token:<12.2f}")


def main():
    if len(sys.argv) < 2:
        print("Branch Token快速查看工具")
        print("\n用法:")
        print("  python view_branch_tokens.py <选项>")
        print("\n选项:")
        print("  stats              - 显示总体统计")
        print("  list               - 列出所有问题")
        print("  q <问题ID>         - 查询特定问题 (如: q question_0)")
        print("  bc <branch数量>    - 查询特定branch count (如: bc 8)")
        print("\n示例:")
        print("  python view_branch_tokens.py stats")
        print("  python view_branch_tokens.py list")
        print("  python view_branch_tokens.py q question_0")
        print("  python view_branch_tokens.py bc 8")
        return
    
    data = load_data()
    command = sys.argv[1].lower()
    
    if command == 'stats':
        show_statistics(data)
    
    elif command == 'list':
        list_all_questions(data)
    
    elif command == 'q' or command == 'question':
        if len(sys.argv) < 3:
            print("错误: 请指定问题ID")
            print("示例: python view_branch_tokens.py q question_0")
            return
        question_id = sys.argv[2]
        if not question_id.startswith('question_'):
            question_id = f'question_{question_id}'
        query_by_question(data, question_id)
    
    elif command == 'bc' or command == 'branch':
        if len(sys.argv) < 3:
            print("错误: 请指定branch count")
            print("示例: python view_branch_tokens.py bc 8")
            return
        try:
            branch_count = int(sys.argv[2])
            if branch_count not in [8, 7, 6, 5]:
                print(f"错误: branch count必须是8, 7, 6或5")
                return
            query_by_branch_count(data, branch_count)
        except ValueError:
            print("错误: branch count必须是数字")
            return
    
    else:
        print(f"未知命令: {command}")
        print("使用 'python view_branch_tokens.py' 查看帮助")


if __name__ == '__main__':
    main()
