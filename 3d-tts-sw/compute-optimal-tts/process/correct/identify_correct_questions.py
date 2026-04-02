#!/usr/bin/env python3
"""
识别题目正确性的脚本
从输出目录中提取每个题目的 majority_vote 结果，判断题目是否做对
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def parse_record_file(record_file: Path) -> Dict:
    """
    解析 record JSONL 文件，提取最后一行的结果
    
    Args:
        record_file: record_0.jsonl 文件路径
        
    Returns:
        包含题目信息的字典
    """
    try:
        with open(record_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return None
            
            last_line = lines[-1]
            data = json.loads(last_line)
            
            result = data.get('result', {})
            
            return {
                'question': data.get('question', 'N/A'),
                'groundtruth': data.get('groundtruth', 'N/A'),
                'majority_vote': result.get('majority_vote', None),
                'total_completion_tokens': result.get('total_completion_tokens', 0),
                'output': data.get('output', [])
            }
    except Exception as e:
        print(f"Error parsing {record_file}: {e}", file=sys.stderr)
        return None


def analyze_output_directory(output_dir: Path) -> Tuple[List[int], List[int], Dict]:
    """
    分析输出目录，识别所有题目的正确性
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        (正确题目列表, 错误题目列表, 详细信息字典)
    """
    correct_questions = []
    incorrect_questions = []
    details = {}
    
    # 遍历所有 question_N 目录
    question_dirs = sorted([d for d in output_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('question_')])
    
    for question_dir in question_dirs:
        # 提取题号
        try:
            question_num = int(question_dir.name.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Cannot parse question number from {question_dir.name}", 
                  file=sys.stderr)
            continue
        
        # 查找 record_0.jsonl 文件
        record_file = question_dir / 'record_0.jsonl'
        if not record_file.exists():
            print(f"Warning: {record_file} not found", file=sys.stderr)
            continue
        
        # 解析记录文件
        result = parse_record_file(record_file)
        if result is None:
            continue
        
        details[question_num] = result
        
        # 判断正确性
        majority_vote = result.get('majority_vote')
        if majority_vote == 1 or majority_vote == 1.0:
            correct_questions.append(question_num)
        else:
            incorrect_questions.append(question_num)
    
    return correct_questions, incorrect_questions, details


def print_summary(correct_questions: List[int], incorrect_questions: List[int], 
                 total: int):
    """打印统计摘要"""
    print("=" * 70)
    print("题目正确性统计")
    print("=" * 70)
    
    print(f"\n✅ 做对的题目 (majority_vote=1):")
    print(f"   题号: {correct_questions}")
    print(f"   总计: {len(correct_questions)}/{total}")
    print(f"   正确率: {len(correct_questions)/total*100:.2f}%")
    
    print(f"\n❌ 做错的题目 (majority_vote=0):")
    print(f"   题号: {incorrect_questions}")
    print(f"   总计: {len(incorrect_questions)}/{total}")
    
    print("\n" + "=" * 70)


def save_results(output_file: Path, correct_questions: List[int], 
                incorrect_questions: List[int], details: Dict):
    """保存结果到 JSON 文件"""
    results = {
        'summary': {
            'correct_count': len(correct_questions),
            'incorrect_count': len(incorrect_questions),
            'total': len(correct_questions) + len(incorrect_questions),
            'accuracy': len(correct_questions) / (len(correct_questions) + len(incorrect_questions))
                       if (len(correct_questions) + len(incorrect_questions)) > 0 else 0
        },
        'correct_questions': correct_questions,
        'incorrect_questions': incorrect_questions,
        'details': {str(k): v for k, v in sorted(details.items())}
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='识别题目正确性，从输出目录中提取 majority_vote 结果'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='输出目录路径 (例如: src/output/AIME24_beam_search/...)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='结果输出文件路径 (默认: 当前目录下的 correct_questions.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    # 检查输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: 输出目录不存在: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not output_dir.is_dir():
        print(f"Error: 路径不是目录: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 分析目录
    print(f"正在分析目录: {output_dir}")
    correct_questions, incorrect_questions, details = analyze_output_directory(output_dir)
    
    total = len(correct_questions) + len(incorrect_questions)
    if total == 0:
        print("Error: 未找到任何题目数据", file=sys.stderr)
        sys.exit(1)
    
    # 打印摘要
    print_summary(correct_questions, incorrect_questions, total)
    
    # 显示详细信息
    if args.verbose:
        print("\n详细信息:")
        for q_num in sorted(details.keys()):
            info = details[q_num]
            status = "✅" if info['majority_vote'] == 1.0 else "❌"
            print(f"\n{status} Question {q_num}:")
            print(f"   Majority Vote: {info['majority_vote']}")
            print(f"   Total Tokens: {info['total_completion_tokens']}")
            if len(str(info['groundtruth'])) < 200:
                print(f"   Groundtruth: {info['groundtruth'][:200]}...")
    
    # 保存结果
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path('correct_questions.json')
    
    save_results(output_file, correct_questions, incorrect_questions, details)


if __name__ == '__main__':
    main()
