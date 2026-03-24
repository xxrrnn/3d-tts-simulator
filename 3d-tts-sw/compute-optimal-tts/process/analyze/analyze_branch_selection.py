#!/usr/bin/env python3
"""
分析模型的branch选择倾向
分析模型倾向于选择token数量较多还是较少的branch
"""

import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import logging
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BranchSelectionAnalyzer:
    def __init__(self, workload_dir: Path, extreme_threshold: float = 4.0):
        self.workload_dir = workload_dir
        self.extreme_threshold = extreme_threshold  # 极端值的阈值（倍数）
        
        self.results = defaultdict(lambda: {
            'total_decisions': 0,
            'select_max': 0,
            'select_min': 0,
            'select_median': 0,
            'select_other': 0,
            'token_position_distribution': defaultdict(int),
            'avg_selected_token_count': 0,
            'avg_total_branches': 0,
            'total_tokens_selected': 0,
        })
        self.config_results = defaultdict(lambda: defaultdict(lambda: {
            'total_decisions': 0,
            'select_max': 0,
            'select_min': 0,
            'select_median': 0,
            'select_other': 0,
            'token_position_distribution': defaultdict(int),
            'avg_selected_token_count': 0,
            'avg_total_branches': 0,
            'total_tokens_selected': 0,
        }))
        
        # 极端情况统计
        self.extreme_results = defaultdict(lambda: defaultdict(lambda: {
            'total_with_extreme_large': 0,
            'selected_extreme_large': 0,
            'total_with_extreme_small': 0,
            'selected_extreme_small': 0,
            'total_with_both_extremes': 0,
            'selected_extreme_in_both': 0,
            'extreme_large_details': [],  # 存储详细信息用于调试
            'extreme_small_details': [],
        }))
    
    def load_workload(self, filepath: Path) -> Dict[str, Any]:
        """加载单个workload文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def detect_extreme_branches(self, step: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """
        检测是否存在极端的branch（token数量明显偏离平均值）
        返回: (极端大的branch索引, 极端小的branch索引)
        如果不存在则返回None
        
        极端定义：某个branch的token数量 >= threshold * 其他branches的平均值
        或者：某个branch的token数量 <= (1/threshold) * 其他branches的平均值
        """
        branch_count = step.get('branch_count', 0)
        branch_tokens = step.get('branch_tokens', [])
        
        # 跳过只有0或1个分支的情况
        if branch_count <= 1:
            return None, None
        
        # 对于2个branch的情况，使用更宽松的标准
        if branch_count == 2:
            # 检查两个值的比例关系
            ratio = max(branch_tokens) / max(min(branch_tokens), 1)
            if ratio >= self.extreme_threshold:
                # 找出哪个是大的，哪个是小的
                extreme_large_idx = branch_tokens.index(max(branch_tokens))
                extreme_small_idx = branch_tokens.index(min(branch_tokens))
                return extreme_large_idx, extreme_small_idx
            return None, None
        
        # 对于3个及以上branch的情况
        extreme_large_idx = None
        extreme_small_idx = None
        
        for i, tokens in enumerate(branch_tokens):
            # 计算除了当前branch外的其他branches的平均值
            other_tokens = [t for j, t in enumerate(branch_tokens) if j != i]
            if not other_tokens:
                continue
            
            avg_others = statistics.mean(other_tokens)
            
            if avg_others > 0:
                # 检查是否是极端大的
                if tokens >= self.extreme_threshold * avg_others:
                    extreme_large_idx = i
                # 检查是否是极端小的
                elif tokens <= avg_others / self.extreme_threshold:
                    extreme_small_idx = i
        
        return extreme_large_idx, extreme_small_idx
    
    def analyze_step(self, step: Dict[str, Any]) -> Tuple[str, int, int, int]:
        """
        分析单个step的选择
        返回: (选择类型, 选中的token数, 分支总数, 选中分支的排名位置)
        """
        branch_count = step.get('branch_count', 0)
        branch_tokens = step.get('branch_tokens', [])
        selected_index = step.get('selected_branch_index', -1)
        
        # 跳过只有0或1个分支的情况
        if branch_count <= 1 or selected_index == -1:
            return None, 0, branch_count, -1
        
        selected_tokens = branch_tokens[selected_index]
        sorted_tokens = sorted(branch_tokens, reverse=True)
        
        # 找出选中token在排序后的位置（排名）
        selected_rank = sorted_tokens.index(selected_tokens)
        
        # 判断选择类型
        max_tokens = max(branch_tokens)
        min_tokens = min(branch_tokens)
        median_tokens = sorted(branch_tokens)[len(branch_tokens) // 2]
        
        if selected_tokens == max_tokens:
            selection_type = 'max'
        elif selected_tokens == min_tokens:
            selection_type = 'min'
        elif selected_tokens == median_tokens:
            selection_type = 'median'
        else:
            selection_type = 'other'
        
        return selection_type, selected_tokens, branch_count, selected_rank
    
    def analyze_workload_file(self, filepath: Path, model_key: str, config_key: str):
        """分析单个workload文件"""
        workload = self.load_workload(filepath)
        if not workload:
            return
        
        steps = workload.get('decode', {}).get('steps', [])
        
        for step in steps:
            selection_type, selected_tokens, branch_count, selected_rank = self.analyze_step(step)
            
            if selection_type is None:
                continue
            
            # 更新总体统计
            stats = self.results[model_key]
            stats['total_decisions'] += 1
            
            if selection_type == 'max':
                stats['select_max'] += 1
            elif selection_type == 'min':
                stats['select_min'] += 1
            elif selection_type == 'median':
                stats['select_median'] += 1
            else:
                stats['select_other'] += 1
            
            stats['token_position_distribution'][selected_rank] += 1
            stats['total_tokens_selected'] += selected_tokens
            stats['avg_total_branches'] += branch_count
            
            # 更新按配置分组的统计
            config_stats = self.config_results[model_key][config_key]
            config_stats['total_decisions'] += 1
            
            if selection_type == 'max':
                config_stats['select_max'] += 1
            elif selection_type == 'min':
                config_stats['select_min'] += 1
            elif selection_type == 'median':
                config_stats['select_median'] += 1
            else:
                config_stats['select_other'] += 1
            
            config_stats['token_position_distribution'][selected_rank] += 1
            config_stats['total_tokens_selected'] += selected_tokens
            config_stats['avg_total_branches'] += branch_count
            
            # 分析极端情况
            extreme_large_idx, extreme_small_idx = self.detect_extreme_branches(step)
            selected_index = step.get('selected_branch_index', -1)
            branch_tokens = step.get('branch_tokens', [])
            
            # 统计极端情况
            extreme_stats = self.extreme_results[model_key][config_key]
            
            if extreme_large_idx is not None:
                extreme_stats['total_with_extreme_large'] += 1
                if selected_index == extreme_large_idx:
                    extreme_stats['selected_extreme_large'] += 1
                    # 存储详细信息（只存储前10个样例）
                    if len(extreme_stats['extreme_large_details']) < 10:
                        extreme_stats['extreme_large_details'].append({
                            'branch_tokens': branch_tokens,
                            'selected_index': selected_index,
                            'selected_tokens': branch_tokens[selected_index]
                        })
            
            if extreme_small_idx is not None:
                extreme_stats['total_with_extreme_small'] += 1
                if selected_index == extreme_small_idx:
                    extreme_stats['selected_extreme_small'] += 1
                    # 存储详细信息（只存储前10个样例）
                    if len(extreme_stats['extreme_small_details']) < 10:
                        extreme_stats['extreme_small_details'].append({
                            'branch_tokens': branch_tokens,
                            'selected_index': selected_index,
                            'selected_tokens': branch_tokens[selected_index]
                        })
            
            # 统计同时存在两个极端的情况
            if extreme_large_idx is not None and extreme_small_idx is not None:
                extreme_stats['total_with_both_extremes'] += 1
                if selected_index in [extreme_large_idx, extreme_small_idx]:
                    extreme_stats['selected_extreme_in_both'] += 1
    
    def analyze_all_workloads(self):
        """分析所有workload文件"""
        logger.info(f"Starting analysis from: {self.workload_dir}")
        
        # 遍历所有数据集
        for dataset_dir in self.workload_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            logger.info(f"Processing dataset: {dataset_name}")
            
            # 遍历策略模型
            for policy_dir in dataset_dir.iterdir():
                if not policy_dir.is_dir():
                    continue
                
                policy_model = policy_dir.name
                
                # 遍历奖励模型
                for reward_dir in policy_dir.iterdir():
                    if not reward_dir.is_dir():
                        continue
                    
                    reward_model = reward_dir.name
                    model_key = f"{dataset_name}/{policy_model}/{reward_model}"
                    
                    logger.info(f"  Analyzing: {model_key}")
                    
                    # 遍历配置
                    for config_dir in reward_dir.iterdir():
                        if not config_dir.is_dir():
                            continue
                        
                        config_name = config_dir.name
                        config_key = f"{model_key}/{config_name}"
                        
                        # 遍历所有workload文件
                        for workload_file in config_dir.glob("*.json"):
                            self.analyze_workload_file(workload_file, model_key, config_name)
        
        # 计算平均值 - 总体统计
        for model_key, stats in self.results.items():
            if stats['total_decisions'] > 0:
                stats['avg_selected_token_count'] = stats['total_tokens_selected'] / stats['total_decisions']
                stats['avg_total_branches'] = stats['avg_total_branches'] / stats['total_decisions']
        
        # 计算平均值 - 按配置分组的统计
        for model_key, configs in self.config_results.items():
            for config_name, stats in configs.items():
                if stats['total_decisions'] > 0:
                    stats['avg_selected_token_count'] = stats['total_tokens_selected'] / stats['total_decisions']
                    stats['avg_total_branches'] = stats['avg_total_branches'] / stats['total_decisions']
    
    def print_results(self):
        """打印分析结果"""
        print("\n" + "="*100)
        print("Branch Selection Analysis Results - BY CONFIGURATION")
        print("="*100)
        
        for model_key in sorted(self.config_results.keys()):
            configs = self.config_results[model_key]
            
            print(f"\n{'═'*100}")
            print(f"Model: {model_key}")
            print(f"{'═'*100}")
            
            for config_name in sorted(configs.keys()):
                stats = configs[config_name]
                
                if stats['total_decisions'] == 0:
                    continue
                
                print(f"\n  {'─'*96}")
                print(f"  Configuration: {config_name}")
                print(f"  {'─'*96}")
                print(f"  Total decisions (branch_count > 1): {stats['total_decisions']}")
                print(f"  Average branches per decision: {stats['avg_total_branches']:.2f}")
                print(f"\n  Selection Distribution:")
                print(f"    Select MAX tokens:    {stats['select_max']:6d} ({stats['select_max']/stats['total_decisions']*100:6.2f}%)")
                print(f"    Select MIN tokens:    {stats['select_min']:6d} ({stats['select_min']/stats['total_decisions']*100:6.2f}%)")
                print(f"    Select MEDIAN tokens: {stats['select_median']:6d} ({stats['select_median']/stats['total_decisions']*100:6.2f}%)")
                print(f"    Select OTHER tokens:  {stats['select_other']:6d} ({stats['select_other']/stats['total_decisions']*100:6.2f}%)")
                
                print(f"\n  Average selected token count: {stats['avg_selected_token_count']:.2f}")
                
                # 打印排名位置分布（前5名）
                print(f"\n  Selected Branch Rank Distribution (Top 5):")
                sorted_positions = sorted(stats['token_position_distribution'].items(), key=lambda x: x[0])
                for rank, count in sorted_positions[:5]:
                    print(f"    Rank {rank+1} (largest): {count:6d} ({count/stats['total_decisions']*100:6.2f}%)")
                
                # 打印极端情况分析
                if model_key in self.extreme_results and config_name in self.extreme_results[model_key]:
                    extreme_stats = self.extreme_results[model_key][config_name]
                    
                    print(f"\n  {'─'*96}")
                    print(f"  Extreme Branch Analysis (threshold={self.extreme_threshold}x):")
                    print(f"  {'─'*96}")
                    
                    if extreme_stats['total_with_extreme_large'] > 0:
                        select_rate = extreme_stats['selected_extreme_large'] / extreme_stats['total_with_extreme_large'] * 100
                        print(f"    Extreme LARGE branches (token >> avg):")
                        print(f"      Total cases:    {extreme_stats['total_with_extreme_large']:6d}")
                        print(f"      Selected:       {extreme_stats['selected_extreme_large']:6d} ({select_rate:6.2f}%)")
                    
                    if extreme_stats['total_with_extreme_small'] > 0:
                        select_rate = extreme_stats['selected_extreme_small'] / extreme_stats['total_with_extreme_small'] * 100
                        print(f"    Extreme SMALL branches (token << avg):")
                        print(f"      Total cases:    {extreme_stats['total_with_extreme_small']:6d}")
                        print(f"      Selected:       {extreme_stats['selected_extreme_small']:6d} ({select_rate:6.2f}%)")
                    
                    if extreme_stats['total_with_both_extremes'] > 0:
                        select_rate = extreme_stats['selected_extreme_in_both'] / extreme_stats['total_with_both_extremes'] * 100
                        print(f"    Both extremes present:")
                        print(f"      Total cases:    {extreme_stats['total_with_both_extremes']:6d}")
                        print(f"      Selected extreme: {extreme_stats['selected_extreme_in_both']:6d} ({select_rate:6.2f}%)")
        
        # 同时打印总体统计（不分配置）
        print("\n\n" + "="*100)
        print("Branch Selection Analysis Results - OVERALL (All Configs Combined)")
        print("="*100)
        
        for model_key in sorted(self.results.keys()):
            stats = self.results[model_key]
            
            if stats['total_decisions'] == 0:
                continue
            
            print(f"\n{'─'*100}")
            print(f"Model: {model_key}")
            print(f"{'─'*100}")
            print(f"Total decisions (branch_count > 1): {stats['total_decisions']}")
            print(f"Average branches per decision: {stats['avg_total_branches']:.2f}")
            print(f"\nSelection Distribution:")
            print(f"  Select MAX tokens:    {stats['select_max']:6d} ({stats['select_max']/stats['total_decisions']*100:6.2f}%)")
            print(f"  Select MIN tokens:    {stats['select_min']:6d} ({stats['select_min']/stats['total_decisions']*100:6.2f}%)")
            print(f"  Select MEDIAN tokens: {stats['select_median']:6d} ({stats['select_median']/stats['total_decisions']*100:6.2f}%)")
            print(f"  Select OTHER tokens:  {stats['select_other']:6d} ({stats['select_other']/stats['total_decisions']*100:6.2f}%)")
            
            print(f"\nAverage selected token count: {stats['avg_selected_token_count']:.2f}")
    
    def save_results(self, output_file: Path):
        """保存结果到JSON文件"""
        output_data = {
            'by_configuration': {},
            'overall': {}
        }
        
        # 保存按配置分组的结果
        for model_key, configs in self.config_results.items():
            output_data['by_configuration'][model_key] = {}
            
            for config_name, stats in configs.items():
                if stats['total_decisions'] == 0:
                    continue
                
                output_data['by_configuration'][model_key][config_name] = {
                    'total_decisions': stats['total_decisions'],
                    'avg_branches_per_decision': round(stats['avg_total_branches'], 2),
                    'selection_distribution': {
                        'select_max': {
                            'count': stats['select_max'],
                            'percentage': round(stats['select_max']/stats['total_decisions']*100, 2)
                        },
                        'select_min': {
                            'count': stats['select_min'],
                            'percentage': round(stats['select_min']/stats['total_decisions']*100, 2)
                        },
                        'select_median': {
                            'count': stats['select_median'],
                            'percentage': round(stats['select_median']/stats['total_decisions']*100, 2)
                        },
                        'select_other': {
                            'count': stats['select_other'],
                            'percentage': round(stats['select_other']/stats['total_decisions']*100, 2)
                        }
                    },
                    'avg_selected_token_count': round(stats['avg_selected_token_count'], 2),
                    'rank_distribution': dict(sorted(stats['token_position_distribution'].items(), key=lambda x: x[0])[:10])
                }
                
                # 添加极端情况分析
                if model_key in self.extreme_results and config_name in self.extreme_results[model_key]:
                    extreme_stats = self.extreme_results[model_key][config_name]
                    
                    extreme_analysis = {
                        'threshold': self.extreme_threshold,
                        'extreme_large': {
                            'total_cases': extreme_stats['total_with_extreme_large'],
                            'selected_count': extreme_stats['selected_extreme_large'],
                            'selection_rate': round(
                                extreme_stats['selected_extreme_large'] / extreme_stats['total_with_extreme_large'] * 100, 2
                            ) if extreme_stats['total_with_extreme_large'] > 0 else 0,
                            'sample_cases': extreme_stats['extreme_large_details'][:5]
                        },
                        'extreme_small': {
                            'total_cases': extreme_stats['total_with_extreme_small'],
                            'selected_count': extreme_stats['selected_extreme_small'],
                            'selection_rate': round(
                                extreme_stats['selected_extreme_small'] / extreme_stats['total_with_extreme_small'] * 100, 2
                            ) if extreme_stats['total_with_extreme_small'] > 0 else 0,
                            'sample_cases': extreme_stats['extreme_small_details'][:5]
                        },
                        'both_extremes': {
                            'total_cases': extreme_stats['total_with_both_extremes'],
                            'selected_extreme_count': extreme_stats['selected_extreme_in_both'],
                            'selection_rate': round(
                                extreme_stats['selected_extreme_in_both'] / extreme_stats['total_with_both_extremes'] * 100, 2
                            ) if extreme_stats['total_with_both_extremes'] > 0 else 0
                        }
                    }
                    
                    output_data['by_configuration'][model_key][config_name]['extreme_branch_analysis'] = extreme_analysis
        
        # 保存总体统计（不分配置）
        for model_key, stats in self.results.items():
            if stats['total_decisions'] == 0:
                continue
            
            output_data['overall'][model_key] = {
                'total_decisions': stats['total_decisions'],
                'avg_branches_per_decision': round(stats['avg_total_branches'], 2),
                'selection_distribution': {
                    'select_max': {
                        'count': stats['select_max'],
                        'percentage': round(stats['select_max']/stats['total_decisions']*100, 2)
                    },
                    'select_min': {
                        'count': stats['select_min'],
                        'percentage': round(stats['select_min']/stats['total_decisions']*100, 2)
                    },
                    'select_median': {
                        'count': stats['select_median'],
                        'percentage': round(stats['select_median']/stats['total_decisions']*100, 2)
                    },
                    'select_other': {
                        'count': stats['select_other'],
                        'percentage': round(stats['select_other']/stats['total_decisions']*100, 2)
                    }
                },
                'avg_selected_token_count': round(stats['avg_selected_token_count'], 2),
                'rank_distribution': dict(sorted(stats['token_position_distribution'].items(), key=lambda x: x[0])[:10])
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析模型的branch选择倾向')
    parser.add_argument(
        '--workload-dir',
        type=str,
        default='/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads',
        help='Workload目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='branch_selection_analysis.json',
        help='输出JSON文件路径'
    )
    parser.add_argument(
        '--extreme-threshold',
        type=float,
        default=4.0,
        help='极端值的阈值倍数（默认4.0，即某个branch的token数量是其他平均值的4.0倍以上）'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    workload_dir = Path(args.workload_dir)
    output_file = Path(args.output)
    
    logger.info(f"Extreme threshold: {args.extreme_threshold}x")
    
    analyzer = BranchSelectionAnalyzer(workload_dir, extreme_threshold=args.extreme_threshold)
    analyzer.analyze_all_workloads()
    analyzer.print_results()
    analyzer.save_results(output_file)


if __name__ == "__main__":
    main()
