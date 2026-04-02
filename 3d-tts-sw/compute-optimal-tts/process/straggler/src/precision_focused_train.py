#!/usr/bin/env python3
"""
高精度训练脚本 - 按配置和branch数分层训练以提高精确率
策略:
1. 使用全部数据(不采样)
2. 按配置单独训练(256_8_1, 16384_4_1等)
3. 按branch数单独训练(2,3,4,5,6,7,8等)
4. 按配置+branch组合训练(更精细)
5. 增加训练代数和种群规模
6. 优化目标极度侧重precision - 避免误报
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict


@dataclass
class ThresholdParams:
    """阈值参数"""
    base_ratio: float  # 基础比值阈值
    base_length: float  # 基础长度阈值
    step_factor: float  # step因子
    branch_factor: float  # branch因子
    std_factor: float  # 标准差因子
    avg_factor: float  # 平均值因子
    history_factor: float  # 历史因子
    
    def to_dict(self):
        return {
            'base_ratio': self.base_ratio,
            'base_length': self.base_length,
            'step_factor': self.step_factor,
            'branch_factor': self.branch_factor,
            'std_factor': self.std_factor,
            'avg_factor': self.avg_factor,
            'history_factor': self.history_factor
        }
    
    @staticmethod
    def from_dict(d):
        return ThresholdParams(**d)
    
    @staticmethod
    def random():
        """生成随机参数"""
        return ThresholdParams(
            base_ratio=random.uniform(2.5, 4.0),
            base_length=random.uniform(80, 150),
            step_factor=random.uniform(-0.1, 0.2),
            branch_factor=random.uniform(-0.1, 0.2),
            std_factor=random.uniform(0, 1.0),
            avg_factor=random.uniform(0, 0.5),
            history_factor=random.uniform(0, 0.3)
        )


@dataclass
class PrecisionFocusedParams(ThresholdParams):
    """高精度参数"""
    
    def to_dict(self):
        return {
            'base_ratio': self.base_ratio,
            'base_length': self.base_length,
            'step_factor': self.step_factor,
            'branch_factor': self.branch_factor,
            'std_factor': self.std_factor,
            'avg_factor': self.avg_factor,
            'history_factor': self.history_factor
        }
    
    @staticmethod
    def from_dict(d):
        return PrecisionFocusedParams(**d)


class StratifiedPrecisionDataLoader:
    """分层精度优化数据加载器"""
    
    def __init__(self):
        # 初始化所有需要的属性
        self.workloads = []
        self.stragglers = []
        self.straggler_set = set()
        self.by_question = defaultdict(list)
        
        # 分层数据
        self.by_config = defaultdict(list)
        self.by_branch = defaultdict(list)
        self.by_config_branch = defaultdict(list)
    
    def load_data(self, model_workloads_dir: str, straggler_file: str):
        """加载并分层组织数据"""
        # 加载stragglers
        with open(straggler_file, 'r') as f:
            self.stragglers = json.load(f)
        
        for s in self.stragglers:
            key = (s['question_id'], s['step_idx'])  # 注意这里用step_idx
            self.straggler_set.add(key)
        
        # 加载workloads
        self.load_from_workloads(model_workloads_dir)
        
        print("组织分层数据...")
        # 按配置、branch数分层 - 不再限制branch > 3
        for workload in self.workloads:
            config = workload['metadata']['config']
            
            for step_data in workload['steps']:
                branch_count = step_data['branch_count']
                
                # 处理所有branch_count >= 2的情况
                if branch_count >= 2:
                    self.by_config[config].append((workload, step_data))
                    self.by_branch[branch_count].append((workload, step_data))
                    key = f"{config}_b{branch_count}"
                    self.by_config_branch[key].append((workload, step_data))
        
        print(f"按配置分类: {len(self.by_config)} 个配置")
        print(f"按branch数分类: {len(self.by_branch)} 个branch数")
        print(f"按配置+branch组合: {len(self.by_config_branch)} 个组合")
        
        for config, data in sorted(self.by_config.items()):
            print(f"  配置 {config}: {len(data)} 步骤")
        
        for branch, data in sorted(self.by_branch.items()):
            print(f"  Branch {branch}: {len(data)} 步骤")
    
    def load_from_workloads(self, model_workloads_dir: str):
        """直接从workload文件加载"""
        self.workloads = []
        model_workloads_path = Path(model_workloads_dir)
        
        print(f"扫描workload文件...")
        all_files = list(model_workloads_path.rglob('*_workload.json'))
        print(f"找到 {len(all_files)} 个文件")
        
        for idx, json_file in enumerate(all_files):
            if idx % 1000 == 0:
                print(f"  加载进度: {idx}/{len(all_files)}")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                question_id = data['question_id']
                steps = data['decode']['steps']
                
                # 提取路径信息作为元数据
                parts = json_file.parts
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
                
                workload = {
                    'question_id': question_id,
                    'steps': steps,
                    'metadata': {
                        'dataset': dataset,
                        'model': model,
                        'prm': prm,
                        'config': config,
                        'file_path': str(json_file)
                    }
                }
                
                self.workloads.append(workload)
                
            except Exception as e:
                print(f"  警告: 读取文件失败 {json_file}: {e}")
                continue
        
        print(f"成功加载 {len(self.workloads)} 个workload")
    
    def get_ground_truth(self, question_id: str, step: int) -> bool:
        """获取ground truth标签"""
        key = (question_id, step)
        return key in self.straggler_set


class PrecisionOptimizer:
    """精度优先优化器 - 侧重提高precision"""
    
    def __init__(self, data_loader: StratifiedPrecisionDataLoader, 
                 subset_data: List[Tuple], subset_name: str,
                 population_size: int = 100):
        self.data_loader = data_loader
        self.subset_data = subset_data
        self.subset_name = subset_name
        self.population_size = population_size
        self.population = []
        self.best_params = None
        self.best_score = 0
        self.best_precision = 0
        self.history = []
        
        # 初始化种群
        self._initialize_population()
    
    def _initialize_population(self):
        """初始化种群 - 更平衡的参数分布"""
        self.population = []
        
        # 1. 添加一些保守(高精度)的初始参数 - 20%
        for _ in range(self.population_size // 5):
            params = ThresholdParams(
                base_ratio=np.random.uniform(3.0, 4.0),  # 较高的比值要求
                base_length=np.random.uniform(100, 140),  # 较高的长度要求
                step_factor=np.random.uniform(-0.05, 0.1),
                branch_factor=np.random.uniform(-0.15, 0.08),
                std_factor=np.random.uniform(0.0, 0.4),
                avg_factor=np.random.uniform(0.0, 0.25),
                history_factor=np.random.uniform(0.0, 0.18)
            )
            self.population.append(params)
        
        # 2. 添加平衡型参数 - 40%
        for _ in range(self.population_size // 5 * 2):
            params = ThresholdParams(
                base_ratio=np.random.uniform(2.5, 3.5),  # 中等比值要求
                base_length=np.random.uniform(80, 120),  # 中等长度要求
                step_factor=np.random.uniform(-0.1, 0.15),
                branch_factor=np.random.uniform(-0.2, 0.1),
                std_factor=np.random.uniform(0.0, 0.6),
                avg_factor=np.random.uniform(0.0, 0.35),
                history_factor=np.random.uniform(0.0, 0.25)
            )
            self.population.append(params)
        
        # 3. 添加探索性参数 - 30%
        for _ in range(self.population_size // 10 * 3):
            params = ThresholdParams(
                base_ratio=np.random.uniform(2.0, 4.5),  # 宽范围
                base_length=np.random.uniform(60, 150),  # 宽范围
                step_factor=np.random.uniform(-0.15, 0.2),
                branch_factor=np.random.uniform(-0.25, 0.15),
                std_factor=np.random.uniform(0.0, 0.8),
                avg_factor=np.random.uniform(0.0, 0.45),
                history_factor=np.random.uniform(0.0, 0.3)
            )
            self.population.append(params)
        
        # 4. 填充剩余
        for _ in range(self.population_size - len(self.population)):
            params = ThresholdParams.random()
            self.population.append(params)
    
    def evaluate_params(self, params: ThresholdParams) -> Dict:
        """评估参数 - 在子集上，使用新的straggler定义"""
        tp = fp = tn = fn = 0
        
        # 按workload组织,跟踪历史
        workload_dict = defaultdict(list)
        for workload, step_data in self.subset_data:
            question_id = workload['question_id']
            workload_dict[question_id].append((step_data, workload))
        
        for question_id, steps in workload_dict.items():
            history_max = 0.0
            
            for step_data, workload in steps:
                step = step_data['step']
                branch_count = step_data['branch_count']
                branch_tokens = step_data['branch_tokens']
                
                if branch_count == 0 or len(branch_tokens) == 0:
                    continue
                
                # 使用新的straggler检测逻辑
                is_pred = self._detect_straggler_new(
                    branch_tokens, params, step, branch_count, history_max
                )
                
                is_true = self.data_loader.get_ground_truth(question_id, step)
                
                if is_pred and is_true:
                    tp += 1
                elif is_pred and not is_true:
                    fp += 1
                elif not is_pred and is_true:
                    fn += 1
                else:
                    tn += 1
                
                if branch_tokens:
                    history_max = max(history_max, max(branch_tokens))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _detect_straggler_new(self, branch_tokens: List[int], params: ThresholdParams,
                             step: int, branch_count: int, history_max: float) -> bool:
        """
        新的straggler检测逻辑
        
        定义：对于每个分支，如果：
        1. token数量 > 100 (或动态阈值)
        2. 该分支token数 > (除自己外的最大token数) × 3 (或动态比值)
        则该分支是straggler
        
        允许有多个straggler (top2或更多的情况)
        不再限制branch_count > 3
        """
        if len(branch_tokens) == 0:
            return False
        
        # 动态计算阈值
        sorted_tokens = sorted(branch_tokens, reverse=True)
        
        # 计算动态比值阈值和长度阈值
        avg_tokens = np.mean(branch_tokens)
        std_tokens = np.std(branch_tokens) if len(branch_tokens) > 1 else 0
        
        ratio_threshold = params.base_ratio + \
                         params.step_factor * step + \
                         params.branch_factor * branch_count + \
                         params.std_factor * std_tokens / (avg_tokens + 1)
        
        length_threshold = params.base_length + \
                          params.avg_factor * avg_tokens + \
                          params.history_factor * history_max
        
        # 检查是否有straggler
        has_straggler = False
        for i, token in enumerate(sorted_tokens):
            if token <= length_threshold:
                break
            
            # 找到除当前分支外的最大值
            if i == 0:
                # 最大值，与第二大值比较
                if len(sorted_tokens) > 1:
                    second_max = sorted_tokens[1]
                    if second_max > 0 and token >= second_max * ratio_threshold:
                        has_straggler = True
                        break
            else:
                # 非最大值，与最大值比较
                max_token = sorted_tokens[0]
                if max_token > 0 and token >= max_token * ratio_threshold:
                    # 这个也是straggler (top2或更多)
                    has_straggler = True
                    break
                # 或者与除自己和比自己大的之外的最大值比较
                other_max = sorted_tokens[0] if i > 0 else (sorted_tokens[1] if len(sorted_tokens) > 1 else 0)
                if i > 0:
                    # 找到比自己小的最大值
                    for j in range(i+1, len(sorted_tokens)):
                        other_max = sorted_tokens[j]
                        break
                if other_max > 0 and token >= other_max * ratio_threshold:
                    has_straggler = True
                    break
        
        return has_straggler
    
    def fitness_function(self, metrics: Dict) -> float:
        """
        适应度函数 - 平衡Precision和Recall
        
        调整后策略：允许适当误报，提高召回率
        """
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        fp = metrics['fp']
        tp = metrics['tp']
        
        # 如果精确率太低(<0.6)，适度惩罚
        if precision < 0.6:
            return f1 * 0.6
        
        # 如果recall太低(< 0.05)，适度惩罚
        if recall < 0.05:
            recall_penalty = 0.85
        else:
            recall_penalty = 1.0
        
        # 更平衡的策略: 60% precision + 40% f1
        # 这样可以在保持高precision的同时，提高recall
        score = 0.6 * precision + 0.4 * f1
        score *= recall_penalty
        
        # 奖励高precision (>0.80)，但奖励幅度降低
        if precision > 0.80:
            score *= 1.15
        if precision > 0.85:
            score *= 1.1
        
        # 放宽误报惩罚 - 允许适当误报
        # 只在误报率很高时才惩罚
        if tp > 0:
            fp_rate = fp / (tp + fp)  # 误报率
            if fp_rate > 0.4:  # 误报率>40%才惩罚
                score *= 0.7
            elif fp_rate > 0.3:  # 误报率>30%
                score *= 0.85
        
        return score
    
    def evolve(self, generations: int = 100):
        """进化训练"""
        print(f"\n训练子集: {self.subset_name}")
        print(f"子集大小: {len(self.subset_data)} 步骤")
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {generations}")
        
        for gen in range(generations):
            # 评估种群
            results = []
            for params in self.population:
                metrics = self.evaluate_params(params)
                score = self.fitness_function(metrics)
                results.append((params, score, metrics))
            
            # 排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 更新最佳
            if results[0][1] > self.best_score:
                self.best_params = results[0][0]
                self.best_score = results[0][1]
                self.best_precision = results[0][2]['precision']
            
            # 记录历史
            self.history.append({
                'generation': gen,
                'best_fitness': results[0][1],
                'best_precision': results[0][2]['precision'],
                'best_recall': results[0][2]['recall'],
                'best_f1': results[0][2]['f1'],
                'avg_fitness': np.mean([r[1] for r in results])
            })
            
            # 打印进度
            if gen % 10 == 0 or gen == generations - 1:
                best = results[0][2]
                print(f"代 {gen:3d}: "
                      f"Score={results[0][1]:.4f}, "
                      f"P={best['precision']:.4f}, "
                      f"R={best['recall']:.4f}, "
                      f"F1={best['f1']:.4f}, "
                      f"FP={best['fp']}, "
                      f"TP={best['tp']}")
            
            # 选择和繁殖
            elite_size = self.population_size // 5
            elite = [r[0] for r in results[:elite_size]]
            
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # 锦标赛选择
                tournament_size = 5
                tournament = random.sample(results[:self.population_size // 2], 
                                         tournament_size)
                parent1 = max(tournament, key=lambda x: x[1])[0]
                
                tournament = random.sample(results[:self.population_size // 2], 
                                         tournament_size)
                parent2 = max(tournament, key=lambda x: x[1])[0]
                
                # 交叉
                child = ThresholdParams(
                    base_ratio=(parent1.base_ratio + parent2.base_ratio) / 2,
                    base_length=(parent1.base_length + parent2.base_length) / 2,
                    step_factor=(parent1.step_factor + parent2.step_factor) / 2,
                    branch_factor=(parent1.branch_factor + parent2.branch_factor) / 2,
                    std_factor=(parent1.std_factor + parent2.std_factor) / 2,
                    avg_factor=(parent1.avg_factor + parent2.avg_factor) / 2,
                    history_factor=(parent1.history_factor + parent2.history_factor) / 2
                )
                
                # 变异
                if random.random() < 0.3:
                    mutation_rate = 0.2
                    child.base_ratio += np.random.normal(0, child.base_ratio * mutation_rate)
                    child.base_length += np.random.normal(0, child.base_length * mutation_rate)
                    child.step_factor += np.random.normal(0, 0.05)
                    child.branch_factor += np.random.normal(0, 0.05)
                    child.std_factor += np.random.normal(0, 0.1)
                    child.avg_factor += np.random.normal(0, 0.05)
                    child.history_factor += np.random.normal(0, 0.05)
                    
                    # 限制范围
                    child.base_ratio = max(1.5, min(5.0, child.base_ratio))
                    child.base_length = max(30, min(200, child.base_length))
                
                new_population.append(child)
            
            self.population = new_population
        
        print(f"\n最终结果:")
        final_metrics = self.evaluate_params(self.best_params)
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall: {final_metrics['recall']:.4f}")
        print(f"  F1: {final_metrics['f1']:.4f}")
        print(f"  TP/FP/FN: {final_metrics['tp']}/{final_metrics['fp']}/{final_metrics['fn']}")


class HierarchicalPrecisionSystem:
    """分层高精度系统"""
    
    def __init__(self):
        self.global_params: Optional[ThresholdParams] = None
        self.config_params: Dict[str, ThresholdParams] = {}
        self.branch_params: Dict[int, ThresholdParams] = {}
        self.config_branch_params: Dict[str, ThresholdParams] = {}
    
    def get_params(self, config: str = None, branch_count: int = None) -> ThresholdParams:
        """获取最适合的参数"""
        # 优先级: config+branch > config > branch > global
        if config and branch_count:
            key = f"{config}_b{branch_count}"
            if key in self.config_branch_params:
                return self.config_branch_params[key]
        
        if config and config in self.config_params:
            return self.config_params[config]
        
        if branch_count and branch_count in self.branch_params:
            return self.branch_params[branch_count]
        
        if self.global_params:
            return self.global_params
        
        # 默认保守参数
        return ThresholdParams(
            base_ratio=3.0,
            base_length=100.0,
            step_factor=0.0,
            branch_factor=0.0,
            std_factor=0.5,
            avg_factor=0.2,
            history_factor=0.1
        )
    
    def save(self, filepath: str):
        """保存系统"""
        data = {
            'global_params': self.global_params.to_dict() if self.global_params else None,
            'config_params': {k: v.to_dict() for k, v in self.config_params.items()},
            'branch_params': {str(k): v.to_dict() for k, v in self.branch_params.items()},
            'config_branch_params': {k: v.to_dict() for k, v in self.config_branch_params.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> 'HierarchicalPrecisionSystem':
        """加载系统"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        system = HierarchicalPrecisionSystem()
        
        if data.get('global_params'):
            system.global_params = ThresholdParams.from_dict(data['global_params'])
        
        system.config_params = {
            k: ThresholdParams.from_dict(v) for k, v in data.get('config_params', {}).items()
        }
        system.branch_params = {
            int(k): ThresholdParams.from_dict(v) for k, v in data.get('branch_params', {}).items()
        }
        system.config_branch_params = {
            k: ThresholdParams.from_dict(v) for k, v in data.get('config_branch_params', {}).items()
        }
        
        return system


def train_precision_system(
    model_workloads_dir: str,
    straggler_file: str,
    output_dir: str,
    population_size: int = 20,      # 降低种群大小加速训练
    generations: int = 30,          # 降低代数加速训练
    train_config: bool = True,
    train_branch: bool = True,
    train_config_branch: bool = False  # 默认关闭组合级训练
):
    """训练高精度系统"""
    
    print("="*80)
    print("高精度分层训练系统")
    print("目标: 提高Precision,同时保持合理的Recall")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    data_loader = StratifiedPrecisionDataLoader()
    data_loader.load_data(model_workloads_dir, straggler_file)
    
    system = HierarchicalPrecisionSystem()
    results = {
        'config': {},
        'branch': {},
        'config_branch': {}
    }
    
    # 1. 按配置训练
    if train_config:
        print("\n" + "="*80)
        print("按配置训练 (提高配置特定精度)")
        print("="*80)
        
        for config, data in data_loader.by_config.items():
            if len(data) < 100:  # 数据太少跳过
                print(f"跳过配置 {config}: 数据不足 ({len(data)} 步骤)")
                continue
            
            optimizer = PrecisionOptimizer(
                data_loader, data, f"配置_{config}",
                population_size=population_size
            )
            optimizer.evolve(generations)
            
            system.config_params[config] = optimizer.best_params
            results['config'][config] = {
                'best_score': optimizer.best_score,
                'best_precision': optimizer.best_precision,
                'history': optimizer.history,
                'final_metrics': optimizer.evaluate_params(optimizer.best_params)
            }
    
    # 2. 按branch数训练
    if train_branch:
        print("\n" + "="*80)
        print("按Branch数训练 (针对不同分支数优化)")
        print("="*80)
        
        for branch, data in data_loader.by_branch.items():
            if len(data) < 100:
                print(f"跳过Branch {branch}: 数据不足 ({len(data)} 步骤)")
                continue
            
            optimizer = PrecisionOptimizer(
                data_loader, data, f"Branch_{branch}",
                population_size=population_size
            )
            optimizer.evolve(generations)
            
            system.branch_params[branch] = optimizer.best_params
            results['branch'][branch] = {
                'best_score': optimizer.best_score,
                'best_precision': optimizer.best_precision,
                'history': optimizer.history,
                'final_metrics': optimizer.evaluate_params(optimizer.best_params)
            }
    
    # 3. 按配置+branch组合训练 (最精细)
    if train_config_branch:
        print("\n" + "="*80)
        print("按配置+Branch组合训练 (最精细化)")
        print("="*80)
        
        for key, data in data_loader.by_config_branch.items():
            if len(data) < 50:  # 组合数据可能更少
                print(f"跳过组合 {key}: 数据不足 ({len(data)} 步骤)")
                continue
            
            optimizer = PrecisionOptimizer(
                data_loader, data, key,
                population_size=max(50, population_size // 2)  # 减少种群避免过拟合
            )
            optimizer.evolve(max(50, generations // 2))  # 减少代数
            
            system.config_branch_params[key] = optimizer.best_params
            results['config_branch'][key] = {
                'best_score': optimizer.best_score,
                'best_precision': optimizer.best_precision,
                'history': optimizer.history,
                'final_metrics': optimizer.evaluate_params(optimizer.best_params)
            }
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    system.save(str(output_path / 'precision_system.json'))
    
    with open(output_path / 'precision_results.json', 'w') as f:
        # 转换不可序列化的类型
        serializable_results = {}
        for category, items in results.items():
            serializable_results[category] = {}
            for key, value in items.items():
                serializable_results[category][str(key)] = {
                    'best_score': value['best_score'],
                    'best_precision': value['best_precision'],
                    'final_metrics': value['final_metrics']
                }
        json.dump(serializable_results, f, indent=2)
    
    # 生成报告
    generate_precision_report(system, results, str(output_path / 'precision_report.txt'))
    
    print(f"\n{'='*80}")
    print(f"训练完成! 结果已保存到: {output_path}")
    print(f"  - precision_system.json: 训练的参数")
    print(f"  - precision_results.json: 详细结果")
    print(f"  - precision_report.txt: 可读报告")
    print(f"{'='*80}")


def generate_precision_report(system: HierarchicalPrecisionSystem,
                              results: Dict,
                              output_file: str):
    """生成精度报告"""
    lines = []
    lines.append("="*80)
    lines.append("高精度分层训练报告")
    lines.append("="*80)
    
    # 按配置
    if results.get('config'):
        lines.append("\n" + "="*80)
        lines.append("按配置训练结果")
        lines.append("="*80)
        
        config_metrics = []
        for config, res in results['config'].items():
            metrics = res['final_metrics']
            config_metrics.append((config, metrics))
            lines.append(f"\n配置: {config}")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1:        {metrics['f1']:.4f}")
            lines.append(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        
        # 总结
        if config_metrics:
            avg_precision = np.mean([m[1]['precision'] for m in config_metrics])
            avg_recall = np.mean([m[1]['recall'] for m in config_metrics])
            avg_f1 = np.mean([m[1]['f1'] for m in config_metrics])
            lines.append(f"\n配置级别平均:")
            lines.append(f"  平均 Precision: {avg_precision:.4f}")
            lines.append(f"  平均 Recall:    {avg_recall:.4f}")
            lines.append(f"  平均 F1:        {avg_f1:.4f}")
    
    # 按branch
    if results.get('branch'):
        lines.append("\n" + "="*80)
        lines.append("按Branch数训练结果")
        lines.append("="*80)
        
        branch_metrics = []
        for branch, res in sorted(results['branch'].items()):
            metrics = res['final_metrics']
            branch_metrics.append((branch, metrics))
            lines.append(f"\nBranch {branch}:")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1:        {metrics['f1']:.4f}")
            lines.append(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        
        if branch_metrics:
            avg_precision = np.mean([m[1]['precision'] for m in branch_metrics])
            avg_recall = np.mean([m[1]['recall'] for m in branch_metrics])
            avg_f1 = np.mean([m[1]['f1'] for m in branch_metrics])
            lines.append(f"\nBranch级别平均:")
            lines.append(f"  平均 Precision: {avg_precision:.4f}")
            lines.append(f"  平均 Recall:    {avg_recall:.4f}")
            lines.append(f"  平均 F1:        {avg_f1:.4f}")
    
    # 按配置+branch组合
    if results.get('config_branch'):
        lines.append("\n" + "="*80)
        lines.append("按配置+Branch组合训练结果 (最精细)")
        lines.append("="*80)
        
        combo_metrics = []
        for key, res in sorted(results['config_branch'].items()):
            metrics = res['final_metrics']
            combo_metrics.append((key, metrics))
            lines.append(f"\n组合: {key}")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1:        {metrics['f1']:.4f}")
            lines.append(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        
        if combo_metrics:
            avg_precision = np.mean([m[1]['precision'] for m in combo_metrics])
            avg_recall = np.mean([m[1]['recall'] for m in combo_metrics])
            avg_f1 = np.mean([m[1]['f1'] for m in combo_metrics])
            lines.append(f"\n组合级别平均:")
            lines.append(f"  平均 Precision: {avg_precision:.4f}")
            lines.append(f"  平均 Recall:    {avg_recall:.4f}")
            lines.append(f"  平均 F1:        {avg_f1:.4f}")
    
    lines.append("\n" + "="*80)
    lines.append("训练完成!")
    lines.append("="*80)
    
    report = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print("\n" + report)


def main():
    model_workloads_dir = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sim/model_workloads"
    straggler_file = "../data/stragglers.json"
    output_dir = "../data"
    
    train_precision_system(
        model_workloads_dir=model_workloads_dir,
        straggler_file=straggler_file,
        output_dir=output_dir,
        population_size=100,      # 较大种群
        generations=150,          # 更多代数
        train_config=True,        # 按配置训练
        train_branch=True,        # 按branch数训练
        train_config_branch=True  # 按组合训练(最精细)
    )


if __name__ == "__main__":
    main()
