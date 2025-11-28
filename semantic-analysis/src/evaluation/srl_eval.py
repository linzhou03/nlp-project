"""
语义角色标注评估模块
SRL Evaluation Module

该模块提供语义角色标注的评估功能和指标计算。
This module provides evaluation functionality and metrics for SRL.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SRLEvaluationResult:
    """
    语义角色标注评估结果
    SRL Evaluation Result
    
    Attributes:
        precision: 精确率
        recall: 召回率
        f1: F1分数
        predicate_accuracy: 谓词识别准确率
        argument_precision: 论元精确率
        argument_recall: 论元召回率
        argument_f1: 论元F1分数
        per_role_metrics: 按角色分类的指标
        exact_match: 完全匹配率
    """
    precision: float
    recall: float
    f1: float
    predicate_accuracy: float = 0.0
    argument_precision: float = 0.0
    argument_recall: float = 0.0
    argument_f1: float = 0.0
    per_role_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    exact_match: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'predicate_accuracy': self.predicate_accuracy,
            'argument_precision': self.argument_precision,
            'argument_recall': self.argument_recall,
            'argument_f1': self.argument_f1,
            'per_role_metrics': self.per_role_metrics,
            'exact_match': self.exact_match,
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "SRL 评估结果",
            "=" * 50,
            f"精确率 (Precision): {self.precision:.4f}",
            f"召回率 (Recall): {self.recall:.4f}",
            f"F1分数: {self.f1:.4f}",
            "-" * 50,
            f"谓词识别准确率: {self.predicate_accuracy:.4f}",
            f"论元精确率: {self.argument_precision:.4f}",
            f"论元召回率: {self.argument_recall:.4f}",
            f"论元F1: {self.argument_f1:.4f}",
            f"完全匹配率: {self.exact_match:.4f}",
        ]
        
        if self.per_role_metrics:
            lines.append("-" * 50)
            lines.append("按角色分类结果:")
            for role, metrics in sorted(self.per_role_metrics.items()):
                lines.append(f"  {role}: P={metrics.get('precision', 0):.4f}, "
                           f"R={metrics.get('recall', 0):.4f}, "
                           f"F1={metrics.get('f1', 0):.4f}")
        
        lines.append("=" * 50)
        return '\n'.join(lines)


class SRLEvaluator:
    """
    语义角色标注评估器
    SRL Evaluator
    
    用于评估SRL模型的性能。
    Used to evaluate SRL model performance.
    
    评估方法：
    1. 精确率：正确预测的论元数 / 预测的论元总数
    2. 召回率：正确预测的论元数 / 正确标注的论元总数
    3. F1分数：精确率和召回率的调和平均
    
    Example:
        >>> evaluator = SRLEvaluator()
        >>> predictions = [...]  # SRLResult列表
        >>> gold_labels = [...]  # 正确的SRL标注
        >>> result = evaluator.evaluate(predictions, gold_labels)
        >>> print(result)
    """
    
    def __init__(self, strict_match: bool = True,
                 include_predicate: bool = True):
        """
        初始化评估器
        Initialize evaluator
        
        Args:
            strict_match: 是否使用严格匹配（必须完全匹配span和角色）
            include_predicate: 是否评估谓词识别
        """
        self.strict_match = strict_match
        self.include_predicate = include_predicate
    
    def evaluate(self, predictions: List, gold_labels: List[Dict]) -> SRLEvaluationResult:
        """
        评估SRL预测结果
        Evaluate SRL predictions
        
        Args:
            predictions: 预测结果 (SRLResult列表)
            gold_labels: 正确的SRL标注
            
        Returns:
            SRLEvaluationResult: 评估结果
        """
        total_pred_args = 0
        total_gold_args = 0
        correct_args = 0
        
        correct_predicates = 0
        total_predicates = 0
        
        exact_matches = 0
        total_sentences = 0
        
        # 按角色统计
        role_stats = defaultdict(lambda: {
            'pred': 0, 'gold': 0, 'correct': 0
        })
        
        for pred, gold in zip(predictions, gold_labels):
            total_sentences += 1
            
            # 提取预测的论元
            if hasattr(pred, 'arguments'):
                pred_args = pred.arguments
                pred_predicate_idx = pred.predicate_index
            else:
                pred_args = pred.get('arguments', [])
                pred_predicate_idx = pred.get('predicate_index', -1)
            
            # 提取正确的论元
            gold_args = gold.get('arguments', [])
            gold_predicate_idx = gold.get('predicate_index', -1)
            
            # 评估谓词
            total_predicates += 1
            if pred_predicate_idx == gold_predicate_idx:
                correct_predicates += 1
            
            # 评估论元
            pred_set = self._args_to_set(pred_args)
            gold_set = self._args_to_set(gold_args)
            
            total_pred_args += len(pred_set)
            total_gold_args += len(gold_set)
            
            # 计算正确的论元数
            matches = pred_set & gold_set
            correct_args += len(matches)
            
            # 按角色统计
            for arg_tuple in pred_set:
                role = arg_tuple[0]
                role_stats[role]['pred'] += 1
            
            for arg_tuple in gold_set:
                role = arg_tuple[0]
                role_stats[role]['gold'] += 1
            
            for arg_tuple in matches:
                role = arg_tuple[0]
                role_stats[role]['correct'] += 1
            
            # 检查完全匹配
            if pred_set == gold_set:
                exact_matches += 1
        
        # 计算总体指标
        precision = correct_args / total_pred_args if total_pred_args > 0 else 0.0
        recall = correct_args / total_gold_args if total_gold_args > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        predicate_accuracy = correct_predicates / total_predicates if total_predicates > 0 else 0.0
        exact_match = exact_matches / total_sentences if total_sentences > 0 else 0.0
        
        # 按角色计算指标
        per_role_metrics = {}
        for role, stats in role_stats.items():
            pred = stats['pred']
            gold = stats['gold']
            correct = stats['correct']
            
            p = correct / pred if pred > 0 else 0.0
            r = correct / gold if gold > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            per_role_metrics[role] = {
                'precision': p,
                'recall': r,
                'f1': f,
                'pred_count': pred,
                'gold_count': gold,
                'correct_count': correct
            }
        
        return SRLEvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            predicate_accuracy=predicate_accuracy,
            argument_precision=precision,
            argument_recall=recall,
            argument_f1=f1,
            per_role_metrics=per_role_metrics,
            exact_match=exact_match
        )
    
    def _args_to_set(self, args: List) -> Set[Tuple]:
        """
        将论元列表转换为可比较的集合
        Convert argument list to comparable set
        
        Args:
            args: 论元列表
            
        Returns:
            论元元组集合
        """
        result = set()
        
        for arg in args:
            if hasattr(arg, 'role'):
                role = arg.role
                span = arg.span
            else:
                role = arg.get('role', '')
                span = arg.get('span', (0, 0))
            
            if self.strict_match:
                # 严格匹配：必须匹配角色和span
                result.add((role, tuple(span)))
            else:
                # 宽松匹配：只匹配角色
                result.add((role,))
        
        return result
    
    def evaluate_bio_tags(self, pred_tags: List[List[str]], 
                          gold_tags: List[List[str]]) -> SRLEvaluationResult:
        """
        评估BIO标签
        Evaluate BIO tags
        
        Args:
            pred_tags: 预测的BIO标签列表
            gold_tags: 正确的BIO标签列表
            
        Returns:
            SRLEvaluationResult: 评估结果
        """
        total_pred = 0
        total_gold = 0
        correct = 0
        
        role_stats = defaultdict(lambda: {'pred': 0, 'gold': 0, 'correct': 0})
        
        for pred, gold in zip(pred_tags, gold_tags):
            # 从BIO标签提取论元span
            pred_spans = self._extract_spans_from_bio(pred)
            gold_spans = self._extract_spans_from_bio(gold)
            
            total_pred += len(pred_spans)
            total_gold += len(gold_spans)
            
            # 计算匹配
            for span in pred_spans:
                role = span[0]
                role_stats[role]['pred'] += 1
                
                if span in gold_spans:
                    correct += 1
                    role_stats[role]['correct'] += 1
            
            for span in gold_spans:
                role = span[0]
                role_stats[role]['gold'] += 1
        
        precision = correct / total_pred if total_pred > 0 else 0.0
        recall = correct / total_gold if total_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 按角色计算指标
        per_role_metrics = {}
        for role, stats in role_stats.items():
            p = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0.0
            r = stats['correct'] / stats['gold'] if stats['gold'] > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            per_role_metrics[role] = {
                'precision': p,
                'recall': r,
                'f1': f
            }
        
        return SRLEvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            argument_precision=precision,
            argument_recall=recall,
            argument_f1=f1,
            per_role_metrics=per_role_metrics
        )
    
    def _extract_spans_from_bio(self, tags: List[str]) -> Set[Tuple]:
        """
        从BIO标签中提取论元span
        Extract argument spans from BIO tags
        
        Args:
            tags: BIO标签列表
            
        Returns:
            论元span集合 {(role, start, end), ...}
        """
        spans = set()
        current_role = None
        current_start = -1
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                # 保存之前的span
                if current_role:
                    spans.add((current_role, current_start, i))
                
                current_role = tag[2:]  # 去掉 'B-'
                current_start = i
                
            elif tag.startswith('I-'):
                # 继续当前span
                pass
                
            else:  # 'O' 或其他
                if current_role:
                    spans.add((current_role, current_start, i))
                    current_role = None
        
        # 处理最后一个span
        if current_role:
            spans.add((current_role, current_start, len(tags)))
        
        return spans


def evaluate_srl(srl_model, test_data: List[Dict],
                 verbose: bool = True) -> SRLEvaluationResult:
    """
    便捷函数：评估SRL模型
    Convenience function to evaluate SRL model
    
    Args:
        srl_model: SRL模型实例
        test_data: 测试数据
        verbose: 是否输出详细信息
        
    Returns:
        SRLEvaluationResult: 评估结果
    
    Example:
        >>> from srl import SyntaxBasedSRL
        >>> from utils.data_loader import SRLDataLoader
        >>> 
        >>> model = SyntaxBasedSRL()
        >>> loader = SRLDataLoader()
        >>> test_data = loader.load("sample")
        >>> result = evaluate_srl(model, test_data)
        >>> print(result)
    """
    if verbose:
        logger.info(f"开始评估 {srl_model.name} ...")
    
    predictions = []
    gold_labels = []
    
    for sample in test_data:
        # 重建句子
        words = sample.get('words', [])
        sentence = ' '.join(words)
        
        # 获取预测
        try:
            results = srl_model.predict(sentence)
            
            # 找到匹配的谓词
            target_predicate_idx = sample.get('predicate_index', -1)
            matched_result = None
            
            for result in results:
                if result.predicate_index == target_predicate_idx:
                    matched_result = result
                    break
            
            if matched_result is None and results:
                matched_result = results[0]
            
            if matched_result:
                predictions.append(matched_result)
                gold_labels.append(sample)
            
        except Exception as e:
            logger.warning(f"预测失败: {e}")
    
    # 评估
    evaluator = SRLEvaluator()
    result = evaluator.evaluate(predictions, gold_labels)
    
    if verbose:
        print(result)
    
    return result


def compare_srl_methods(models: List, test_data: List[Dict],
                        verbose: bool = True) -> Dict[str, SRLEvaluationResult]:
    """
    比较多个SRL方法
    Compare multiple SRL methods
    
    Args:
        models: SRL模型列表
        test_data: 测试数据
        verbose: 是否输出详细信息
        
    Returns:
        {模型名称: 评估结果} 字典
    """
    results = {}
    
    for model in models:
        if verbose:
            logger.info(f"评估 {model.name} ...")
        
        result = evaluate_srl(model, test_data, verbose=False)
        results[model.name] = result
    
    if verbose:
        print("\n" + "=" * 70)
        print("SRL 方法对比结果")
        print("=" * 70)
        print(f"{'方法':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for name, result in sorted(results.items(), 
                                   key=lambda x: x[1].f1, 
                                   reverse=True):
            print(f"{name:<25} {result.precision:<12.4f} "
                  f"{result.recall:<12.4f} {result.f1:<12.4f}")
        
        print("=" * 70)
    
    return results
