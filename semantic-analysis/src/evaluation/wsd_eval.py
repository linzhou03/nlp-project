"""
词义消歧评估模块
WSD Evaluation Module

该模块提供词义消歧的评估功能和指标计算。
This module provides evaluation functionality and metrics for WSD.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WSDEvaluationResult:
    """
    词义消歧评估结果
    WSD Evaluation Result
    
    Attributes:
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1: F1分数
        total: 总样本数
        correct: 正确预测数
        per_pos_metrics: 按词性分类的指标
        per_word_metrics: 按词分类的指标
        confusion_matrix: 混淆矩阵
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    total: int
    correct: int
    per_pos_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_word_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'total': self.total,
            'correct': self.correct,
            'per_pos_metrics': self.per_pos_metrics,
            'per_word_metrics': self.per_word_metrics,
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "WSD 评估结果",
            "=" * 50,
            f"总样本数: {self.total}",
            f"正确预测: {self.correct}",
            f"准确率 (Accuracy): {self.accuracy:.4f}",
            f"精确率 (Precision): {self.precision:.4f}",
            f"召回率 (Recall): {self.recall:.4f}",
            f"F1分数: {self.f1:.4f}",
        ]
        
        if self.per_pos_metrics:
            lines.append("-" * 50)
            lines.append("按词性分类结果:")
            for pos, metrics in self.per_pos_metrics.items():
                lines.append(f"  {pos}: Acc={metrics.get('accuracy', 0):.4f}, "
                           f"F1={metrics.get('f1', 0):.4f}")
        
        lines.append("=" * 50)
        return '\n'.join(lines)


class WSDEvaluator:
    """
    词义消歧评估器
    WSD Evaluator
    
    用于评估WSD模型的性能。
    Used to evaluate WSD model performance.
    
    Example:
        >>> evaluator = WSDEvaluator()
        >>> predictions = [...]  # WSDResult列表
        >>> gold_labels = [...]  # 正确的词义键列表
        >>> result = evaluator.evaluate(predictions, gold_labels)
        >>> print(result)
    """
    
    def __init__(self, ignore_case: bool = True, 
                 match_mode: str = "exact"):
        """
        初始化评估器
        Initialize evaluator
        
        Args:
            ignore_case: 是否忽略大小写
            match_mode: 匹配模式 ("exact", "lemma", "first_sense")
        """
        self.ignore_case = ignore_case
        self.match_mode = match_mode
    
    def evaluate(self, predictions: List, gold_labels: List[str],
                 words: Optional[List[str]] = None,
                 pos_tags: Optional[List[str]] = None) -> WSDEvaluationResult:
        """
        评估WSD预测结果
        Evaluate WSD predictions
        
        Args:
            predictions: 预测结果 (WSDResult列表或词义键列表)
            gold_labels: 正确的词义键列表
            words: 目标词列表（用于按词统计）
            pos_tags: 词性列表（用于按词性统计）
            
        Returns:
            WSDEvaluationResult: 评估结果
        """
        # 提取预测的词义键
        pred_senses = []
        for pred in predictions:
            if hasattr(pred, 'sense_key'):
                pred_senses.append(pred.sense_key)
            else:
                pred_senses.append(str(pred))
        
        # 基本统计
        total = len(gold_labels)
        correct = 0
        
        for pred, gold in zip(pred_senses, gold_labels):
            if self._match(pred, gold):
                correct += 1
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0.0
        
        # 计算精确率、召回率、F1（对于多类分类，这里使用微平均）
        precision = accuracy  # 对于WSD，微平均精确率等于准确率
        recall = accuracy     # 微平均召回率也等于准确率
        f1 = accuracy         # 微平均F1也等于准确率
        
        # 按词性统计
        per_pos_metrics = {}
        if pos_tags:
            per_pos_metrics = self._compute_per_pos_metrics(
                pred_senses, gold_labels, pos_tags
            )
        
        # 按词统计
        per_word_metrics = {}
        if words:
            per_word_metrics = self._compute_per_word_metrics(
                pred_senses, gold_labels, words
            )
        
        return WSDEvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            total=total,
            correct=correct,
            per_pos_metrics=per_pos_metrics,
            per_word_metrics=per_word_metrics
        )
    
    def _match(self, pred: str, gold: str) -> bool:
        """
        检查预测是否匹配正确答案
        Check if prediction matches gold label
        
        Args:
            pred: 预测的词义键
            gold: 正确的词义键
            
        Returns:
            是否匹配
        """
        if self.ignore_case:
            pred = pred.lower()
            gold = gold.lower()
        
        if self.match_mode == "exact":
            return pred == gold
        elif self.match_mode == "lemma":
            # 只比较词元部分 (如 "bank.n.01" -> "bank")
            pred_lemma = pred.split('.')[0] if '.' in pred else pred
            gold_lemma = gold.split('.')[0] if '.' in gold else gold
            return pred_lemma == gold_lemma
        elif self.match_mode == "first_sense":
            # 如果预测了任何词义，认为正确（用于基线评估）
            return pred != "unknown" and gold != "unknown"
        else:
            return pred == gold
    
    def _compute_per_pos_metrics(self, predictions: List[str], 
                                  gold_labels: List[str],
                                  pos_tags: List[str]) -> Dict[str, Dict[str, float]]:
        """
        计算按词性分类的指标
        Compute metrics per POS tag
        
        Args:
            predictions: 预测列表
            gold_labels: 正确标签列表
            pos_tags: 词性列表
            
        Returns:
            按词性分类的指标字典
        """
        pos_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred, gold, pos in zip(predictions, gold_labels, pos_tags):
            pos_stats[pos]['total'] += 1
            if self._match(pred, gold):
                pos_stats[pos]['correct'] += 1
        
        metrics = {}
        for pos, stats in pos_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = correct / total if total > 0 else 0.0
            metrics[pos] = {
                'accuracy': accuracy,
                'f1': accuracy,  # 对于单个类别，F1等于准确率
                'total': total,
                'correct': correct
            }
        
        return metrics
    
    def _compute_per_word_metrics(self, predictions: List[str],
                                   gold_labels: List[str],
                                   words: List[str]) -> Dict[str, Dict[str, float]]:
        """
        计算按词分类的指标
        Compute metrics per word
        
        Args:
            predictions: 预测列表
            gold_labels: 正确标签列表
            words: 目标词列表
            
        Returns:
            按词分类的指标字典
        """
        word_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred, gold, word in zip(predictions, gold_labels, words):
            word_lower = word.lower()
            word_stats[word_lower]['total'] += 1
            if self._match(pred, gold):
                word_stats[word_lower]['correct'] += 1
        
        metrics = {}
        for word, stats in word_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = correct / total if total > 0 else 0.0
            metrics[word] = {
                'accuracy': accuracy,
                'f1': accuracy,
                'total': total,
                'correct': correct
            }
        
        return metrics
    
    def compute_macro_f1(self, predictions: List[str], 
                         gold_labels: List[str]) -> float:
        """
        计算宏平均F1分数
        Compute macro-averaged F1 score
        
        Args:
            predictions: 预测列表
            gold_labels: 正确标签列表
            
        Returns:
            宏平均F1分数
        """
        # 收集所有唯一的词义
        all_senses = set(gold_labels) | set(predictions)
        
        f1_scores = []
        for sense in all_senses:
            # 计算每个词义的精确率和召回率
            tp = sum(1 for p, g in zip(predictions, gold_labels) 
                    if p == sense and g == sense)
            fp = sum(1 for p, g in zip(predictions, gold_labels) 
                    if p == sense and g != sense)
            fn = sum(1 for p, g in zip(predictions, gold_labels) 
                    if p != sense and g == sense)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0


def evaluate_wsd(wsd_model, test_data: List[Dict], 
                 verbose: bool = True) -> WSDEvaluationResult:
    """
    便捷函数：评估WSD模型
    Convenience function to evaluate WSD model
    
    Args:
        wsd_model: WSD模型实例
        test_data: 测试数据
        verbose: 是否输出详细信息
        
    Returns:
        WSDEvaluationResult: 评估结果
    
    Example:
        >>> from wsd import LeskWSD
        >>> from utils.data_loader import WSDDataLoader
        >>> 
        >>> model = LeskWSD()
        >>> loader = WSDDataLoader()
        >>> test_data = loader.load("semcor", max_samples=100)
        >>> result = evaluate_wsd(model, test_data)
        >>> print(result)
    """
    if verbose:
        logger.info(f"开始评估 {wsd_model.name} ...")
    
    predictions = []
    gold_labels = []
    words = []
    
    for sample in test_data:
        context = sample.get('context') or sample.get('sentence', '')
        target_word = sample.get('target_word', '')
        gold_sense = sample.get('sense_key', '')
        
        # 获取预测
        try:
            result = wsd_model.disambiguate(
                context=context,
                target_word=target_word,
                target_position=sample.get('target_position'),
                pos=sample.get('pos')
            )
            predictions.append(result)
            gold_labels.append(gold_sense)
            words.append(target_word)
        except Exception as e:
            logger.warning(f"预测失败: {e}")
    
    # 评估
    evaluator = WSDEvaluator()
    result = evaluator.evaluate(predictions, gold_labels, words=words)
    
    if verbose:
        print(result)
    
    return result


def compare_wsd_methods(models: List, test_data: List[Dict],
                        verbose: bool = True) -> Dict[str, WSDEvaluationResult]:
    """
    比较多个WSD方法
    Compare multiple WSD methods
    
    Args:
        models: WSD模型列表
        test_data: 测试数据
        verbose: 是否输出详细信息
        
    Returns:
        {模型名称: 评估结果} 字典
    """
    results = {}
    
    for model in models:
        if verbose:
            logger.info(f"评估 {model.name} ...")
        
        result = evaluate_wsd(model, test_data, verbose=False)
        results[model.name] = result
    
    if verbose:
        print("\n" + "=" * 60)
        print("WSD 方法对比结果")
        print("=" * 60)
        print(f"{'方法':<25} {'准确率':<12} {'F1':<12}")
        print("-" * 60)
        
        for name, result in sorted(results.items(), 
                                   key=lambda x: x[1].accuracy, 
                                   reverse=True):
            print(f"{name:<25} {result.accuracy:<12.4f} {result.f1:<12.4f}")
        
        print("=" * 60)
    
    return results
