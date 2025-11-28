"""
词义消歧实验脚本
WSD Experiment Script

该脚本用于运行和评估词义消歧实验。
This script is used to run and evaluate WSD experiments.

使用方法 / Usage:
    python run_wsd.py --method all --max_samples 100
    python run_wsd.py --method lesk
    python run_wsd.py --method bert --dataset semcor
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_lesk_experiment(test_data: List[Dict], **kwargs) -> Dict:
    """
    运行Lesk算法实验
    Run Lesk algorithm experiment
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果字典
    """
    from wsd import LeskWSD
    from evaluation import evaluate_wsd
    
    logger.info("运行 Lesk 算法实验...")
    
    model = LeskWSD(
        use_examples=kwargs.get('use_examples', True),
        use_relations=kwargs.get('use_relations', False)
    )
    
    result = evaluate_wsd(model, test_data, verbose=True)
    
    return {
        'method': 'LeskWSD',
        'accuracy': result.accuracy,
        'f1': result.f1,
        'total': result.total,
        'correct': result.correct
    }


def run_bert_experiment(test_data: List[Dict], **kwargs) -> Dict:
    """
    运行BERT上下文实验
    Run BERT context experiment
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果字典
    """
    from wsd import BERTContextWSD
    from evaluation import evaluate_wsd
    
    logger.info("运行 BERT 上下文实验...")
    
    model = BERTContextWSD(
        model_name=kwargs.get('model_name', 'bert-base-uncased'),
        similarity_metric=kwargs.get('similarity_metric', 'cosine')
    )
    
    result = evaluate_wsd(model, test_data, verbose=True)
    
    return {
        'method': 'BERTContextWSD',
        'accuracy': result.accuracy,
        'f1': result.f1,
        'total': result.total,
        'correct': result.correct
    }


def run_knowledge_experiment(test_data: List[Dict], **kwargs) -> Dict:
    """
    运行知识增强实验
    Run knowledge-enhanced experiment
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果字典
    """
    from wsd import KnowledgeEnhancedWSD
    from evaluation import evaluate_wsd
    
    logger.info("运行知识增强实验...")
    
    model = KnowledgeEnhancedWSD(
        use_definitions=True,
        use_examples=True,
        use_hypernyms=kwargs.get('use_hypernyms', True),
        use_hyponyms=kwargs.get('use_hyponyms', True)
    )
    
    result = evaluate_wsd(model, test_data, verbose=True)
    
    return {
        'method': 'KnowledgeEnhancedWSD',
        'accuracy': result.accuracy,
        'f1': result.f1,
        'total': result.total,
        'correct': result.correct
    }


def run_graph_experiment(test_data: List[Dict], **kwargs) -> Dict:
    """
    运行基于图的实验
    Run graph-based experiment
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果字典
    """
    from wsd import GraphBasedWSD
    from evaluation import evaluate_wsd
    
    logger.info("运行基于图的实验...")
    
    model = GraphBasedWSD(
        algorithm=kwargs.get('algorithm', 'pagerank'),
        max_depth=kwargs.get('max_depth', 2)
    )
    
    result = evaluate_wsd(model, test_data, verbose=True)
    
    return {
        'method': 'GraphBasedWSD',
        'accuracy': result.accuracy,
        'f1': result.f1,
        'total': result.total,
        'correct': result.correct
    }


def run_all_experiments(test_data: List[Dict], **kwargs) -> List[Dict]:
    """
    运行所有WSD实验
    Run all WSD experiments
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果列表
    """
    results = []
    
    # Lesk
    try:
        results.append(run_lesk_experiment(test_data, **kwargs))
    except Exception as e:
        logger.error(f"Lesk实验失败: {e}")
    
    # BERT (可选，需要GPU或较长时间)
    if kwargs.get('include_bert', False):
        try:
            results.append(run_bert_experiment(test_data, **kwargs))
        except Exception as e:
            logger.error(f"BERT实验失败: {e}")
    
    # Knowledge-enhanced
    if kwargs.get('include_knowledge', False):
        try:
            results.append(run_knowledge_experiment(test_data, **kwargs))
        except Exception as e:
            logger.error(f"知识增强实验失败: {e}")
    
    # Graph-based
    try:
        results.append(run_graph_experiment(test_data, **kwargs))
    except Exception as e:
        logger.error(f"图方法实验失败: {e}")
    
    return results


def print_results_table(results: List[Dict]):
    """
    打印结果表格
    Print results table
    
    Args:
        results: 实验结果列表
    """
    print("\n" + "=" * 60)
    print("WSD 实验结果汇总")
    print("=" * 60)
    print(f"{'方法':<25} {'准确率':<12} {'F1':<12} {'样本数':<10}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True):
        print(f"{result['method']:<25} "
              f"{result.get('accuracy', 0):<12.4f} "
              f"{result.get('f1', 0):<12.4f} "
              f"{result.get('total', 0):<10}")
    
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="词义消歧实验脚本 / WSD Experiment Script"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["lesk", "bert", "knowledge", "graph", "all"],
        default="all",
        help="选择运行的方法"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        help="数据集名称 (sample, semcor)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="最大样本数量"
    )
    parser.add_argument(
        "--include_bert",
        action="store_true",
        help="是否包含BERT实验（较慢）"
    )
    parser.add_argument(
        "--include_knowledge",
        action="store_true",
        help="是否包含知识增强实验（较慢）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"加载数据集: {args.dataset}")
    from utils.data_loader import WSDDataLoader
    
    loader = WSDDataLoader()
    test_data = loader.load(dataset=args.dataset, max_samples=args.max_samples)
    
    logger.info(f"加载了 {len(test_data)} 个样本")
    
    # 运行实验
    if args.method == "all":
        results = run_all_experiments(
            test_data,
            include_bert=args.include_bert,
            include_knowledge=args.include_knowledge
        )
    elif args.method == "lesk":
        results = [run_lesk_experiment(test_data)]
    elif args.method == "bert":
        results = [run_bert_experiment(test_data)]
    elif args.method == "knowledge":
        results = [run_knowledge_experiment(test_data)]
    elif args.method == "graph":
        results = [run_graph_experiment(test_data)]
    else:
        results = []
    
    # 打印结果
    print_results_table(results)
    
    # 保存结果
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到 {args.output}")
    
    return results


if __name__ == "__main__":
    main()
