"""
语义角色标注实验脚本
SRL Experiment Script

该脚本用于运行和评估语义角色标注实验。
This script is used to run and evaluate SRL experiments.

使用方法 / Usage:
    python run_srl.py --method all
    python run_srl.py --method syntax
    python run_srl.py --method neural
"""

import argparse
import logging
import sys
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


def run_syntax_experiment(test_data: List[Dict], **kwargs) -> Dict:
    """
    运行基于句法的SRL实验
    Run syntax-based SRL experiment
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果字典
    """
    from srl import SyntaxBasedSRL
    from evaluation import evaluate_srl
    
    logger.info("运行基于句法的SRL实验...")
    
    model = SyntaxBasedSRL()
    result = evaluate_srl(model, test_data, verbose=True)
    
    return {
        'method': 'SyntaxBasedSRL',
        'precision': result.precision,
        'recall': result.recall,
        'f1': result.f1,
        'predicate_accuracy': result.predicate_accuracy
    }


def run_neural_experiment(test_data: List[Dict], train: bool = False, **kwargs) -> Dict:
    """
    运行神经网络SRL实验
    Run neural SRL experiment
    
    Args:
        test_data: 测试数据
        train: 是否训练模型
        **kwargs: 其他参数
        
    Returns:
        实验结果字典
    """
    from srl import BiLSTMCRFSRL
    from evaluation import evaluate_srl
    
    logger.info("运行神经网络SRL实验...")
    
    model = BiLSTMCRFSRL(
        embedding_dim=kwargs.get('embedding_dim', 128),
        hidden_dim=kwargs.get('hidden_dim', 256),
        num_layers=kwargs.get('num_layers', 2),
        dropout=kwargs.get('dropout', 0.5)
    )
    
    if train:
        logger.info("训练BiLSTM-CRF模型...")
        # 使用部分测试数据作为训练数据（演示用）
        train_data = test_data[:int(len(test_data) * 0.8)]
        val_data = test_data[int(len(test_data) * 0.8):]
        
        model.train(
            train_data,
            val_data=val_data,
            epochs=kwargs.get('epochs', 10),
            batch_size=kwargs.get('batch_size', 16)
        )
    
    result = evaluate_srl(model, test_data, verbose=True)
    
    return {
        'method': 'BiLSTMCRFSRL',
        'precision': result.precision,
        'recall': result.recall,
        'f1': result.f1,
        'predicate_accuracy': result.predicate_accuracy
    }


def run_all_experiments(test_data: List[Dict], **kwargs) -> List[Dict]:
    """
    运行所有SRL实验
    Run all SRL experiments
    
    Args:
        test_data: 测试数据
        **kwargs: 其他参数
        
    Returns:
        实验结果列表
    """
    results = []
    
    # 基于句法的方法
    try:
        results.append(run_syntax_experiment(test_data, **kwargs))
    except Exception as e:
        logger.error(f"句法方法实验失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 神经网络方法（可选训练）
    if kwargs.get('train_neural', False):
        try:
            results.append(run_neural_experiment(test_data, train=True, **kwargs))
        except Exception as e:
            logger.error(f"神经网络实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def demo_srl(sentences: List[str] = None):
    """
    SRL演示
    SRL demonstration
    
    Args:
        sentences: 演示用句子列表
    """
    from srl import SyntaxBasedSRL
    
    if sentences is None:
        sentences = [
            "The cat ate the mouse in the garden.",
            "She gave him a book yesterday.",
            "The teacher asked the students to read the book.",
            "John bought a new car in Tokyo last week.",
        ]
    
    model = SyntaxBasedSRL()
    
    print("\n" + "=" * 60)
    print("SRL 演示")
    print("=" * 60)
    
    for sentence in sentences:
        print(f"\n句子: {sentence}")
        print("-" * 40)
        
        results = model.predict(sentence)
        
        for result in results:
            print(f"谓词: {result.predicate} (位置: {result.predicate_index})")
            print("论元:")
            for arg in result.arguments:
                print(f"  - [{arg.role}] {arg.text}")
        
        print("-" * 40)


def print_results_table(results: List[Dict]):
    """
    打印结果表格
    Print results table
    
    Args:
        results: 实验结果列表
    """
    print("\n" + "=" * 70)
    print("SRL 实验结果汇总")
    print("=" * 70)
    print(f"{'方法':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x.get('f1', 0), reverse=True):
        print(f"{result['method']:<25} "
              f"{result.get('precision', 0):<12.4f} "
              f"{result.get('recall', 0):<12.4f} "
              f"{result.get('f1', 0):<12.4f}")
    
    print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="语义角色标注实验脚本 / SRL Experiment Script"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["syntax", "neural", "all"],
        default="syntax",
        help="选择运行的方法"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        help="数据集名称"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示模式"
    )
    parser.add_argument(
        "--train_neural",
        action="store_true",
        help="是否训练神经网络模型"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出文件路径"
    )
    
    args = parser.parse_args()
    
    # 演示模式
    if args.demo:
        demo_srl()
        return
    
    # 加载数据
    logger.info(f"加载数据集: {args.dataset}")
    from utils.data_loader import SRLDataLoader
    
    loader = SRLDataLoader()
    test_data = loader.load(dataset=args.dataset)
    
    logger.info(f"加载了 {len(test_data)} 个样本")
    
    # 运行实验
    if args.method == "all":
        results = run_all_experiments(
            test_data,
            train_neural=args.train_neural,
            epochs=args.epochs
        )
    elif args.method == "syntax":
        results = [run_syntax_experiment(test_data)]
    elif args.method == "neural":
        results = [run_neural_experiment(test_data, train=args.train_neural, epochs=args.epochs)]
    else:
        results = []
    
    # 打印结果
    if results:
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
