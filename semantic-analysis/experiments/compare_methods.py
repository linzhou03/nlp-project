"""
方法对比脚本
Method Comparison Script

该脚本用于对比不同WSD和SRL方法的性能。
This script is used to compare the performance of different WSD and SRL methods.

使用方法 / Usage:
    python compare_methods.py --task wsd
    python compare_methods.py --task srl
    python compare_methods.py --task all
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_wsd_methods(test_data: List[Dict], 
                        methods: List[str] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    比较WSD方法
    Compare WSD methods
    
    Args:
        test_data: 测试数据
        methods: 要比较的方法列表
        **kwargs: 其他参数
        
    Returns:
        比较结果字典
    """
    from wsd import LeskWSD, BERTContextWSD, KnowledgeEnhancedWSD, GraphBasedWSD
    from evaluation.wsd_eval import compare_wsd_methods as compare_wsd
    
    if methods is None:
        methods = ['lesk', 'graph']
    
    models = []
    
    if 'lesk' in methods:
        models.append(LeskWSD())
    
    if 'lesk_extended' in methods:
        models.append(LeskWSD(use_examples=True, use_relations=True))
    
    if 'bert' in methods:
        models.append(BERTContextWSD())
    
    if 'knowledge' in methods:
        models.append(KnowledgeEnhancedWSD())
    
    if 'graph' in methods:
        models.append(GraphBasedWSD())
    
    if not models:
        logger.warning("没有选择任何方法")
        return {}
    
    results = compare_wsd(models, test_data, verbose=True)
    
    # 转换为可序列化的格式
    output = {}
    for name, result in results.items():
        output[name] = result.to_dict()
    
    return output


def compare_srl_methods(test_data: List[Dict],
                        methods: List[str] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    比较SRL方法
    Compare SRL methods
    
    Args:
        test_data: 测试数据
        methods: 要比较的方法列表
        **kwargs: 其他参数
        
    Returns:
        比较结果字典
    """
    from srl import SyntaxBasedSRL, BiLSTMCRFSRL
    from evaluation.srl_eval import compare_srl_methods as compare_srl
    
    if methods is None:
        methods = ['syntax']
    
    models = []
    
    if 'syntax' in methods:
        models.append(SyntaxBasedSRL())
    
    if 'neural' in methods:
        model = BiLSTMCRFSRL()
        if kwargs.get('train_neural', False):
            train_data = test_data[:int(len(test_data) * 0.8)]
            model.train(train_data, epochs=kwargs.get('epochs', 5))
        models.append(model)
    
    if not models:
        logger.warning("没有选择任何方法")
        return {}
    
    results = compare_srl(models, test_data, verbose=True)
    
    # 转换为可序列化的格式
    output = {}
    for name, result in results.items():
        output[name] = result.to_dict()
    
    return output


def analyze_wsd_results(results: Dict[str, Any], 
                        test_data: List[Dict]) -> Dict:
    """
    分析WSD结果
    Analyze WSD results
    
    Args:
        results: WSD比较结果
        test_data: 测试数据
        
    Returns:
        分析结果
    """
    analysis = {
        'summary': {},
        'per_method_analysis': {},
        'recommendations': []
    }
    
    # 找出最佳方法
    best_method = None
    best_accuracy = 0
    
    for method, result in results.items():
        accuracy = result.get('accuracy', 0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = method
        
        analysis['per_method_analysis'][method] = {
            'accuracy': accuracy,
            'f1': result.get('f1', 0),
            'correct': result.get('correct', 0),
            'total': result.get('total', 0)
        }
    
    analysis['summary']['best_method'] = best_method
    analysis['summary']['best_accuracy'] = best_accuracy
    analysis['summary']['total_methods'] = len(results)
    
    # 生成建议
    if best_accuracy < 0.5:
        analysis['recommendations'].append(
            "准确率较低，建议：1) 增加训练数据 2) 尝试使用预训练模型 3) 结合多种方法"
        )
    elif best_accuracy < 0.7:
        analysis['recommendations'].append(
            "准确率中等，可以尝试使用知识增强方法或BERT进一步提升"
        )
    else:
        analysis['recommendations'].append(
            "准确率较好，可以考虑模型融合或针对特定词类型优化"
        )
    
    return analysis


def analyze_srl_results(results: Dict[str, Any],
                        test_data: List[Dict]) -> Dict:
    """
    分析SRL结果
    Analyze SRL results
    
    Args:
        results: SRL比较结果
        test_data: 测试数据
        
    Returns:
        分析结果
    """
    analysis = {
        'summary': {},
        'per_method_analysis': {},
        'per_role_analysis': {},
        'recommendations': []
    }
    
    # 找出最佳方法
    best_method = None
    best_f1 = 0
    
    for method, result in results.items():
        f1 = result.get('f1', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_method = method
        
        analysis['per_method_analysis'][method] = {
            'precision': result.get('precision', 0),
            'recall': result.get('recall', 0),
            'f1': f1
        }
        
        # 按角色分析
        per_role = result.get('per_role_metrics', {})
        for role, metrics in per_role.items():
            if role not in analysis['per_role_analysis']:
                analysis['per_role_analysis'][role] = {}
            analysis['per_role_analysis'][role][method] = metrics.get('f1', 0)
    
    analysis['summary']['best_method'] = best_method
    analysis['summary']['best_f1'] = best_f1
    analysis['summary']['total_methods'] = len(results)
    
    # 生成建议
    if best_f1 < 0.5:
        analysis['recommendations'].append(
            "F1分数较低，建议：1) 检查数据质量 2) 增加训练数据 3) 调整模型超参数"
        )
    elif best_f1 < 0.7:
        analysis['recommendations'].append(
            "F1分数中等，可以尝试使用BERT微调或增加数据增强"
        )
    else:
        analysis['recommendations'].append(
            "F1分数较好，可以针对特定角色类型进行优化"
        )
    
    return analysis


def generate_report(wsd_results: Dict, srl_results: Dict,
                    wsd_analysis: Dict, srl_analysis: Dict,
                    output_path: str = None) -> str:
    """
    生成实验报告
    Generate experiment report
    
    Args:
        wsd_results: WSD结果
        srl_results: SRL结果
        wsd_analysis: WSD分析
        srl_analysis: SRL分析
        output_path: 输出路径
        
    Returns:
        报告文本
    """
    report = []
    report.append("# 语义分析方法对比报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # WSD部分
    if wsd_results:
        report.append("\n## 词义消歧 (WSD) 结果\n")
        report.append("| 方法 | 准确率 | F1分数 |")
        report.append("|------|--------|--------|")
        
        for method, result in sorted(wsd_results.items(), 
                                     key=lambda x: x[1].get('accuracy', 0),
                                     reverse=True):
            acc = result.get('accuracy', 0)
            f1 = result.get('f1', 0)
            report.append(f"| {method} | {acc:.4f} | {f1:.4f} |")
        
        if wsd_analysis:
            report.append(f"\n**最佳方法**: {wsd_analysis['summary']['best_method']}")
            report.append(f"**最高准确率**: {wsd_analysis['summary']['best_accuracy']:.4f}")
            
            report.append("\n### 建议")
            for rec in wsd_analysis.get('recommendations', []):
                report.append(f"- {rec}")
    
    # SRL部分
    if srl_results:
        report.append("\n## 语义角色标注 (SRL) 结果\n")
        report.append("| 方法 | Precision | Recall | F1 |")
        report.append("|------|-----------|--------|-----|")
        
        for method, result in sorted(srl_results.items(),
                                     key=lambda x: x[1].get('f1', 0),
                                     reverse=True):
            p = result.get('precision', 0)
            r = result.get('recall', 0)
            f1 = result.get('f1', 0)
            report.append(f"| {method} | {p:.4f} | {r:.4f} | {f1:.4f} |")
        
        if srl_analysis:
            report.append(f"\n**最佳方法**: {srl_analysis['summary']['best_method']}")
            report.append(f"**最高F1**: {srl_analysis['summary']['best_f1']:.4f}")
            
            report.append("\n### 建议")
            for rec in srl_analysis.get('recommendations', []):
                report.append(f"- {rec}")
    
    report.append("\n## 总结\n")
    report.append("本报告对比了多种词义消歧和语义角色标注方法的性能。")
    report.append("根据实验结果，建议根据具体应用场景选择合适的方法。")
    
    report_text = '\n'.join(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"报告已保存到 {output_path}")
    
    return report_text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="方法对比脚本 / Method Comparison Script"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["wsd", "srl", "all"],
        default="all",
        help="选择任务类型"
    )
    parser.add_argument(
        "--wsd_methods",
        type=str,
        nargs="+",
        default=["lesk", "graph"],
        help="WSD方法列表"
    )
    parser.add_argument(
        "--srl_methods",
        type=str,
        nargs="+",
        default=["syntax"],
        help="SRL方法列表"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="最大样本数量"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="报告输出路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON结果输出路径"
    )
    
    args = parser.parse_args()
    
    wsd_results = {}
    srl_results = {}
    wsd_analysis = {}
    srl_analysis = {}
    
    # WSD对比
    if args.task in ["wsd", "all"]:
        logger.info("加载WSD数据...")
        from utils.data_loader import WSDDataLoader
        
        loader = WSDDataLoader()
        wsd_data = loader.load(dataset="sample", max_samples=args.max_samples)
        
        logger.info(f"WSD数据: {len(wsd_data)} 样本")
        
        wsd_results = compare_wsd_methods(wsd_data, methods=args.wsd_methods)
        wsd_analysis = analyze_wsd_results(wsd_results, wsd_data)
    
    # SRL对比
    if args.task in ["srl", "all"]:
        logger.info("加载SRL数据...")
        from utils.data_loader import SRLDataLoader
        
        loader = SRLDataLoader()
        srl_data = loader.load(dataset="sample")
        
        logger.info(f"SRL数据: {len(srl_data)} 样本")
        
        srl_results = compare_srl_methods(srl_data, methods=args.srl_methods)
        srl_analysis = analyze_srl_results(srl_results, srl_data)
    
    # 生成报告
    if args.report:
        report = generate_report(
            wsd_results, srl_results,
            wsd_analysis, srl_analysis,
            output_path=args.report
        )
        print("\n" + report)
    
    # 保存JSON结果
    if args.output:
        output_data = {
            'wsd_results': wsd_results,
            'srl_results': srl_results,
            'wsd_analysis': wsd_analysis,
            'srl_analysis': srl_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到 {args.output}")
    
    return {
        'wsd': wsd_results,
        'srl': srl_results
    }


if __name__ == "__main__":
    main()
