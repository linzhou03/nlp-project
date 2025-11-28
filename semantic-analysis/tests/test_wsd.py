"""
词义消歧单元测试
WSD Unit Tests

该模块包含WSD模块的单元测试。
This module contains unit tests for the WSD module.
"""

import pytest
import sys
from pathlib import Path

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestWSDBase:
    """WSD基类测试"""
    
    def test_import(self):
        """测试模块导入"""
        from wsd import WSDBase, WSDResult
        assert WSDBase is not None
        assert WSDResult is not None
    
    def test_wsd_result_creation(self):
        """测试WSDResult创建"""
        from wsd import WSDResult
        
        result = WSDResult(
            word="bank",
            sense_key="bank.n.01",
            definition="a financial institution",
            confidence=0.85,
            method="TestWSD"
        )
        
        assert result.word == "bank"
        assert result.sense_key == "bank.n.01"
        assert result.confidence == 0.85
    
    def test_wsd_result_to_dict(self):
        """测试WSDResult转换为字典"""
        from wsd import WSDResult
        
        result = WSDResult(
            word="bank",
            sense_key="bank.n.01",
            definition="a financial institution",
            confidence=0.85,
            method="TestWSD"
        )
        
        result_dict = result.to_dict()
        
        assert 'word' in result_dict
        assert 'sense_key' in result_dict
        assert 'confidence' in result_dict


class TestLeskWSD:
    """Lesk算法测试"""
    
    def test_init(self):
        """测试初始化"""
        from wsd import LeskWSD
        
        lesk = LeskWSD()
        assert lesk.name == "LeskWSD"
    
    def test_disambiguate_bank_financial(self):
        """测试消歧：bank（金融机构）"""
        from wsd import LeskWSD
        
        lesk = LeskWSD()
        result = lesk.disambiguate(
            context="I went to the bank to deposit money.",
            target_word="bank"
        )
        
        assert result.word == "bank"
        assert result.sense_key != "unknown"
        assert 0 <= result.confidence <= 1
    
    def test_disambiguate_bank_river(self):
        """测试消歧：bank（河岸）"""
        from wsd import LeskWSD
        
        lesk = LeskWSD()
        result = lesk.disambiguate(
            context="The river bank was covered with flowers.",
            target_word="bank"
        )
        
        assert result.word == "bank"
        assert result.sense_key != "unknown"
    
    def test_disambiguate_unknown_word(self):
        """测试未知词"""
        from wsd import LeskWSD
        
        lesk = LeskWSD()
        result = lesk.disambiguate(
            context="This is a test sentence.",
            target_word="xyzabc123"  # 不存在的词
        )
        
        assert result.sense_key == "unknown"
    
    def test_get_signature(self):
        """测试获取词义签名"""
        from wsd import LeskWSD
        
        lesk = LeskWSD()
        synsets = lesk.get_word_senses("bank", pos='n')
        
        if synsets:
            signature = lesk.get_signature(synsets[0])
            assert isinstance(signature, list)
            assert len(signature) > 0


class TestBERTContextWSD:
    """BERT上下文WSD测试"""
    
    @pytest.mark.slow
    def test_init(self):
        """测试初始化"""
        from wsd import BERTContextWSD
        
        bert_wsd = BERTContextWSD()
        assert bert_wsd.name == "BERTContextWSD"
    
    @pytest.mark.slow
    def test_disambiguate(self):
        """测试消歧（需要加载模型，较慢）"""
        from wsd import BERTContextWSD
        
        bert_wsd = BERTContextWSD()
        result = bert_wsd.disambiguate(
            context="I deposited money at the bank.",
            target_word="bank"
        )
        
        assert result.word == "bank"
        assert result.sense_key != "unknown"


class TestKnowledgeEnhancedWSD:
    """知识增强WSD测试"""
    
    @pytest.mark.slow
    def test_init(self):
        """测试初始化"""
        from wsd import KnowledgeEnhancedWSD
        
        ke_wsd = KnowledgeEnhancedWSD()
        assert ke_wsd.name == "KnowledgeEnhancedWSD"
    
    @pytest.mark.slow
    def test_get_extended_gloss(self):
        """测试获取扩展gloss"""
        from wsd import KnowledgeEnhancedWSD
        
        ke_wsd = KnowledgeEnhancedWSD()
        synsets = ke_wsd.get_word_senses("bank", pos='n')
        
        if synsets:
            gloss = ke_wsd.get_extended_gloss(synsets[0])
            assert isinstance(gloss, str)
            assert len(gloss) > 0


class TestGraphBasedWSD:
    """基于图的WSD测试"""
    
    def test_init(self):
        """测试初始化"""
        from wsd import GraphBasedWSD
        
        graph_wsd = GraphBasedWSD()
        assert graph_wsd.name == "GraphBasedWSD"
    
    def test_build_local_graph(self):
        """测试构建局部图"""
        from wsd import GraphBasedWSD
        
        graph_wsd = GraphBasedWSD()
        synsets = graph_wsd.get_word_senses("bank")
        
        if synsets:
            graph = graph_wsd.build_local_graph(synsets, depth=1)
            assert isinstance(graph, dict)
    
    def test_disambiguate(self):
        """测试消歧"""
        from wsd import GraphBasedWSD
        
        graph_wsd = GraphBasedWSD()
        result = graph_wsd.disambiguate(
            context="The bank is near the river.",
            target_word="bank"
        )
        
        assert result.word == "bank"
        assert result.sense_key != "unknown"


class TestWSDEvaluation:
    """WSD评估测试"""
    
    def test_evaluator_init(self):
        """测试评估器初始化"""
        from evaluation import WSDEvaluator
        
        evaluator = WSDEvaluator()
        assert evaluator is not None
    
    def test_evaluate_perfect(self):
        """测试完美预测的评估"""
        from evaluation import WSDEvaluator
        from wsd import WSDResult
        
        evaluator = WSDEvaluator()
        
        predictions = [
            WSDResult("bank", "bank.n.01", "definition", 0.9, "test"),
            WSDResult("star", "star.n.01", "definition", 0.8, "test"),
        ]
        gold_labels = ["bank.n.01", "star.n.01"]
        
        result = evaluator.evaluate(predictions, gold_labels)
        
        assert result.accuracy == 1.0
        assert result.correct == 2
    
    def test_evaluate_imperfect(self):
        """测试部分正确预测的评估"""
        from evaluation import WSDEvaluator
        from wsd import WSDResult
        
        evaluator = WSDEvaluator()
        
        predictions = [
            WSDResult("bank", "bank.n.01", "definition", 0.9, "test"),
            WSDResult("bank", "bank.n.02", "definition", 0.8, "test"),
        ]
        gold_labels = ["bank.n.01", "bank.n.01"]
        
        result = evaluator.evaluate(predictions, gold_labels)
        
        assert result.accuracy == 0.5
        assert result.correct == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
