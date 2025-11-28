"""
语义角色标注单元测试
SRL Unit Tests

该模块包含SRL模块的单元测试。
This module contains unit tests for the SRL module.
"""

import pytest
import sys
from pathlib import Path

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSRLBase:
    """SRL基类测试"""
    
    def test_import(self):
        """测试模块导入"""
        from srl import SRLBase, SRLResult, SemanticRole
        assert SRLBase is not None
        assert SRLResult is not None
        assert SemanticRole is not None
    
    def test_semantic_role_creation(self):
        """测试SemanticRole创建"""
        from srl import SemanticRole
        
        role = SemanticRole(
            role="ARG0",
            text="The cat",
            span=(0, 2),
            head_index=1
        )
        
        assert role.role == "ARG0"
        assert role.text == "The cat"
        assert role.span == (0, 2)
    
    def test_semantic_role_description(self):
        """测试获取角色描述"""
        from srl import SemanticRole
        
        role = SemanticRole(
            role="ARG0",
            text="The cat",
            span=(0, 2)
        )
        
        description = role.role_description
        assert "施事者" in description or "Agent" in description
    
    def test_srl_result_creation(self):
        """测试SRLResult创建"""
        from srl import SRLResult, SemanticRole
        
        args = [
            SemanticRole("ARG0", "The cat", (0, 2)),
            SemanticRole("ARG1", "the mouse", (3, 5)),
        ]
        
        result = SRLResult(
            sentence="The cat ate the mouse.",
            words=["The", "cat", "ate", "the", "mouse", "."],
            predicate="ate",
            predicate_index=2,
            arguments=args,
            method="TestSRL"
        )
        
        assert result.predicate == "ate"
        assert len(result.arguments) == 2
    
    def test_srl_result_to_bio_tags(self):
        """测试SRLResult转换为BIO标签"""
        from srl import SRLResult, SemanticRole
        
        args = [
            SemanticRole("ARG0", "The cat", (0, 2)),
            SemanticRole("ARG1", "the mouse", (3, 5)),
        ]
        
        result = SRLResult(
            sentence="The cat ate the mouse.",
            words=["The", "cat", "ate", "the", "mouse", "."],
            predicate="ate",
            predicate_index=2,
            arguments=args,
            method="TestSRL"
        )
        
        tags = result.to_bio_tags()
        
        assert tags[0] == "B-ARG0"
        assert tags[1] == "I-ARG0"
        assert tags[2] == "B-V"
        assert tags[3] == "B-ARG1"
        assert tags[4] == "I-ARG1"
        assert tags[5] == "O"


class TestSyntaxBasedSRL:
    """基于句法的SRL测试"""
    
    def test_init(self):
        """测试初始化"""
        from srl import SyntaxBasedSRL
        
        srl = SyntaxBasedSRL()
        assert srl.name == "SyntaxBasedSRL"
    
    def test_predict_simple_sentence(self):
        """测试简单句子的预测"""
        from srl import SyntaxBasedSRL
        
        srl = SyntaxBasedSRL()
        results = srl.predict("The cat ate the mouse.")
        
        assert len(results) > 0
        assert results[0].predicate == "ate"
    
    def test_predict_with_location(self):
        """测试带地点的句子"""
        from srl import SyntaxBasedSRL
        
        srl = SyntaxBasedSRL()
        results = srl.predict("The cat ate the fish in the garden.")
        
        assert len(results) > 0
        
        # 检查是否有地点论元
        has_location = False
        for result in results:
            for arg in result.arguments:
                if 'LOC' in arg.role:
                    has_location = True
                    break
    
    def test_predict_passive_sentence(self):
        """测试被动句"""
        from srl import SyntaxBasedSRL
        
        srl = SyntaxBasedSRL()
        results = srl.predict("The mouse was eaten by the cat.")
        
        assert len(results) > 0
    
    def test_get_dependency_tree(self):
        """测试获取依存树"""
        from srl import SyntaxBasedSRL
        
        srl = SyntaxBasedSRL()
        tree = srl.get_dependency_tree("The cat ate the mouse.")
        
        assert 'sentence' in tree
        assert 'nodes' in tree
        assert len(tree['nodes']) > 0
    
    def test_visualize(self):
        """测试可视化"""
        from srl import SyntaxBasedSRL
        
        srl = SyntaxBasedSRL()
        output = srl.visualize("The cat ate the mouse.")
        
        assert isinstance(output, str)
        assert len(output) > 0


class TestBiLSTMCRFSRL:
    """BiLSTM-CRF SRL测试"""
    
    def test_init(self):
        """测试初始化"""
        from srl import BiLSTMCRFSRL
        
        srl = BiLSTMCRFSRL()
        assert srl.name == "BiLSTMCRFSRL"
        assert srl.is_trained is False
    
    def test_predict_without_training(self):
        """测试未训练时的预测（应该回退到句法方法）"""
        from srl import BiLSTMCRFSRL
        
        srl = BiLSTMCRFSRL()
        results = srl.predict("The cat ate the mouse.")
        
        # 未训练时会回退到句法方法
        assert len(results) > 0
    
    @pytest.mark.slow
    def test_train_and_predict(self):
        """测试训练和预测"""
        from srl import BiLSTMCRFSRL
        
        # 创建简单的训练数据
        train_data = [
            {
                'words': ['The', 'cat', 'ate', 'fish', '.'],
                'predicate': 'ate',
                'predicate_index': 2,
                'arguments': [
                    {'role': 'ARG0', 'span': (0, 2), 'text': 'The cat'},
                    {'role': 'ARG1', 'span': (3, 4), 'text': 'fish'},
                ]
            },
            {
                'words': ['She', 'gave', 'him', 'a', 'book', '.'],
                'predicate': 'gave',
                'predicate_index': 1,
                'arguments': [
                    {'role': 'ARG0', 'span': (0, 1), 'text': 'She'},
                    {'role': 'ARG1', 'span': (3, 5), 'text': 'a book'},
                    {'role': 'ARG2', 'span': (2, 3), 'text': 'him'},
                ]
            },
        ]
        
        srl = BiLSTMCRFSRL(embedding_dim=32, hidden_dim=64, num_layers=1)
        srl.train(train_data, epochs=2, batch_size=2)
        
        assert srl.is_trained is True
        
        results = srl.predict("The dog chased the cat.")
        assert len(results) > 0


class TestSRLEvaluation:
    """SRL评估测试"""
    
    def test_evaluator_init(self):
        """测试评估器初始化"""
        from evaluation import SRLEvaluator
        
        evaluator = SRLEvaluator()
        assert evaluator is not None
    
    def test_evaluate_perfect(self):
        """测试完美预测的评估"""
        from evaluation import SRLEvaluator
        from srl import SRLResult, SemanticRole
        
        evaluator = SRLEvaluator()
        
        # 创建预测和真实标签
        pred_args = [
            SemanticRole("ARG0", "The cat", (0, 2)),
            SemanticRole("ARG1", "the mouse", (3, 5)),
        ]
        
        prediction = SRLResult(
            sentence="The cat ate the mouse.",
            words=["The", "cat", "ate", "the", "mouse", "."],
            predicate="ate",
            predicate_index=2,
            arguments=pred_args
        )
        
        gold = {
            'predicate_index': 2,
            'arguments': [
                {'role': 'ARG0', 'span': (0, 2)},
                {'role': 'ARG1', 'span': (3, 5)},
            ]
        }
        
        result = evaluator.evaluate([prediction], [gold])
        
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
    
    def test_evaluate_imperfect(self):
        """测试部分正确预测的评估"""
        from evaluation import SRLEvaluator
        from srl import SRLResult, SemanticRole
        
        evaluator = SRLEvaluator()
        
        # 预测少了一个论元
        pred_args = [
            SemanticRole("ARG0", "The cat", (0, 2)),
        ]
        
        prediction = SRLResult(
            sentence="The cat ate the mouse.",
            words=["The", "cat", "ate", "the", "mouse", "."],
            predicate="ate",
            predicate_index=2,
            arguments=pred_args
        )
        
        gold = {
            'predicate_index': 2,
            'arguments': [
                {'role': 'ARG0', 'span': (0, 2)},
                {'role': 'ARG1', 'span': (3, 5)},
            ]
        }
        
        result = evaluator.evaluate([prediction], [gold])
        
        assert result.precision == 1.0  # 预测的都正确
        assert result.recall == 0.5     # 只预测了一半
    
    def test_evaluate_bio_tags(self):
        """测试BIO标签评估"""
        from evaluation import SRLEvaluator
        
        evaluator = SRLEvaluator()
        
        pred_tags = [
            ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'O']
        ]
        gold_tags = [
            ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'O']
        ]
        
        result = evaluator.evaluate_bio_tags(pred_tags, gold_tags)
        
        assert result.f1 == 1.0


class TestDataLoader:
    """数据加载器测试"""
    
    def test_wsd_loader_sample(self):
        """测试WSD数据加载器"""
        from utils.data_loader import WSDDataLoader
        
        loader = WSDDataLoader()
        data = loader.load(dataset="sample")
        
        assert len(data) > 0
        assert 'sentence' in data[0]
        assert 'target_word' in data[0]
    
    def test_srl_loader_sample(self):
        """测试SRL数据加载器"""
        from utils.data_loader import SRLDataLoader
        
        loader = SRLDataLoader()
        data = loader.load(dataset="sample")
        
        assert len(data) > 0
        assert 'words' in data[0]
        assert 'predicate' in data[0]
    
    def test_wsd_get_word_senses(self):
        """测试获取词义"""
        from utils.data_loader import WSDDataLoader
        
        loader = WSDDataLoader()
        senses = loader.get_word_senses("bank")
        
        assert len(senses) > 0
        assert 'sense_key' in senses[0]
        assert 'definition' in senses[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
