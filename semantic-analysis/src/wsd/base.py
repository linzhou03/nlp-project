"""
词义消歧基类模块
WSD Base Module

该模块定义了词义消歧的基类和通用数据结构。
This module defines the base class and common data structures for WSD.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class WSDResult:
    """
    词义消歧结果
    WSD Result
    
    存储词义消歧的结果信息。
    Stores WSD result information.
    
    Attributes:
        word: 目标词 / Target word
        sense_key: 词义键 (如 'bank.n.01') / Sense key
        definition: 词义定义 / Sense definition
        confidence: 置信度 (0-1) / Confidence score
        method: 使用的方法 / Method used
        context: 原始上下文 / Original context
        all_senses: 所有候选词义及其分数 / All candidate senses with scores
    """
    word: str
    sense_key: str
    definition: str
    confidence: float
    method: str
    context: Optional[str] = None
    all_senses: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        Convert to dictionary
        
        Returns:
            结果字典
        """
        return {
            'word': self.word,
            'sense_key': self.sense_key,
            'definition': self.definition,
            'confidence': self.confidence,
            'method': self.method,
            'context': self.context,
            'all_senses': self.all_senses
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"WSDResult(word='{self.word}', sense='{self.sense_key}', "
                f"confidence={self.confidence:.3f})")


class WSDBase(ABC):
    """
    词义消歧基类
    WSD Base Class
    
    所有WSD方法都应该继承这个类。
    All WSD methods should inherit from this class.
    
    Example:
        >>> class MyWSD(WSDBase):
        ...     def disambiguate(self, context, target_word, target_pos=None):
        ...         # 实现词义消歧逻辑
        ...         pass
    """
    
    def __init__(self, name: str = "BaseWSD"):
        """
        初始化WSD基类
        Initialize WSD base class
        
        Args:
            name: 方法名称
        """
        self.name = name
        self._nltk_initialized = False
        self._wordnet = None
    
    def _init_nltk(self):
        """
        初始化NLTK资源
        Initialize NLTK resources
        """
        if not self._nltk_initialized:
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            try:
                nltk.data.find('corpora/omw-1.4')
            except LookupError:
                nltk.download('omw-1.4', quiet=True)
            self._nltk_initialized = True
    
    @property
    def wordnet(self):
        """
        获取WordNet
        Get WordNet
        """
        if self._wordnet is None:
            self._init_nltk()
            from nltk.corpus import wordnet
            self._wordnet = wordnet
        return self._wordnet
    
    @abstractmethod
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        执行词义消歧
        Perform word sense disambiguation
        
        Args:
            context: 上下文句子 / Context sentence
            target_word: 目标词 / Target word
            target_position: 目标词在句子中的位置（词索引）/ Target word position
            pos: 词性 (n, v, a, r) / Part of speech
            
        Returns:
            WSDResult: 消歧结果
        """
        pass
    
    def disambiguate_batch(self, samples: List[Dict]) -> List[WSDResult]:
        """
        批量词义消歧
        Batch word sense disambiguation
        
        Args:
            samples: 样本列表，每个样本包含:
                - context/sentence: 上下文句子
                - target_word: 目标词
                - target_position (可选): 目标词位置
                - pos (可选): 词性
                
        Returns:
            WSDResult列表
        """
        results = []
        for sample in samples:
            context = sample.get('context') or sample.get('sentence', '')
            target_word = sample.get('target_word', '')
            target_position = sample.get('target_position')
            pos = sample.get('pos')
            
            try:
                result = self.disambiguate(
                    context=context,
                    target_word=target_word,
                    target_position=target_position,
                    pos=pos
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"消歧失败 '{target_word}': {e}")
                # 返回默认结果
                results.append(WSDResult(
                    word=target_word,
                    sense_key="unknown",
                    definition="",
                    confidence=0.0,
                    method=self.name,
                    context=context
                ))
        
        return results
    
    def get_word_senses(self, word: str, pos: Optional[str] = None) -> List[Any]:
        """
        获取词的所有词义（Synsets）
        Get all senses (Synsets) for a word
        
        Args:
            word: 目标词
            pos: 词性
            
        Returns:
            Synset列表
        """
        return self.wordnet.synsets(word, pos=pos)
    
    def get_sense_definition(self, sense_key: str) -> str:
        """
        获取词义的定义
        Get definition of a sense
        
        Args:
            sense_key: 词义键 (如 'bank.n.01')
            
        Returns:
            定义文本
        """
        try:
            synset = self.wordnet.synset(sense_key)
            return synset.definition()
        except Exception:
            return ""
    
    def get_sense_examples(self, sense_key: str) -> List[str]:
        """
        获取词义的例句
        Get examples of a sense
        
        Args:
            sense_key: 词义键
            
        Returns:
            例句列表
        """
        try:
            synset = self.wordnet.synset(sense_key)
            return synset.examples()
        except Exception:
            return []
    
    def get_sense_relations(self, sense_key: str) -> Dict[str, List[str]]:
        """
        获取词义的语义关系
        Get semantic relations of a sense
        
        Args:
            sense_key: 词义键
            
        Returns:
            关系字典 {'hypernyms': [...], 'hyponyms': [...], ...}
        """
        try:
            synset = self.wordnet.synset(sense_key)
            return {
                'hypernyms': [h.name() for h in synset.hypernyms()],
                'hyponyms': [h.name() for h in synset.hyponyms()],
                'holonyms': [h.name() for h in synset.member_holonyms()],
                'meronyms': [m.name() for m in synset.part_meronyms()],
            }
        except Exception:
            return {}
    
    def __repr__(self) -> str:
        """类的字符串表示"""
        return f"{self.__class__.__name__}(name='{self.name}')"
