"""
语义角色标注基类模块
SRL Base Module

该模块定义了语义角色标注的基类和通用数据结构。
This module defines the base class and common data structures for SRL.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# 常见的语义角色定义
# Common semantic role definitions
SEMANTIC_ROLES = {
    # 核心论元 / Core Arguments
    'ARG0': '施事者/原型施事 (Agent/Proto-Agent)',
    'ARG1': '受事者/原型受事 (Patient/Proto-Patient)',
    'ARG2': '工具/受益者/属性 (Instrument/Benefactive/Attribute)',
    'ARG3': '起点/终点 (Start/End Point)',
    'ARG4': '终点/结果 (End Point/Result)',
    'ARG5': '次要论元 (Secondary Argument)',
    
    # 附加论元 / Adjunct Arguments
    'ARGM-LOC': '地点 (Location)',
    'ARGM-TMP': '时间 (Temporal)',
    'ARGM-MNR': '方式 (Manner)',
    'ARGM-CAU': '原因 (Cause)',
    'ARGM-PRP': '目的 (Purpose)',
    'ARGM-DIR': '方向 (Direction)',
    'ARGM-EXT': '程度 (Extent)',
    'ARGM-NEG': '否定 (Negation)',
    'ARGM-MOD': '情态 (Modal)',
    'ARGM-DIS': '话语连接 (Discourse)',
    'ARGM-ADV': '副词性修饰 (Adverbial)',
    'ARGM-PRD': '次级谓词 (Secondary Predication)',
    
    # 谓词 / Predicate
    'V': '谓词 (Verb/Predicate)',
}


@dataclass
class SemanticRole:
    """
    语义角色
    Semantic Role
    
    表示一个语义角色及其对应的论元。
    Represents a semantic role and its corresponding argument.
    
    Attributes:
        role: 角色标签 (如 'ARG0', 'ARGM-TMP')
        text: 论元文本
        span: 论元在句子中的位置 (start, end)
        head_index: 论元头词的索引
    """
    role: str
    text: str
    span: Tuple[int, int]
    head_index: Optional[int] = None
    
    @property
    def role_description(self) -> str:
        """获取角色的描述"""
        return SEMANTIC_ROLES.get(self.role, self.role)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'role': self.role,
            'text': self.text,
            'span': self.span,
            'head_index': self.head_index,
            'description': self.role_description
        }
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.text}"


@dataclass
class SRLResult:
    """
    语义角色标注结果
    SRL Result
    
    存储一个谓词及其所有论元的标注结果。
    Stores annotation result for a predicate and all its arguments.
    
    Attributes:
        sentence: 原始句子
        words: 分词结果
        predicate: 谓词
        predicate_index: 谓词在句子中的索引
        arguments: 论元列表
        method: 使用的方法
    """
    sentence: str
    words: List[str]
    predicate: str
    predicate_index: int
    arguments: List[SemanticRole] = field(default_factory=list)
    method: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'sentence': self.sentence,
            'words': self.words,
            'predicate': self.predicate,
            'predicate_index': self.predicate_index,
            'arguments': [arg.to_dict() for arg in self.arguments],
            'method': self.method
        }
    
    def to_bio_tags(self) -> List[str]:
        """
        转换为BIO标注格式
        Convert to BIO tag format
        
        Returns:
            BIO标签列表
        """
        tags = ['O'] * len(self.words)
        
        # 标记谓词
        if 0 <= self.predicate_index < len(tags):
            tags[self.predicate_index] = 'B-V'
        
        # 标记论元
        for arg in self.arguments:
            start, end = arg.span
            for i in range(start, min(end, len(tags))):
                if i == start:
                    tags[i] = f'B-{arg.role}'
                else:
                    tags[i] = f'I-{arg.role}'
        
        return tags
    
    def __str__(self) -> str:
        lines = [f"句子: {self.sentence}"]
        lines.append(f"谓词: {self.predicate} (位置: {self.predicate_index})")
        lines.append("论元:")
        for arg in self.arguments:
            lines.append(f"  - {arg}")
        return '\n'.join(lines)


class SRLBase(ABC):
    """
    语义角色标注基类
    SRL Base Class
    
    所有SRL方法都应该继承这个类。
    All SRL methods should inherit from this class.
    
    Example:
        >>> class MySRL(SRLBase):
        ...     def predict(self, sentence):
        ...         # 实现语义角色标注逻辑
        ...         pass
    """
    
    def __init__(self, name: str = "BaseSRL"):
        """
        初始化SRL基类
        Initialize SRL base class
        
        Args:
            name: 方法名称
        """
        self.name = name
        self._spacy_model = None
    
    @abstractmethod
    def predict(self, sentence: str) -> List[SRLResult]:
        """
        对句子进行语义角色标注
        Perform SRL on a sentence
        
        Args:
            sentence: 输入句子
            
        Returns:
            SRLResult列表（每个谓词一个结果）
        """
        pass
    
    def predict_batch(self, sentences: List[str]) -> List[List[SRLResult]]:
        """
        批量语义角色标注
        Batch SRL prediction
        
        Args:
            sentences: 句子列表
            
        Returns:
            每个句子的SRLResult列表
        """
        return [self.predict(sentence) for sentence in sentences]
    
    def identify_predicates(self, sentence: str) -> List[Tuple[str, int]]:
        """
        识别句子中的谓词
        Identify predicates in a sentence
        
        Args:
            sentence: 输入句子
            
        Returns:
            (谓词, 位置) 列表
        """
        # 默认实现：使用spaCy识别动词
        doc = self._get_spacy_doc(sentence)
        predicates = []
        
        for i, token in enumerate(doc):
            if token.pos_ == 'VERB':
                predicates.append((token.text, i))
        
        return predicates
    
    def _get_spacy_doc(self, sentence: str):
        """
        获取spaCy文档对象
        Get spaCy document object
        
        Args:
            sentence: 输入句子
            
        Returns:
            spaCy Doc对象
        """
        if self._spacy_model is None:
            import spacy
            try:
                self._spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                # 如果模型未下载，使用空白模型
                logger.warning("spaCy模型未找到，使用空白英语模型")
                self._spacy_model = spacy.blank("en")
        
        return self._spacy_model(sentence)
    
    def tokenize(self, sentence: str) -> List[str]:
        """
        分词
        Tokenize sentence
        
        Args:
            sentence: 输入句子
            
        Returns:
            词列表
        """
        doc = self._get_spacy_doc(sentence)
        return [token.text for token in doc]
    
    def visualize(self, sentence: str, output_format: str = "text") -> str:
        """
        可视化SRL结果
        Visualize SRL results
        
        Args:
            sentence: 输入句子
            output_format: 输出格式 ("text", "html")
            
        Returns:
            格式化的输出
        """
        results = self.predict(sentence)
        
        if output_format == "text":
            return self._visualize_text(results)
        elif output_format == "html":
            return self._visualize_html(results)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    
    def _visualize_text(self, results: List[SRLResult]) -> str:
        """文本格式可视化"""
        if not results:
            return "未找到谓词和论元"
        
        lines = []
        for result in results:
            lines.append(str(result))
            lines.append("-" * 50)
        
        return '\n'.join(lines)
    
    def _visualize_html(self, results: List[SRLResult]) -> str:
        """HTML格式可视化"""
        if not results:
            return "<p>未找到谓词和论元</p>"
        
        html_parts = ['<div class="srl-results">']
        
        for result in results:
            html_parts.append('<div class="srl-result">')
            html_parts.append(f'<p><strong>句子:</strong> {result.sentence}</p>')
            html_parts.append(f'<p><strong>谓词:</strong> <span class="predicate">{result.predicate}</span></p>')
            html_parts.append('<ul class="arguments">')
            
            for arg in result.arguments:
                html_parts.append(
                    f'<li><span class="role">{arg.role}</span>: {arg.text}</li>'
                )
            
            html_parts.append('</ul>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
