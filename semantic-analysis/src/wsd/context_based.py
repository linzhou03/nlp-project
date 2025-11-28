"""
基于上下文的词义消歧模块
Context-based WSD Module

该模块实现了基于上下文的词义消歧方法，包括：
1. Lesk算法及其变体
2. 基于BERT的上下文表示方法
3. 基于KNN的词义分类器

This module implements context-based WSD methods including:
1. Lesk algorithm and its variants
2. BERT-based contextual representation methods
3. KNN-based sense classifier
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import numpy as np

from .base import WSDBase, WSDResult

logger = logging.getLogger(__name__)


class ContextBasedWSD(WSDBase):
    """
    基于上下文的词义消歧基类
    Context-based WSD Base Class
    
    所有基于上下文方法的WSD实现都应该继承这个类。
    All context-based WSD implementations should inherit from this class.
    """
    
    def __init__(self, context_window: int = 50, name: str = "ContextBasedWSD"):
        """
        初始化基于上下文的WSD
        Initialize context-based WSD
        
        Args:
            context_window: 上下文窗口大小（词数）
            name: 方法名称
        """
        super().__init__(name=name)
        self.context_window = context_window
        self._stopwords = None
    
    @property
    def stopwords(self) -> set:
        """
        获取停用词集合
        Get stopwords set
        """
        if self._stopwords is None:
            self._init_nltk()
            import nltk
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words('english'))
        return self._stopwords
    
    def get_context_words(self, context: str, target_position: Optional[int] = None,
                          remove_stopwords: bool = True) -> List[str]:
        """
        获取上下文词
        Get context words
        
        Args:
            context: 上下文句子
            target_position: 目标词位置
            remove_stopwords: 是否移除停用词
            
        Returns:
            上下文词列表
        """
        words = context.lower().split()
        
        # 如果指定了位置，取窗口内的词
        if target_position is not None:
            start = max(0, target_position - self.context_window)
            end = min(len(words), target_position + self.context_window + 1)
            words = words[start:target_position] + words[target_position + 1:end]
        
        # 移除停用词和标点
        if remove_stopwords:
            words = [w for w in words if w.isalnum() and w not in self.stopwords]
        
        return words


class LeskWSD(ContextBasedWSD):
    """
    Lesk词义消歧算法
    Lesk Word Sense Disambiguation Algorithm
    
    基于词义定义与上下文词重叠的经典方法。
    A classic method based on overlap between sense definitions and context words.
    
    Example:
        >>> lesk = LeskWSD()
        >>> result = lesk.disambiguate(
        ...     "I went to the bank to deposit money",
        ...     "bank"
        ... )
        >>> print(result.definition)
    """
    
    def __init__(self, context_window: int = 50, use_examples: bool = True,
                 use_relations: bool = False):
        """
        初始化Lesk算法
        Initialize Lesk algorithm
        
        Args:
            context_window: 上下文窗口大小
            use_examples: 是否使用例句扩展定义
            use_relations: 是否使用语义关系扩展定义
        """
        super().__init__(context_window=context_window, name="LeskWSD")
        self.use_examples = use_examples
        self.use_relations = use_relations
    
    def get_signature(self, synset) -> List[str]:
        """
        获取词义的签名（定义词集合）
        Get signature (definition word set) of a sense
        
        Args:
            synset: WordNet Synset对象
            
        Returns:
            签名词列表
        """
        # 获取定义词
        signature_words = synset.definition().lower().split()
        
        # 添加例句词
        if self.use_examples:
            for example in synset.examples():
                signature_words.extend(example.lower().split())
        
        # 添加相关词义的词
        if self.use_relations:
            for hypernym in synset.hypernyms():
                signature_words.extend(hypernym.definition().lower().split())
            for hyponym in synset.hyponyms()[:3]:  # 限制数量
                signature_words.extend(hyponym.definition().lower().split())
        
        # 添加同义词
        for lemma in synset.lemmas():
            signature_words.append(lemma.name().lower().replace('_', ' '))
        
        # 清洗：移除停用词和标点
        signature_words = [
            w for w in signature_words 
            if w.isalnum() and w not in self.stopwords
        ]
        
        return signature_words
    
    def compute_overlap(self, context_words: List[str], 
                        signature_words: List[str]) -> float:
        """
        计算上下文词与签名词的重叠度
        Compute overlap between context words and signature words
        
        Args:
            context_words: 上下文词列表
            signature_words: 签名词列表
            
        Returns:
            重叠分数
        """
        context_set = set(context_words)
        signature_set = set(signature_words)
        
        overlap = context_set.intersection(signature_set)
        
        # 返回重叠词数量作为分数
        return len(overlap)
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        使用Lesk算法进行词义消歧
        Perform WSD using Lesk algorithm
        
        Args:
            context: 上下文句子
            target_word: 目标词
            target_position: 目标词位置
            pos: 词性
            
        Returns:
            WSDResult: 消歧结果
        """
        # 获取所有候选词义
        synsets = self.get_word_senses(target_word, pos=pos)
        
        if not synsets:
            return WSDResult(
                word=target_word,
                sense_key="unknown",
                definition=f"No senses found for '{target_word}'",
                confidence=0.0,
                method=self.name,
                context=context
            )
        
        # 获取上下文词
        context_words = self.get_context_words(context, target_position)
        
        # 计算每个词义的重叠分数
        best_sense = synsets[0]
        best_score = 0.0
        all_senses = []
        
        for synset in synsets:
            signature = self.get_signature(synset)
            score = self.compute_overlap(context_words, signature)
            
            all_senses.append({
                'sense_key': synset.name(),
                'definition': synset.definition(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_sense = synset
        
        # 计算置信度（归一化分数）
        max_score = max(s['score'] for s in all_senses) if all_senses else 0
        confidence = best_score / (max_score + 1) if max_score > 0 else 1.0 / len(synsets)
        
        return WSDResult(
            word=target_word,
            sense_key=best_sense.name(),
            definition=best_sense.definition(),
            confidence=min(confidence, 1.0),
            method=self.name,
            context=context,
            all_senses=all_senses
        )


class BERTContextWSD(ContextBasedWSD):
    """
    基于BERT的上下文词义消歧
    BERT-based Contextual WSD
    
    使用BERT模型获取目标词的上下文表示，然后与词义定义向量进行相似度计算。
    Uses BERT to get contextual representation of target word, then computes
    similarity with sense definition vectors.
    
    Example:
        >>> bert_wsd = BERTContextWSD()
        >>> result = bert_wsd.disambiguate(
        ...     "I deposited money at the bank",
        ...     "bank"
        ... )
    """
    
    def __init__(self, model_name: str = "bert-base-uncased",
                 context_window: int = 50,
                 similarity_metric: str = "cosine"):
        """
        初始化BERT WSD
        Initialize BERT WSD
        
        Args:
            model_name: BERT模型名称
            context_window: 上下文窗口大小
            similarity_metric: 相似度度量 ("cosine", "dot")
        """
        super().__init__(context_window=context_window, name="BERTContextWSD")
        self.model_name = model_name
        self.similarity_metric = similarity_metric
        
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """
        加载BERT模型
        Load BERT model
        """
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            
            # 检测设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self.device)
    
    def get_word_embedding(self, text: str, target_word: str) -> np.ndarray:
        """
        获取目标词在上下文中的BERT嵌入
        Get BERT embedding of target word in context
        
        Args:
            text: 上下文文本
            target_word: 目标词
            
        Returns:
            词向量 (numpy array)
        """
        import torch
        
        self._load_model()
        
        # 分词
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # 找到目标词的token位置
        tokens = self._tokenizer.tokenize(text)
        target_tokens = self._tokenizer.tokenize(target_word)
        
        # 简单的位置查找
        target_positions = []
        for i in range(len(tokens)):
            if i + len(target_tokens) <= len(tokens):
                if tokens[i:i + len(target_tokens)] == target_tokens:
                    target_positions.extend(range(i + 1, i + len(target_tokens) + 1))
                    break
        
        if not target_positions:
            # 如果找不到，使用[CLS]向量
            word_embedding = hidden_states[0, 0, :].cpu().numpy()
        else:
            # 平均目标词的所有token的向量
            word_embedding = hidden_states[0, target_positions, :].mean(dim=0).cpu().numpy()
        
        return word_embedding
    
    def get_sense_embedding(self, sense_key: str) -> np.ndarray:
        """
        获取词义定义的BERT嵌入
        Get BERT embedding of sense definition
        
        Args:
            sense_key: 词义键
            
        Returns:
            定义向量
        """
        definition = self.get_sense_definition(sense_key)
        if not definition:
            return np.zeros(768)  # BERT base hidden size
        
        import torch
        
        self._load_model()
        
        inputs = self._tokenizer(
            definition,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # 使用[CLS]向量作为句子表示
            sense_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return sense_embedding
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的相似度
        Compute similarity between two vectors
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        if self.similarity_metric == "cosine":
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        elif self.similarity_metric == "dot":
            return float(np.dot(vec1, vec2))
        else:
            raise ValueError(f"不支持的相似度度量: {self.similarity_metric}")
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        使用BERT进行词义消歧
        Perform WSD using BERT
        
        Args:
            context: 上下文句子
            target_word: 目标词
            target_position: 目标词位置
            pos: 词性
            
        Returns:
            WSDResult: 消歧结果
        """
        # 获取所有候选词义
        synsets = self.get_word_senses(target_word, pos=pos)
        
        if not synsets:
            return WSDResult(
                word=target_word,
                sense_key="unknown",
                definition=f"No senses found for '{target_word}'",
                confidence=0.0,
                method=self.name,
                context=context
            )
        
        # 获取目标词的上下文嵌入
        word_embedding = self.get_word_embedding(context, target_word)
        
        # 计算与每个词义的相似度
        best_sense = synsets[0]
        best_score = -float('inf')
        all_senses = []
        
        for synset in synsets:
            sense_embedding = self.get_sense_embedding(synset.name())
            score = self.compute_similarity(word_embedding, sense_embedding)
            
            all_senses.append({
                'sense_key': synset.name(),
                'definition': synset.definition(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_sense = synset
        
        # 归一化置信度
        scores = [s['score'] for s in all_senses]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score > min_score:
            confidence = (best_score - min_score) / (max_score - min_score)
        else:
            confidence = 1.0 / len(synsets)
        
        return WSDResult(
            word=target_word,
            sense_key=best_sense.name(),
            definition=best_sense.definition(),
            confidence=confidence,
            method=self.name,
            context=context,
            all_senses=all_senses
        )


class KNNContextWSD(ContextBasedWSD):
    """
    基于KNN的上下文词义消歧
    KNN-based Contextual WSD
    
    使用上下文向量和KNN分类器进行词义消歧。
    Uses context vectors and KNN classifier for WSD.
    """
    
    def __init__(self, n_neighbors: int = 5, model_name: str = "bert-base-uncased"):
        """
        初始化KNN WSD
        Initialize KNN WSD
        
        Args:
            n_neighbors: KNN的邻居数
            model_name: 用于提取特征的模型名称
        """
        super().__init__(name="KNNWSD")
        self.n_neighbors = n_neighbors
        self.model_name = model_name
        
        self._model = None
        self._tokenizer = None
        self._training_data = {}  # {word: [(embedding, sense_key), ...]}
    
    def _load_model(self):
        """加载模型"""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self.device)
    
    def add_training_example(self, word: str, context: str, sense_key: str):
        """
        添加训练样本
        Add training example
        
        Args:
            word: 目标词
            context: 上下文
            sense_key: 正确的词义键
        """
        self._load_model()
        
        # 获取上下文嵌入
        embedding = self._get_context_embedding(context, word)
        
        if word not in self._training_data:
            self._training_data[word] = []
        self._training_data[word].append((embedding, sense_key))
    
    def _get_context_embedding(self, context: str, target_word: str) -> np.ndarray:
        """获取上下文嵌入"""
        import torch
        
        inputs = self._tokenizer(
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # 使用[CLS]向量
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return embedding
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        使用KNN进行词义消歧
        Perform WSD using KNN
        
        Args:
            context: 上下文句子
            target_word: 目标词
            target_position: 目标词位置
            pos: 词性
            
        Returns:
            WSDResult: 消歧结果
        """
        self._load_model()
        
        # 获取查询嵌入
        query_embedding = self._get_context_embedding(context, target_word)
        
        # 如果有训练数据，使用KNN
        if target_word in self._training_data and self._training_data[target_word]:
            training_examples = self._training_data[target_word]
            
            # 计算与所有训练样本的距离
            distances = []
            for emb, sense_key in training_examples:
                dist = np.linalg.norm(query_embedding - emb)
                distances.append((dist, sense_key))
            
            # 排序并取K个最近邻
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.n_neighbors]
            
            # 投票
            sense_votes = Counter(sense_key for _, sense_key in k_nearest)
            best_sense_key = sense_votes.most_common(1)[0][0]
            confidence = sense_votes[best_sense_key] / len(k_nearest)
            
            return WSDResult(
                word=target_word,
                sense_key=best_sense_key,
                definition=self.get_sense_definition(best_sense_key),
                confidence=confidence,
                method=self.name,
                context=context
            )
        
        # 没有训练数据时，回退到第一个词义
        synsets = self.get_word_senses(target_word, pos=pos)
        
        if synsets:
            return WSDResult(
                word=target_word,
                sense_key=synsets[0].name(),
                definition=synsets[0].definition(),
                confidence=1.0 / len(synsets),
                method=self.name,
                context=context
            )
        
        return WSDResult(
            word=target_word,
            sense_key="unknown",
            definition="",
            confidence=0.0,
            method=self.name,
            context=context
        )
