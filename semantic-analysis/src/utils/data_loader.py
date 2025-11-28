"""
数据加载器模块
Data Loader Module

该模块提供WSD和SRL数据集的加载功能。
This module provides data loading functionality for WSD and SRL datasets.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator, Any

import numpy as np

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """
    数据加载器基类
    Base class for data loaders
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据加载器
        Initialize data loader
        
        Args:
            data_dir: 数据目录路径 / Data directory path
        """
        self.data_dir = Path(data_dir)
        self._cache = {}
    
    @abstractmethod
    def load(self, **kwargs) -> Any:
        """
        加载数据
        Load data
        """
        pass
    
    def _get_cache_key(self, **kwargs) -> str:
        """
        生成缓存键
        Generate cache key
        """
        return json.dumps(kwargs, sort_keys=True)
    
    def clear_cache(self):
        """
        清除缓存
        Clear cache
        """
        self._cache.clear()


class WSDDataLoader(DataLoader):
    """
    词义消歧数据加载器
    Word Sense Disambiguation Data Loader
    
    支持加载SemCor和其他WSD数据集。
    Supports loading SemCor and other WSD datasets.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化WSD数据加载器
        Initialize WSD data loader
        
        Args:
            data_dir: 数据目录路径
        """
        super().__init__(data_dir)
        self._nltk_initialized = False
    
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
                nltk.data.find('corpora/semcor')
            except LookupError:
                nltk.download('semcor', quiet=True)
            self._nltk_initialized = True
    
    def load(self, dataset: str = "semcor", split: str = "train",
             max_samples: Optional[int] = None, **kwargs) -> List[Dict]:
        """
        加载WSD数据集
        Load WSD dataset
        
        Args:
            dataset: 数据集名称 ("semcor", "senseval2", etc.)
            split: 数据划分 ("train", "dev", "test")
            max_samples: 最大样本数量
            
        Returns:
            样本列表，每个样本包含:
            - sentence: 句子文本
            - target_word: 目标词
            - target_position: 目标词位置
            - sense_key: 词义键
            - lemma: 词元
            
        Example:
            >>> loader = WSDDataLoader()
            >>> samples = loader.load("semcor", max_samples=100)
            >>> print(samples[0])
        """
        cache_key = self._get_cache_key(
            dataset=dataset, split=split, max_samples=max_samples
        )
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if dataset == "semcor":
            data = self._load_semcor(max_samples)
        elif dataset == "sample":
            data = self._load_sample_data()
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
        
        self._cache[cache_key] = data
        return data
    
    def _load_semcor(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        从NLTK加载SemCor数据集
        Load SemCor dataset from NLTK
        
        Args:
            max_samples: 最大样本数量
            
        Returns:
            WSD样本列表
        """
        self._init_nltk()
        from nltk.corpus import semcor
        from nltk.corpus import wordnet as wn
        
        samples = []
        count = 0
        
        for sent in semcor.tagged_sents(tag='sem'):
            # 获取句子中的词
            words = []
            for item in sent:
                if hasattr(item, 'label'):
                    # 带有词义标注的词
                    word = ' '.join(item.leaves())
                    words.append(word)
                else:
                    # 普通词
                    if isinstance(item, str):
                        words.append(item)
                    else:
                        words.append(' '.join(item.leaves()) if hasattr(item, 'leaves') else str(item))
            
            sentence = ' '.join(words)
            
            # 提取标注的词
            for i, item in enumerate(sent):
                if hasattr(item, 'label') and item.label() is not None:
                    try:
                        label = item.label()
                        if hasattr(label, 'synset'):
                            synset = label.synset()
                            word = ' '.join(item.leaves())
                            
                            sample = {
                                'sentence': sentence,
                                'target_word': word,
                                'target_position': i,
                                'sense_key': synset.name(),
                                'lemma': synset.lemmas()[0].name() if synset.lemmas() else word,
                                'definition': synset.definition()
                            }
                            samples.append(sample)
                            count += 1
                            
                            if max_samples and count >= max_samples:
                                return samples
                    except Exception as e:
                        continue
        
        logger.info(f"从SemCor加载了 {len(samples)} 个样本")
        return samples
    
    def _load_sample_data(self) -> List[Dict]:
        """
        加载示例数据
        Load sample data
        
        Returns:
            示例WSD样本列表
        """
        # 内置的示例数据
        samples = [
            {
                'sentence': 'I went to the bank to deposit money.',
                'target_word': 'bank',
                'target_position': 4,
                'sense_key': 'bank.n.01',
                'lemma': 'bank',
                'definition': 'a financial institution'
            },
            {
                'sentence': 'The river bank was covered with flowers.',
                'target_word': 'bank',
                'target_position': 2,
                'sense_key': 'bank.n.02',
                'lemma': 'bank',
                'definition': 'sloping land beside a body of water'
            },
            {
                'sentence': 'The bright star is visible tonight.',
                'target_word': 'star',
                'target_position': 2,
                'sense_key': 'star.n.01',
                'lemma': 'star',
                'definition': 'a celestial body of hot gases'
            },
            {
                'sentence': 'The movie star attended the premiere.',
                'target_word': 'star',
                'target_position': 2,
                'sense_key': 'star.n.04',
                'lemma': 'star',
                'definition': 'a performer who attracts special attention'
            },
            {
                'sentence': 'Please open the window.',
                'target_word': 'window',
                'target_position': 3,
                'sense_key': 'window.n.01',
                'lemma': 'window',
                'definition': 'a framework of wood or metal in a wall'
            },
        ]
        return samples
    
    def get_word_senses(self, word: str, pos: Optional[str] = None) -> List[Dict]:
        """
        获取词的所有词义
        Get all senses for a word
        
        Args:
            word: 目标词
            pos: 词性 (n, v, a, r)
            
        Returns:
            词义列表
        """
        self._init_nltk()
        from nltk.corpus import wordnet as wn
        
        synsets = wn.synsets(word, pos=pos)
        senses = []
        
        for synset in synsets:
            sense = {
                'sense_key': synset.name(),
                'definition': synset.definition(),
                'examples': synset.examples(),
                'lemmas': [l.name() for l in synset.lemmas()],
                'hypernyms': [h.name() for h in synset.hypernyms()],
                'hyponyms': [h.name() for h in synset.hyponyms()],
            }
            senses.append(sense)
        
        return senses


class SRLDataLoader(DataLoader):
    """
    语义角色标注数据加载器
    Semantic Role Labeling Data Loader
    
    支持加载CoNLL格式的SRL数据。
    Supports loading SRL data in CoNLL format.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化SRL数据加载器
        Initialize SRL data loader
        
        Args:
            data_dir: 数据目录路径
        """
        super().__init__(data_dir)
    
    def load(self, dataset: str = "sample", split: str = "train",
             max_samples: Optional[int] = None, **kwargs) -> List[Dict]:
        """
        加载SRL数据集
        Load SRL dataset
        
        Args:
            dataset: 数据集名称 ("propbank", "conll2005", "sample")
            split: 数据划分
            max_samples: 最大样本数量
            
        Returns:
            样本列表，每个样本包含:
            - words: 词列表
            - predicates: 谓词列表
            - arguments: 论元标注列表
        """
        cache_key = self._get_cache_key(
            dataset=dataset, split=split, max_samples=max_samples
        )
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if dataset == "sample":
            data = self._load_sample_data()
        elif dataset == "conll":
            data_path = self.data_dir / "raw/srl" / f"{split}.txt"
            data = self._load_conll_file(data_path, max_samples)
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
        
        self._cache[cache_key] = data
        return data
    
    def _load_sample_data(self) -> List[Dict]:
        """
        加载示例SRL数据
        Load sample SRL data
        
        Returns:
            示例SRL样本列表
        """
        samples = [
            {
                'words': ['The', 'cat', 'ate', 'the', 'fish', '.'],
                'predicate': 'ate',
                'predicate_index': 2,
                'arguments': [
                    {'role': 'ARG0', 'span': (0, 2), 'text': 'The cat'},
                    {'role': 'ARG1', 'span': (3, 5), 'text': 'the fish'},
                ]
            },
            {
                'words': ['She', 'gave', 'him', 'a', 'book', 'yesterday', '.'],
                'predicate': 'gave',
                'predicate_index': 1,
                'arguments': [
                    {'role': 'ARG0', 'span': (0, 1), 'text': 'She'},
                    {'role': 'ARG1', 'span': (3, 5), 'text': 'a book'},
                    {'role': 'ARG2', 'span': (2, 3), 'text': 'him'},
                    {'role': 'ARGM-TMP', 'span': (5, 6), 'text': 'yesterday'},
                ]
            },
            {
                'words': ['The', 'teacher', 'asked', 'the', 'students', 'to', 'read', '.'],
                'predicate': 'asked',
                'predicate_index': 2,
                'arguments': [
                    {'role': 'ARG0', 'span': (0, 2), 'text': 'The teacher'},
                    {'role': 'ARG1', 'span': (3, 5), 'text': 'the students'},
                    {'role': 'ARG2', 'span': (5, 7), 'text': 'to read'},
                ]
            },
            {
                'words': ['John', 'bought', 'a', 'car', 'in', 'Tokyo', '.'],
                'predicate': 'bought',
                'predicate_index': 1,
                'arguments': [
                    {'role': 'ARG0', 'span': (0, 1), 'text': 'John'},
                    {'role': 'ARG1', 'span': (2, 4), 'text': 'a car'},
                    {'role': 'ARGM-LOC', 'span': (4, 6), 'text': 'in Tokyo'},
                ]
            },
        ]
        return samples
    
    def _load_conll_file(self, file_path: Path, 
                         max_samples: Optional[int] = None) -> List[Dict]:
        """
        加载CoNLL格式文件
        Load CoNLL format file
        
        Args:
            file_path: 文件路径
            max_samples: 最大样本数量
            
        Returns:
            SRL样本列表
        """
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return []
        
        samples = []
        current_sentence = {'words': [], 'tags': []}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 空行表示句子结束
                if not line:
                    if current_sentence['words']:
                        sample = self._parse_conll_sentence(current_sentence)
                        if sample:
                            samples.append(sample)
                        current_sentence = {'words': [], 'tags': []}
                    
                    if max_samples and len(samples) >= max_samples:
                        break
                    continue
                
                # 跳过注释行
                if line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 4:
                    current_sentence['words'].append(parts[0])
                    current_sentence['tags'].append(parts[-1])
        
        # 处理最后一个句子
        if current_sentence['words']:
            sample = self._parse_conll_sentence(current_sentence)
            if sample:
                samples.append(sample)
        
        logger.info(f"从 {file_path} 加载了 {len(samples)} 个样本")
        return samples
    
    def _parse_conll_sentence(self, sentence: Dict) -> Optional[Dict]:
        """
        解析CoNLL格式的句子
        Parse CoNLL format sentence
        
        Args:
            sentence: 包含words和tags的字典
            
        Returns:
            解析后的SRL样本
        """
        words = sentence['words']
        tags = sentence['tags']
        
        # 查找谓词和论元
        predicate = None
        predicate_index = -1
        arguments = []
        current_arg = None
        
        for i, (word, tag) in enumerate(zip(words, tags)):
            if tag == 'B-V':
                predicate = word
                predicate_index = i
            elif tag.startswith('B-ARG'):
                if current_arg:
                    arguments.append(current_arg)
                role = tag[2:]  # 去掉 'B-' 前缀
                current_arg = {
                    'role': role,
                    'span': [i, i + 1],
                    'text': word
                }
            elif tag.startswith('I-ARG') and current_arg:
                current_arg['span'][1] = i + 1
                current_arg['text'] += ' ' + word
            elif tag == 'O' and current_arg:
                arguments.append(current_arg)
                current_arg = None
        
        if current_arg:
            arguments.append(current_arg)
        
        if predicate is None:
            return None
        
        # 转换span为元组
        for arg in arguments:
            arg['span'] = tuple(arg['span'])
        
        return {
            'words': words,
            'predicate': predicate,
            'predicate_index': predicate_index,
            'arguments': arguments
        }


def load_wsd_data(dataset: str = "semcor", **kwargs) -> List[Dict]:
    """
    便捷函数：加载WSD数据
    Convenience function to load WSD data
    
    Args:
        dataset: 数据集名称
        **kwargs: 其他参数
        
    Returns:
        WSD样本列表
    """
    loader = WSDDataLoader()
    return loader.load(dataset=dataset, **kwargs)


def load_srl_data(dataset: str = "sample", **kwargs) -> List[Dict]:
    """
    便捷函数：加载SRL数据
    Convenience function to load SRL data
    
    Args:
        dataset: 数据集名称
        **kwargs: 其他参数
        
    Returns:
        SRL样本列表
    """
    loader = SRLDataLoader()
    return loader.load(dataset=dataset, **kwargs)
