"""
预处理工具模块
Preprocessing Utilities Module

该模块提供文本预处理功能。
This module provides text preprocessing functionality.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    文本预处理器
    Text Preprocessor
    
    提供文本清洗、分词、词形还原等功能。
    Provides text cleaning, tokenization, lemmatization, etc.
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 lemmatize: bool = True):
        """
        初始化预处理器
        Initialize preprocessor
        
        Args:
            lowercase: 是否转换为小写
            remove_punctuation: 是否移除标点
            remove_stopwords: 是否移除停用词
            lemmatize: 是否进行词形还原
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize_flag = lemmatize
        
        self._stopwords = None
        self._lemmatizer = None
        self._nltk_initialized = False
    
    def _init_nltk(self):
        """
        初始化NLTK资源
        Initialize NLTK resources
        """
        if not self._nltk_initialized:
            import nltk
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            self._nltk_initialized = True
    
    @property
    def stopwords(self) -> set:
        """
        获取停用词集合
        Get stopwords set
        """
        if self._stopwords is None:
            self._init_nltk()
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words('english'))
        return self._stopwords
    
    @property
    def lemmatizer(self):
        """
        获取词形还原器
        Get lemmatizer
        """
        if self._lemmatizer is None:
            self._init_nltk()
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer
    
    def preprocess(self, text: str) -> str:
        """
        预处理文本
        Preprocess text
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        words = text.split()
        
        if self.remove_stopwords:
            words = [w for w in words if w.lower() not in self.stopwords]
        
        if self.lemmatize_flag:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        Tokenize text
        
        Args:
            text: 输入文本
            
        Returns:
            词列表
        """
        # 简单的空格分词
        words = text.split()
        
        # 处理标点
        tokens = []
        for word in words:
            # 分离开头的标点
            while word and not word[0].isalnum():
                tokens.append(word[0])
                word = word[1:]
            
            # 分离结尾的标点
            end_puncts = []
            while word and not word[-1].isalnum():
                end_puncts.insert(0, word[-1])
                word = word[:-1]
            
            if word:
                tokens.append(word)
            tokens.extend(end_puncts)
        
        return tokens
    
    def get_context_window(self, text: str, target_position: int, 
                           window_size: int = 50) -> Tuple[str, str]:
        """
        获取目标词的上下文窗口
        Get context window around target word
        
        Args:
            text: 输入文本
            target_position: 目标词位置（词索引）
            window_size: 窗口大小（词数）
            
        Returns:
            (左侧上下文, 右侧上下文)
        """
        words = text.split()
        
        start = max(0, target_position - window_size)
        end = min(len(words), target_position + window_size + 1)
        
        left_context = ' '.join(words[start:target_position])
        right_context = ' '.join(words[target_position + 1:end])
        
        return left_context, right_context
    
    def lemmatize(self, word: str, pos: Optional[str] = None) -> str:
        """
        词形还原
        Lemmatize word
        
        Args:
            word: 输入词
            pos: 词性 (n, v, a, r)
            
        Returns:
            词元
        """
        if pos:
            return self.lemmatizer.lemmatize(word, pos)
        return self.lemmatizer.lemmatize(word)


class TokenizerWrapper:
    """
    分词器包装类
    Tokenizer Wrapper
    
    封装BERT等模型的分词器，提供统一接口。
    Wraps tokenizers like BERT to provide a unified interface.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 max_length: int = 512):
        """
        初始化分词器
        Initialize tokenizer
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """
        获取分词器（延迟加载）
        Get tokenizer (lazy loading)
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        Tokenize text
        
        Args:
            text: 输入文本
            
        Returns:
            token列表
        """
        return self.tokenizer.tokenize(text)
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Dict:
        """
        编码文本
        Encode text
        
        Args:
            text: 输入文本
            return_tensors: 返回张量类型 ("pt" for PyTorch)
            
        Returns:
            编码结果字典
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
    
    def encode_pair(self, text1: str, text2: str, 
                    return_tensors: Optional[str] = None) -> Dict:
        """
        编码文本对
        Encode text pair
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            return_tensors: 返回张量类型
            
        Returns:
            编码结果字典
        """
        return self.tokenizer(
            text1, text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids: List[int]) -> str:
        """
        解码token ID
        Decode token IDs
        
        Args:
            token_ids: token ID列表
            
        Returns:
            解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def find_word_positions(self, text: str, word: str) -> List[Tuple[int, int]]:
        """
        查找词在token序列中的位置
        Find word positions in token sequence
        
        Args:
            text: 输入文本
            word: 目标词
            
        Returns:
            位置列表 [(start, end), ...]
        """
        tokens = self.tokenize(text)
        word_tokens = self.tokenize(word)
        
        positions = []
        for i in range(len(tokens) - len(word_tokens) + 1):
            if tokens[i:i + len(word_tokens)] == word_tokens:
                positions.append((i + 1, i + len(word_tokens) + 1))  # +1 for [CLS]
        
        return positions


def get_pos_tag(word: str, context: Optional[str] = None) -> str:
    """
    获取词性标注
    Get POS tag for a word
    
    Args:
        word: 目标词
        context: 上下文句子
        
    Returns:
        词性标注
    """
    import nltk
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    if context:
        words = context.split()
        tags = nltk.pos_tag(words)
        for w, t in tags:
            if w.lower() == word.lower():
                return t
    
    # 默认情况
    tags = nltk.pos_tag([word])
    return tags[0][1]


def convert_pos_to_wordnet(pos_tag: str) -> Optional[str]:
    """
    将Penn Treebank词性转换为WordNet词性
    Convert Penn Treebank POS to WordNet POS
    
    Args:
        pos_tag: Penn Treebank词性标注
        
    Returns:
        WordNet词性 (n, v, a, r) 或 None
    """
    from nltk.corpus import wordnet as wn
    
    if pos_tag.startswith('NN'):
        return wn.NOUN
    elif pos_tag.startswith('VB'):
        return wn.VERB
    elif pos_tag.startswith('JJ'):
        return wn.ADJ
    elif pos_tag.startswith('RB'):
        return wn.ADV
    else:
        return None


def extract_context_features(text: str, target_position: int, 
                             window_size: int = 5) -> Dict:
    """
    提取上下文特征
    Extract context features
    
    Args:
        text: 输入文本
        target_position: 目标词位置
        window_size: 窗口大小
        
    Returns:
        特征字典
    """
    words = text.split()
    target_word = words[target_position] if target_position < len(words) else ""
    
    start = max(0, target_position - window_size)
    end = min(len(words), target_position + window_size + 1)
    
    left_words = words[start:target_position]
    right_words = words[target_position + 1:end]
    
    return {
        'target_word': target_word,
        'left_context': left_words,
        'right_context': right_words,
        'full_context': words[start:end],
        'position_in_window': target_position - start,
    }
