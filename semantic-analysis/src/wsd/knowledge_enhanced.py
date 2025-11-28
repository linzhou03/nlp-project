"""
基于知识库增强的词义消歧模块
Knowledge-Enhanced WSD Module

该模块实现了整合WordNet知识库的词义消歧方法，包括：
1. 基于词义定义、例句和语义关系的联合建模
2. 基于知识图谱的表示学习
3. GlossBERT风格的联合编码

This module implements knowledge-enhanced WSD methods including:
1. Joint modeling of sense definitions, examples and semantic relations
2. Knowledge graph-based representation learning
3. GlossBERT-style joint encoding
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict

import numpy as np

from .base import WSDBase, WSDResult

logger = logging.getLogger(__name__)


class KnowledgeEnhancedWSD(WSDBase):
    """
    知识库增强的词义消歧
    Knowledge-Enhanced WSD
    
    结合WordNet的词义定义、例句和语义关系进行词义消歧。
    Combines WordNet sense definitions, examples and semantic relations for WSD.
    
    Example:
        >>> ke_wsd = KnowledgeEnhancedWSD()
        >>> result = ke_wsd.disambiguate(
        ...     "I went to the bank to deposit money",
        ...     "bank"
        ... )
    """
    
    def __init__(self, 
                 use_definitions: bool = True,
                 use_examples: bool = True,
                 use_hypernyms: bool = True,
                 use_hyponyms: bool = True,
                 use_holonyms: bool = False,
                 use_meronyms: bool = False,
                 model_name: str = "bert-base-uncased"):
        """
        初始化知识增强WSD
        Initialize Knowledge-Enhanced WSD
        
        Args:
            use_definitions: 是否使用词义定义
            use_examples: 是否使用例句
            use_hypernyms: 是否使用上位词
            use_hyponyms: 是否使用下位词
            use_holonyms: 是否使用整体词
            use_meronyms: 是否使用部分词
            model_name: 预训练模型名称
        """
        super().__init__(name="KnowledgeEnhancedWSD")
        
        self.use_definitions = use_definitions
        self.use_examples = use_examples
        self.use_hypernyms = use_hypernyms
        self.use_hyponyms = use_hyponyms
        self.use_holonyms = use_holonyms
        self.use_meronyms = use_meronyms
        self.model_name = model_name
        
        self._model = None
        self._tokenizer = None
        self._stopwords = None
    
    def _load_model(self):
        """加载预训练模型"""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self.device)
    
    @property
    def stopwords(self) -> set:
        """获取停用词"""
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
    
    def get_extended_gloss(self, synset, max_relations: int = 3) -> str:
        """
        获取扩展的词义注释（gloss）
        Get extended gloss for a synset
        
        包含定义、例句和相关词义的信息。
        Includes definition, examples and related senses information.
        
        Args:
            synset: WordNet Synset对象
            max_relations: 每种关系最多包含的数量
            
        Returns:
            扩展的gloss文本
        """
        parts = []
        
        # 词义定义
        if self.use_definitions:
            parts.append(synset.definition())
        
        # 例句
        if self.use_examples:
            for example in synset.examples()[:max_relations]:
                parts.append(example)
        
        # 同义词
        lemma_names = [l.name().replace('_', ' ') for l in synset.lemmas()]
        parts.append(' '.join(lemma_names))
        
        # 上位词
        if self.use_hypernyms:
            for hypernym in synset.hypernyms()[:max_relations]:
                parts.append(hypernym.definition())
        
        # 下位词
        if self.use_hyponyms:
            for hyponym in synset.hyponyms()[:max_relations]:
                parts.append(hyponym.definition())
        
        # 整体词
        if self.use_holonyms:
            for holonym in synset.member_holonyms()[:max_relations]:
                parts.append(holonym.definition())
        
        # 部分词
        if self.use_meronyms:
            for meronym in synset.part_meronyms()[:max_relations]:
                parts.append(meronym.definition())
        
        return ' '.join(parts)
    
    def get_gloss_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入表示
        Get embedding for text
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        import torch
        
        self._load_model()
        
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # 使用平均池化
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
        
        return embedding
    
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        Compute cosine similarity
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        使用知识增强方法进行词义消歧
        Perform WSD using knowledge-enhanced method
        
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
        
        # 获取上下文嵌入
        context_embedding = self.get_gloss_embedding(context)
        
        # 计算与每个词义扩展gloss的相似度
        best_sense = synsets[0]
        best_score = -float('inf')
        all_senses = []
        
        for synset in synsets:
            extended_gloss = self.get_extended_gloss(synset)
            gloss_embedding = self.get_gloss_embedding(extended_gloss)
            score = self.compute_cosine_similarity(context_embedding, gloss_embedding)
            
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
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
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


class GraphBasedWSD(WSDBase):
    """
    基于图的词义消歧
    Graph-based WSD
    
    利用WordNet的图结构进行词义消歧，使用PageRank等图算法。
    Uses WordNet's graph structure for WSD with algorithms like PageRank.
    
    Example:
        >>> graph_wsd = GraphBasedWSD()
        >>> result = graph_wsd.disambiguate(
        ...     "The bank is near the river",
        ...     "bank"
        ... )
    """
    
    def __init__(self, 
                 algorithm: str = "pagerank",
                 max_depth: int = 2,
                 damping_factor: float = 0.85):
        """
        初始化基于图的WSD
        Initialize Graph-based WSD
        
        Args:
            algorithm: 图算法 ("pagerank", "degree", "betweenness")
            max_depth: 图构建时的最大深度
            damping_factor: PageRank阻尼系数
        """
        super().__init__(name="GraphBasedWSD")
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.damping_factor = damping_factor
    
    def build_local_graph(self, synsets: List, depth: int = 2) -> Dict[str, Set[str]]:
        """
        构建局部语义图
        Build local semantic graph
        
        Args:
            synsets: 候选词义列表
            depth: 扩展深度
            
        Returns:
            邻接表表示的图
        """
        graph = defaultdict(set)
        visited = set()
        current_level = set(s.name() for s in synsets)
        
        for _ in range(depth):
            next_level = set()
            for sense_key in current_level:
                if sense_key in visited:
                    continue
                visited.add(sense_key)
                
                try:
                    synset = self.wordnet.synset(sense_key)
                    
                    # 添加上位词边
                    for hypernym in synset.hypernyms():
                        graph[sense_key].add(hypernym.name())
                        graph[hypernym.name()].add(sense_key)
                        next_level.add(hypernym.name())
                    
                    # 添加下位词边
                    for hyponym in synset.hyponyms()[:5]:
                        graph[sense_key].add(hyponym.name())
                        graph[hyponym.name()].add(sense_key)
                        next_level.add(hyponym.name())
                    
                except Exception:
                    continue
            
            current_level = next_level
        
        return graph
    
    def compute_pagerank(self, graph: Dict[str, Set[str]], 
                         initial_nodes: Set[str],
                         num_iterations: int = 20) -> Dict[str, float]:
        """
        计算PageRank分数
        Compute PageRank scores
        
        Args:
            graph: 邻接表图
            initial_nodes: 初始节点集合（给予更高初始权重）
            num_iterations: 迭代次数
            
        Returns:
            节点分数字典
        """
        nodes = list(graph.keys())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # 初始化分数
        scores = {node: 1.0 / n for node in nodes}
        
        # 给初始节点更高的权重
        for node in initial_nodes:
            if node in scores:
                scores[node] = 2.0 / n
        
        # PageRank迭代
        for _ in range(num_iterations):
            new_scores = {}
            for node in nodes:
                # 随机跳转分量
                rank = (1 - self.damping_factor) / n
                
                # 邻居贡献
                for neighbor in graph.get(node, set()):
                    if neighbor in scores:
                        out_degree = len(graph.get(neighbor, set()))
                        if out_degree > 0:
                            rank += self.damping_factor * scores[neighbor] / out_degree
                
                new_scores[node] = rank
            
            scores = new_scores
        
        return scores
    
    def compute_degree_centrality(self, graph: Dict[str, Set[str]]) -> Dict[str, float]:
        """
        计算度中心性
        Compute degree centrality
        
        Args:
            graph: 邻接表图
            
        Returns:
            节点分数字典
        """
        n = len(graph)
        if n <= 1:
            return {node: 1.0 for node in graph}
        
        return {node: len(neighbors) / (n - 1) for node, neighbors in graph.items()}
    
    def get_context_senses(self, context: str) -> List:
        """
        获取上下文中所有词的词义
        Get senses of all words in context
        
        Args:
            context: 上下文句子
            
        Returns:
            所有词义的列表
        """
        words = context.lower().split()
        all_synsets = []
        
        for word in words:
            if word.isalpha() and len(word) > 2:
                synsets = self.wordnet.synsets(word)
                all_synsets.extend(synsets)
        
        return all_synsets
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        使用图方法进行词义消歧
        Perform WSD using graph method
        
        Args:
            context: 上下文句子
            target_word: 目标词
            target_position: 目标词位置
            pos: 词性
            
        Returns:
            WSDResult: 消歧结果
        """
        # 获取目标词的候选词义
        target_synsets = self.get_word_senses(target_word, pos=pos)
        
        if not target_synsets:
            return WSDResult(
                word=target_word,
                sense_key="unknown",
                definition=f"No senses found for '{target_word}'",
                confidence=0.0,
                method=self.name,
                context=context
            )
        
        # 获取上下文词的所有词义
        context_synsets = self.get_context_senses(context)
        
        # 合并所有词义
        all_synsets = target_synsets + context_synsets
        
        # 构建局部图
        graph = self.build_local_graph(all_synsets, depth=self.max_depth)
        
        # 计算节点重要性
        if self.algorithm == "pagerank":
            context_sense_keys = set(s.name() for s in context_synsets)
            scores = self.compute_pagerank(graph, context_sense_keys)
        else:
            scores = self.compute_degree_centrality(graph)
        
        # 在目标词义中选择分数最高的
        best_sense = target_synsets[0]
        best_score = scores.get(target_synsets[0].name(), 0.0)
        all_senses = []
        
        for synset in target_synsets:
            score = scores.get(synset.name(), 0.0)
            
            all_senses.append({
                'sense_key': synset.name(),
                'definition': synset.definition(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_sense = synset
        
        # 归一化置信度
        target_scores = [scores.get(s.name(), 0.0) for s in target_synsets]
        max_score = max(target_scores) if target_scores else 0
        min_score = min(target_scores) if target_scores else 0
        
        if max_score > min_score:
            confidence = (best_score - min_score) / (max_score - min_score)
        else:
            confidence = 1.0 / len(target_synsets)
        
        return WSDResult(
            word=target_word,
            sense_key=best_sense.name(),
            definition=best_sense.definition(),
            confidence=confidence,
            method=self.name,
            context=context,
            all_senses=all_senses
        )


class GlossBERTWSD(WSDBase):
    """
    GlossBERT风格的词义消歧
    GlossBERT-style WSD
    
    使用BERT联合编码上下文和词义定义，进行词义消歧。
    Uses BERT to jointly encode context and sense definitions for WSD.
    
    Reference:
        Huang et al. (2019) "GlossBERT: BERT for Word Sense Disambiguation 
        with Gloss Knowledge"
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        初始化GlossBERT WSD
        Initialize GlossBERT WSD
        
        Args:
            model_name: 预训练模型名称
        """
        super().__init__(name="GlossBERTWSD")
        self.model_name = model_name
        
        self._model = None
        self._tokenizer = None
    
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
    
    def encode_context_gloss_pair(self, context: str, gloss: str, 
                                  target_word: str) -> np.ndarray:
        """
        编码上下文-词义定义对
        Encode context-gloss pair
        
        使用BERT的句子对编码方式，将上下文和词义定义作为输入。
        
        Args:
            context: 上下文句子
            gloss: 词义定义
            target_word: 目标词（用于标记）
            
        Returns:
            编码向量
        """
        import torch
        
        self._load_model()
        
        # 构造输入：[CLS] context [SEP] target_word: gloss [SEP]
        gloss_with_word = f"{target_word}: {gloss}"
        
        inputs = self._tokenizer(
            context,
            gloss_with_word,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # 使用[CLS]向量
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return cls_embedding
    
    def compute_pair_score(self, context: str, gloss: str, 
                           target_word: str) -> float:
        """
        计算上下文-词义对的匹配分数
        Compute matching score for context-gloss pair
        
        Args:
            context: 上下文句子
            gloss: 词义定义
            target_word: 目标词
            
        Returns:
            匹配分数
        """
        embedding = self.encode_context_gloss_pair(context, gloss, target_word)
        # 使用向量范数作为简单的分数度量
        # 在实际应用中，可以训练一个分类层
        return float(np.linalg.norm(embedding))
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: Optional[int] = None,
                     pos: Optional[str] = None) -> WSDResult:
        """
        使用GlossBERT风格方法进行词义消歧
        Perform WSD using GlossBERT-style method
        
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
        
        # 对每个词义计算匹配分数
        all_senses = []
        
        for synset in synsets:
            # 获取扩展的gloss（定义 + 例句）
            gloss = synset.definition()
            if synset.examples():
                gloss += ' ' + ' '.join(synset.examples()[:2])
            
            # 编码并计算相似度
            pair_embedding = self.encode_context_gloss_pair(context, gloss, target_word)
            context_embedding = self.encode_context_gloss_pair(context, "", target_word)
            
            # 计算余弦相似度
            score = float(np.dot(pair_embedding, context_embedding) / 
                         (np.linalg.norm(pair_embedding) * np.linalg.norm(context_embedding) + 1e-9))
            
            all_senses.append({
                'sense_key': synset.name(),
                'definition': synset.definition(),
                'score': score
            })
        
        # 选择分数最高的词义
        all_senses.sort(key=lambda x: x['score'], reverse=True)
        best = all_senses[0]
        
        # 计算置信度
        scores = [s['score'] for s in all_senses]
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score > min_score:
            confidence = (best['score'] - min_score) / (max_score - min_score)
        else:
            confidence = 1.0 / len(synsets)
        
        return WSDResult(
            word=target_word,
            sense_key=best['sense_key'],
            definition=best['definition'],
            confidence=confidence,
            method=self.name,
            context=context,
            all_senses=all_senses
        )
