"""
基于神经网络的语义角色标注模块
Neural SRL Module

该模块实现了基于神经网络的语义角色标注方法，包括：
1. BiLSTM-CRF序列标注模型
2. 基于BERT的SRL模型

This module implements neural network-based SRL methods including:
1. BiLSTM-CRF sequence labeling model
2. BERT-based SRL model
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from .base import SRLBase, SRLResult, SemanticRole

logger = logging.getLogger(__name__)


class NeuralSRL(SRLBase):
    """
    基于神经网络的语义角色标注基类
    Neural Network-based SRL Base Class
    
    所有神经网络SRL方法应该继承这个类。
    All neural SRL methods should inherit from this class.
    """
    
    def __init__(self, name: str = "NeuralSRL"):
        """
        初始化神经网络SRL
        Initialize Neural SRL
        
        Args:
            name: 方法名称
        """
        super().__init__(name=name)
        self.model = None
        self.is_trained = False
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None,
              epochs: int = 10, batch_size: int = 32, **kwargs):
        """
        训练模型
        Train the model
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 训练轮数
            batch_size: 批次大小
            **kwargs: 其他参数
        """
        raise NotImplementedError("子类必须实现train方法")
    
    def save(self, path: str):
        """
        保存模型
        Save model
        
        Args:
            path: 保存路径
        """
        raise NotImplementedError("子类必须实现save方法")
    
    def load(self, path: str):
        """
        加载模型
        Load model
        
        Args:
            path: 模型路径
        """
        raise NotImplementedError("子类必须实现load方法")


class BiLSTMCRFSRL(NeuralSRL):
    """
    BiLSTM-CRF语义角色标注模型
    BiLSTM-CRF SRL Model
    
    使用双向LSTM提取特征，CRF层进行序列标注。
    Uses bidirectional LSTM for feature extraction and CRF layer for sequence labeling.
    
    Example:
        >>> model = BiLSTMCRFSRL()
        >>> model.train(train_data, epochs=10)
        >>> results = model.predict("The cat ate the fish.")
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 use_pretrained_embeddings: bool = False,
                 pretrained_model: str = "bert-base-uncased"):
        """
        初始化BiLSTM-CRF模型
        Initialize BiLSTM-CRF model
        
        Args:
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout率
            use_pretrained_embeddings: 是否使用预训练词嵌入
            pretrained_model: 预训练模型名称
        """
        super().__init__(name="BiLSTMCRFSRL")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_model = pretrained_model
        
        # 标签映射
        self.tag2idx = {}
        self.idx2tag = {}
        
        # 词汇表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        self._model = None
        self._device = None
    
    def _build_model(self, vocab_size: int, num_tags: int):
        """
        构建模型
        Build model
        
        Args:
            vocab_size: 词汇表大小
            num_tags: 标签数量
        """
        import torch
        import torch.nn as nn
        try:
            from TorchCRF import CRF
            USE_TORCHCRF = True
        except ImportError:
            try:
                from torchcrf import CRF
                USE_TORCHCRF = False
            except ImportError:
                CRF = None
                USE_TORCHCRF = False
        
        class BiLSTMCRF(nn.Module):
            """BiLSTM-CRF模型"""
            
            def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                         num_tags, num_layers, dropout):
                super().__init__()
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim, 
                                             padding_idx=0)
                self.lstm = nn.LSTM(
                    embedding_dim, hidden_dim // 2,
                    num_layers=num_layers,
                    bidirectional=True,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.hidden2tag = nn.Linear(hidden_dim, num_tags)
                self.num_tags = num_tags
                
                # 根据CRF库版本初始化
                if CRF is not None:
                    if USE_TORCHCRF:
                        # TorchCRF library (different API)
                        self.crf = CRF(num_tags)
                    else:
                        # pytorch-crf library
                        self.crf = CRF(num_tags, batch_first=True)
                else:
                    self.crf = None
                    
                self.use_torchcrf = USE_TORCHCRF
            
            def forward(self, x, mask=None):
                """前向传播，返回发射分数"""
                embeddings = self.embedding(x)
                embeddings = self.dropout(embeddings)
                lstm_out, _ = self.lstm(embeddings)
                emissions = self.hidden2tag(lstm_out)
                return emissions
            
            def loss(self, x, tags, mask=None):
                """计算CRF损失"""
                emissions = self.forward(x, mask)
                if mask is None:
                    mask = torch.ones_like(x, dtype=torch.bool)
                
                if self.crf is not None:
                    if self.use_torchcrf:
                        # TorchCRF uses different API and returns per-sample loss
                        log_likelihood = self.crf(emissions, tags, mask)
                        return -log_likelihood.mean()  # Average over batch
                    else:
                        return -self.crf(emissions, tags, mask=mask, reduction='mean')
                else:
                    # Fallback to simple cross entropy
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    return loss_fn(emissions.view(-1, self.num_tags), tags.view(-1))
            
            def decode(self, x, mask=None):
                """解码得到最佳标签序列"""
                emissions = self.forward(x, mask)
                if mask is None:
                    mask = torch.ones_like(x, dtype=torch.bool)
                    
                if self.crf is not None:
                    if self.use_torchcrf:
                        return self.crf.viterbi_decode(emissions, mask)
                    else:
                        return self.crf.decode(emissions, mask=mask)
                else:
                    # Fallback to argmax
                    return torch.argmax(emissions, dim=-1).tolist()
        
        self._model = BiLSTMCRF(
            vocab_size, self.embedding_dim, self.hidden_dim,
            num_tags, self.num_layers, self.dropout
        )
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
    
    def _build_vocab(self, train_data: List[Dict]):
        """
        构建词汇表和标签映射
        Build vocabulary and tag mapping
        
        Args:
            train_data: 训练数据
        """
        # 收集所有词和标签
        for sample in train_data:
            words = sample.get('words', [])
            for word in words:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        
        # 构建标签映射
        # 使用BIO标注格式
        tags = ['O', 'B-V']
        for role in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4',
                     'ARGM-LOC', 'ARGM-TMP', 'ARGM-MNR', 'ARGM-CAU', 
                     'ARGM-DIR', 'ARGM-PRP', 'ARGM-NEG', 'ARGM-MOD']:
            tags.append(f'B-{role}')
            tags.append(f'I-{role}')
        
        for i, tag in enumerate(tags):
            self.tag2idx[tag] = i
            self.idx2tag[i] = tag
    
    def _prepare_batch(self, samples: List[Dict]) -> Tuple:
        """
        准备批次数据
        Prepare batch data
        
        Args:
            samples: 样本列表
            
        Returns:
            (输入张量, 标签张量, 掩码张量)
        """
        import torch
        
        max_len = max(len(s.get('words', [])) for s in samples)
        
        batch_x = []
        batch_y = []
        batch_mask = []
        
        for sample in samples:
            words = sample.get('words', [])
            
            # 转换词为索引
            word_ids = [self.word2idx.get(w, 1) for w in words]  # 1是<UNK>
            
            # 生成BIO标签
            tags = ['O'] * len(words)
            predicate_idx = sample.get('predicate_index', -1)
            if 0 <= predicate_idx < len(tags):
                tags[predicate_idx] = 'B-V'
            
            for arg in sample.get('arguments', []):
                role = arg.get('role', 'ARG1')
                start, end = arg.get('span', (0, 0))
                for i in range(start, min(end, len(tags))):
                    if i == start:
                        tags[i] = f'B-{role}'
                    else:
                        tags[i] = f'I-{role}'
            
            # 转换标签为索引
            tag_ids = [self.tag2idx.get(t, 0) for t in tags]
            
            # 填充
            padding_len = max_len - len(word_ids)
            word_ids = word_ids + [0] * padding_len
            tag_ids = tag_ids + [0] * padding_len
            mask = [1] * len(words) + [0] * padding_len
            
            batch_x.append(word_ids)
            batch_y.append(tag_ids)
            batch_mask.append(mask)
        
        return (
            torch.tensor(batch_x, dtype=torch.long, device=self._device),
            torch.tensor(batch_y, dtype=torch.long, device=self._device),
            torch.tensor(batch_mask, dtype=torch.bool, device=self._device)
        )
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              **kwargs):
        """
        训练模型
        Train the model
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        import torch
        import torch.optim as optim
        
        # 构建词汇表
        self._build_vocab(train_data)
        
        # 构建模型
        vocab_size = len(self.word2idx)
        num_tags = len(self.tag2idx)
        self._build_model(vocab_size, num_tags)
        
        # 优化器
        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        
        logger.info(f"开始训练 BiLSTM-CRF 模型...")
        logger.info(f"词汇表大小: {vocab_size}, 标签数量: {num_tags}")
        
        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0
            num_batches = 0
            
            # 打乱数据
            import random
            random.shuffle(train_data)
            
            # 批次训练
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                x, y, mask = self._prepare_batch(batch)
                
                optimizer.zero_grad()
                loss = self._model.loss(x, y, mask)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # 验证
            if val_data:
                val_loss = self._evaluate(val_data, batch_size)
                logger.info(f"Epoch {epoch + 1}/{epochs} - "
                           f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("训练完成!")
    
    def _evaluate(self, data: List[Dict], batch_size: int) -> float:
        """
        评估模型
        Evaluate model
        
        Args:
            data: 评估数据
            batch_size: 批次大小
            
        Returns:
            平均损失
        """
        self._model.eval()
        total_loss = 0.0
        num_batches = 0
        
        import torch
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                x, y, mask = self._prepare_batch(batch)
                loss = self._model.loss(x, y, mask)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def predict(self, sentence: str) -> List[SRLResult]:
        """
        对句子进行语义角色标注
        Perform SRL on a sentence
        
        Args:
            sentence: 输入句子
            
        Returns:
            SRLResult列表
        """
        import torch
        
        # 如果模型未训练，使用简单规则
        if not self.is_trained or self._model is None:
            logger.warning("模型未训练，使用基于句法的方法")
            from .syntax_based import SyntaxBasedSRL
            return SyntaxBasedSRL().predict(sentence)
        
        # 分词
        words = self.tokenize(sentence)
        
        # 转换为索引
        word_ids = [self.word2idx.get(w, 1) for w in words]
        x = torch.tensor([word_ids], dtype=torch.long, device=self._device)
        mask = torch.ones_like(x, dtype=torch.bool)
        
        # 解码
        self._model.eval()
        with torch.no_grad():
            tag_ids_list = self._model.decode(x, mask)
        
        tag_ids = tag_ids_list[0]
        tags = [self.idx2tag.get(idx, 'O') for idx in tag_ids]
        
        # 转换为SRLResult
        return self._tags_to_results(sentence, words, tags)
    
    def _tags_to_results(self, sentence: str, words: List[str], 
                         tags: List[str]) -> List[SRLResult]:
        """
        将BIO标签转换为SRLResult
        Convert BIO tags to SRLResult
        
        Args:
            sentence: 原句子
            words: 词列表
            tags: 标签列表
            
        Returns:
            SRLResult列表
        """
        # 找出所有谓词
        predicates = []
        for i, tag in enumerate(tags):
            if tag == 'B-V':
                predicates.append(i)
        
        if not predicates:
            # 如果没有找到谓词，尝试找动词
            doc = self._get_spacy_doc(sentence)
            for token in doc:
                if token.pos_ == 'VERB':
                    predicates.append(token.i)
        
        results = []
        
        for pred_idx in predicates:
            arguments = []
            
            current_role = None
            current_start = -1
            
            for i, tag in enumerate(tags):
                if tag.startswith('B-') and tag != 'B-V':
                    # 保存之前的论元
                    if current_role:
                        text = ' '.join(words[current_start:i])
                        arguments.append(SemanticRole(
                            role=current_role,
                            text=text,
                            span=(current_start, i),
                            head_index=current_start
                        ))
                    
                    current_role = tag[2:]  # 去掉 'B-'
                    current_start = i
                    
                elif tag.startswith('I-'):
                    continue  # 继续当前论元
                    
                else:
                    # O或B-V，结束当前论元
                    if current_role:
                        text = ' '.join(words[current_start:i])
                        arguments.append(SemanticRole(
                            role=current_role,
                            text=text,
                            span=(current_start, i),
                            head_index=current_start
                        ))
                        current_role = None
            
            # 处理最后一个论元
            if current_role:
                text = ' '.join(words[current_start:len(words)])
                arguments.append(SemanticRole(
                    role=current_role,
                    text=text,
                    span=(current_start, len(words)),
                    head_index=current_start
                ))
            
            result = SRLResult(
                sentence=sentence,
                words=words,
                predicate=words[pred_idx] if pred_idx < len(words) else "",
                predicate_index=pred_idx,
                arguments=arguments,
                method=self.name
            )
            results.append(result)
        
        return results
    
    def save(self, path: str):
        """保存模型"""
        import torch
        import json
        from pathlib import Path
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型参数
        if self._model:
            torch.save(self._model.state_dict(), save_dir / "model.pt")
        
        # 保存词汇表和配置
        config = {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'word2idx': self.word2idx,
            'tag2idx': self.tag2idx,
        }
        
        with open(save_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        import torch
        import json
        from pathlib import Path
        
        load_dir = Path(path)
        
        # 加载配置
        with open(load_dir / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.word2idx = config['word2idx']
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.tag2idx = config['tag2idx']
        self.idx2tag = {int(v): k for k, v in self.tag2idx.items()}
        
        # 构建并加载模型
        vocab_size = len(self.word2idx)
        num_tags = len(self.tag2idx)
        self._build_model(vocab_size, num_tags)
        
        self._model.load_state_dict(torch.load(load_dir / "model.pt"))
        self.is_trained = True
        
        logger.info(f"模型已从 {path} 加载")


class BERTBasedSRL(NeuralSRL):
    """
    基于BERT的语义角色标注
    BERT-based SRL
    
    使用BERT作为编码器进行序列标注。
    Uses BERT as encoder for sequence labeling.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        初始化BERT SRL
        Initialize BERT SRL
        
        Args:
            model_name: BERT模型名称
        """
        super().__init__(name="BERTBasedSRL")
        self.model_name = model_name
        
        self._model = None
        self._tokenizer = None
        self._classifier = None
        
        # 标签映射
        self.tag2idx = {}
        self.idx2tag = {}
    
    def _load_model(self):
        """加载BERT模型"""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
    
    def predict(self, sentence: str) -> List[SRLResult]:
        """
        使用BERT进行语义角色标注
        Perform SRL using BERT
        
        注意：这是一个简化的实现，真正的BERT SRL需要在标注数据上微调。
        Note: This is a simplified implementation. Real BERT SRL requires fine-tuning on labeled data.
        
        Args:
            sentence: 输入句子
            
        Returns:
            SRLResult列表
        """
        # 如果模型未训练，回退到句法方法
        if not self.is_trained:
            logger.info("BERT SRL模型未训练，使用基于句法的方法")
            from .syntax_based import SyntaxBasedSRL
            return SyntaxBasedSRL().predict(sentence)
        
        self._load_model()
        
        import torch
        
        # 编码
        inputs = self._tokenizer(
            sentence,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # 获取BERT输出
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # 分类（需要训练好的分类层）
        if self._classifier:
            logits = self._classifier(hidden_states)
            predictions = torch.argmax(logits, dim=-1)
            tags = [self.idx2tag.get(p.item(), 'O') for p in predictions[0]]
        else:
            # 没有分类器，返回空结果
            tags = ['O'] * len(self._tokenizer.tokenize(sentence))
        
        # 转换为SRLResult
        words = self._tokenizer.tokenize(sentence)
        return self._tags_to_results(sentence, words, tags)
    
    def _tags_to_results(self, sentence: str, words: List[str],
                         tags: List[str]) -> List[SRLResult]:
        """将标签转换为结果"""
        # 与BiLSTM-CRF的实现类似
        predicates = []
        for i, tag in enumerate(tags):
            if tag == 'B-V':
                predicates.append(i)
        
        if not predicates:
            doc = self._get_spacy_doc(sentence)
            for token in doc:
                if token.pos_ == 'VERB':
                    predicates.append(token.i)
        
        results = []
        
        for pred_idx in predicates:
            arguments = []
            
            current_role = None
            current_start = -1
            
            for i, tag in enumerate(tags):
                if tag.startswith('B-') and tag != 'B-V':
                    if current_role:
                        text = ' '.join(words[current_start:i])
                        arguments.append(SemanticRole(
                            role=current_role,
                            text=text,
                            span=(current_start, i),
                            head_index=current_start
                        ))
                    
                    current_role = tag[2:]
                    current_start = i
                    
                elif tag.startswith('I-'):
                    continue
                    
                else:
                    if current_role:
                        text = ' '.join(words[current_start:i])
                        arguments.append(SemanticRole(
                            role=current_role,
                            text=text,
                            span=(current_start, i),
                            head_index=current_start
                        ))
                        current_role = None
            
            if current_role:
                text = ' '.join(words[current_start:len(words)])
                arguments.append(SemanticRole(
                    role=current_role,
                    text=text,
                    span=(current_start, len(words)),
                    head_index=current_start
                ))
            
            result = SRLResult(
                sentence=sentence,
                words=words,
                predicate=words[pred_idx] if pred_idx < len(words) else "",
                predicate_index=pred_idx,
                arguments=arguments,
                method=self.name
            )
            results.append(result)
        
        return results
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              **kwargs):
        """
        训练BERT SRL模型
        Train BERT SRL model
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self._load_model()
        
        # 构建标签映射
        tags = ['O', 'B-V']
        for role in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4',
                     'ARGM-LOC', 'ARGM-TMP', 'ARGM-MNR', 'ARGM-CAU',
                     'ARGM-DIR', 'ARGM-PRP', 'ARGM-NEG', 'ARGM-MOD']:
            tags.append(f'B-{role}')
            tags.append(f'I-{role}')
        
        for i, tag in enumerate(tags):
            self.tag2idx[tag] = i
            self.idx2tag[i] = tag
        
        # 添加分类层
        hidden_size = self._model.config.hidden_size
        num_tags = len(self.tag2idx)
        self._classifier = nn.Linear(hidden_size, num_tags).to(self._device)
        
        # 优化器
        optimizer = optim.AdamW(
            list(self._model.parameters()) + list(self._classifier.parameters()),
            lr=learning_rate
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        logger.info(f"开始训练 BERT SRL 模型...")
        
        for epoch in range(epochs):
            self._model.train()
            self._classifier.train()
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                # 准备数据
                texts = [' '.join(s.get('words', [])) for s in batch]
                inputs = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                # 准备标签
                max_len = inputs['input_ids'].size(1)
                labels = torch.full((len(batch), max_len), -100, 
                                   dtype=torch.long, device=self._device)
                
                for j, sample in enumerate(batch):
                    words = sample.get('words', [])
                    sample_tags = ['O'] * len(words)
                    
                    pred_idx = sample.get('predicate_index', -1)
                    if 0 <= pred_idx < len(sample_tags):
                        sample_tags[pred_idx] = 'B-V'
                    
                    for arg in sample.get('arguments', []):
                        role = arg.get('role', 'ARG1')
                        start, end = arg.get('span', (0, 0))
                        for k in range(start, min(end, len(sample_tags))):
                            if k == start:
                                sample_tags[k] = f'B-{role}'
                            else:
                                sample_tags[k] = f'I-{role}'
                    
                    # 转换为索引（简化：假设每个词对应一个token）
                    for k, tag in enumerate(sample_tags):
                        if k + 1 < max_len:  # +1 for [CLS]
                            labels[j, k + 1] = self.tag2idx.get(tag, 0)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self._model(**inputs)
                logits = self._classifier(outputs.last_hidden_state)
                
                # 计算损失
                loss = criterion(
                    logits.view(-1, num_tags),
                    labels.view(-1)
                )
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("训练完成!")
    
    def save(self, path: str):
        """保存模型"""
        from pathlib import Path
        import torch
        import json
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存BERT模型
        self._model.save_pretrained(save_dir / "bert")
        self._tokenizer.save_pretrained(save_dir / "bert")
        
        # 保存分类层
        if self._classifier:
            torch.save(self._classifier.state_dict(), save_dir / "classifier.pt")
        
        # 保存配置
        config = {
            'model_name': self.model_name,
            'tag2idx': self.tag2idx,
        }
        with open(save_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        from pathlib import Path
        from transformers import AutoModel, AutoTokenizer
        import torch
        import json
        
        load_dir = Path(path)
        
        # 加载配置
        with open(load_dir / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.tag2idx = config['tag2idx']
        self.idx2tag = {int(v): k for k, v in self.tag2idx.items()}
        
        # 加载BERT模型
        self._tokenizer = AutoTokenizer.from_pretrained(load_dir / "bert")
        self._model = AutoModel.from_pretrained(load_dir / "bert")
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        
        # 加载分类层
        import torch.nn as nn
        hidden_size = self._model.config.hidden_size
        num_tags = len(self.tag2idx)
        self._classifier = nn.Linear(hidden_size, num_tags).to(self._device)
        self._classifier.load_state_dict(torch.load(load_dir / "classifier.pt"))
        
        self.is_trained = True
        logger.info(f"模型已从 {path} 加载")
