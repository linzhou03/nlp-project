"""
基于句法树的语义角色标注模块
Syntax-based SRL Module

该模块实现了基于依存句法分析的语义角色标注方法。
This module implements SRL methods based on dependency parsing.

主要功能：
1. 使用spaCy进行依存句法分析
2. 基于句法规则识别谓词和论元
3. 依存路径特征提取
"""

import logging
from typing import List, Dict, Optional, Tuple, Set

from .base import SRLBase, SRLResult, SemanticRole

logger = logging.getLogger(__name__)


# 依存关系到语义角色的映射规则
# Mapping rules from dependency relations to semantic roles
DEP_TO_ROLE_RULES = {
    # 主语相关 / Subject related
    'nsubj': 'ARG0',      # 名词主语 -> 施事者
    'nsubjpass': 'ARG1',  # 被动主语 -> 受事者
    'csubj': 'ARG0',      # 分句主语
    
    # 宾语相关 / Object related
    'dobj': 'ARG1',       # 直接宾语 -> 受事者
    'obj': 'ARG1',        # 宾语 (spaCy v3)
    'iobj': 'ARG2',       # 间接宾语 -> 接受者
    'pobj': None,         # 介词宾语（需要根据介词判断）
    
    # 修饰语相关 / Modifier related
    'advmod': 'ARGM-MNR', # 副词修饰 -> 方式
    'npadvmod': 'ARGM-TMP',  # 名词作副词修饰 -> 时间
    'tmod': 'ARGM-TMP',   # 时间修饰
    
    # 介词短语相关 / Prepositional phrases
    'prep': None,         # 需要根据具体介词判断
    
    # 其他
    'agent': 'ARG0',      # 被动句中的by施事者
    'xcomp': 'ARG2',      # 开放补足语
    'ccomp': 'ARG1',      # 分句补语
    'acomp': 'ARG1',      # 形容词补语
}

# 介词到语义角色的映射
# Preposition to semantic role mapping
PREP_TO_ROLE = {
    # 地点介词
    'in': 'ARGM-LOC',
    'at': 'ARGM-LOC',
    'on': 'ARGM-LOC',
    'near': 'ARGM-LOC',
    'beside': 'ARGM-LOC',
    'under': 'ARGM-LOC',
    'above': 'ARGM-LOC',
    'inside': 'ARGM-LOC',
    'outside': 'ARGM-LOC',
    
    # 时间介词
    'before': 'ARGM-TMP',
    'after': 'ARGM-TMP',
    'during': 'ARGM-TMP',
    'since': 'ARGM-TMP',
    'until': 'ARGM-TMP',
    
    # 方向介词
    'to': 'ARGM-DIR',
    'from': 'ARGM-DIR',
    'towards': 'ARGM-DIR',
    'into': 'ARGM-DIR',
    
    # 原因/目的介词
    'because': 'ARGM-CAU',
    'for': 'ARGM-PRP',
    
    # 方式介词
    'with': 'ARG2',  # 通常是工具或伴随
    'by': 'ARGM-MNR',
    'via': 'ARGM-MNR',
    
    # 受益者
    'to': 'ARG2',  # give sth to sb
}


class SyntaxBasedSRL(SRLBase):
    """
    基于句法树的语义角色标注
    Syntax-based Semantic Role Labeling
    
    使用依存句法分析来识别谓词的论元及其语义角色。
    Uses dependency parsing to identify predicate arguments and their semantic roles.
    
    Example:
        >>> srl = SyntaxBasedSRL()
        >>> results = srl.predict("The cat ate the mouse in the garden.")
        >>> for result in results:
        ...     print(result)
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        初始化基于句法的SRL
        Initialize syntax-based SRL
        
        Args:
            spacy_model: spaCy模型名称
        """
        super().__init__(name="SyntaxBasedSRL")
        self.spacy_model_name = spacy_model
        self._nlp = None
    
    @property
    def nlp(self):
        """获取spaCy模型（延迟加载）"""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.spacy_model_name)
            except OSError:
                logger.warning(f"无法加载spaCy模型 {self.spacy_model_name}，尝试下载...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.spacy_model_name])
                self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp
    
    def predict(self, sentence: str) -> List[SRLResult]:
        """
        对句子进行语义角色标注
        Perform SRL on a sentence
        
        Args:
            sentence: 输入句子
            
        Returns:
            SRLResult列表
        """
        doc = self.nlp(sentence)
        results = []
        
        # 找出所有谓词（动词）
        for token in doc:
            if token.pos_ == 'VERB':
                result = self._extract_arguments(doc, token)
                if result:
                    results.append(result)
        
        return results
    
    def _extract_arguments(self, doc, predicate_token) -> Optional[SRLResult]:
        """
        提取谓词的论元
        Extract arguments for a predicate
        
        Args:
            doc: spaCy Doc对象
            predicate_token: 谓词token
            
        Returns:
            SRLResult或None
        """
        words = [token.text for token in doc]
        arguments = []
        
        # 遍历谓词的所有子节点
        for child in predicate_token.children:
            role = self._get_role_for_dependency(child, predicate_token)
            
            if role:
                # 获取完整的论元短语
                span_start, span_end, text = self._get_argument_span(child, doc)
                
                argument = SemanticRole(
                    role=role,
                    text=text,
                    span=(span_start, span_end),
                    head_index=child.i
                )
                arguments.append(argument)
        
        # 检查介词短语
        for child in predicate_token.children:
            if child.dep_ == 'prep':
                prep_args = self._handle_prepositional_phrase(child, doc)
                arguments.extend(prep_args)
        
        # 处理被动语态的by-agent
        if self._is_passive(predicate_token):
            for child in predicate_token.children:
                if child.dep_ == 'agent' or (child.text.lower() == 'by' and child.dep_ == 'prep'):
                    for pobj in child.children:
                        if pobj.dep_ == 'pobj':
                            span_start, span_end, text = self._get_argument_span(pobj, doc)
                            argument = SemanticRole(
                                role='ARG0',
                                text=text,
                                span=(span_start, span_end),
                                head_index=pobj.i
                            )
                            arguments.append(argument)
        
        return SRLResult(
            sentence=doc.text,
            words=words,
            predicate=predicate_token.text,
            predicate_index=predicate_token.i,
            arguments=arguments,
            method=self.name
        )
    
    def _get_role_for_dependency(self, token, predicate) -> Optional[str]:
        """
        根据依存关系获取语义角色
        Get semantic role based on dependency relation
        
        Args:
            token: 子节点token
            predicate: 谓词token
            
        Returns:
            语义角色或None
        """
        dep = token.dep_
        
        # 直接映射
        if dep in DEP_TO_ROLE_RULES:
            role = DEP_TO_ROLE_RULES[dep]
            if role:
                return role
        
        # 处理复合依存关系
        if dep == 'compound':
            return None  # 复合词不单独标注
        
        # 处理时间表达式
        if token.ent_type_ in ('DATE', 'TIME'):
            return 'ARGM-TMP'
        
        # 处理地点表达式
        if token.ent_type_ in ('GPE', 'LOC', 'FAC'):
            return 'ARGM-LOC'
        
        return None
    
    def _handle_prepositional_phrase(self, prep_token, doc) -> List[SemanticRole]:
        """
        处理介词短语
        Handle prepositional phrases
        
        Args:
            prep_token: 介词token
            doc: spaCy Doc对象
            
        Returns:
            SemanticRole列表
        """
        arguments = []
        prep_text = prep_token.text.lower()
        
        # 根据介词确定角色
        role = PREP_TO_ROLE.get(prep_text)
        
        # 如果没有预定义映射，尝试根据介词宾语的类型判断
        if role is None:
            for pobj in prep_token.children:
                if pobj.dep_ == 'pobj':
                    if pobj.ent_type_ in ('GPE', 'LOC', 'FAC'):
                        role = 'ARGM-LOC'
                    elif pobj.ent_type_ in ('DATE', 'TIME'):
                        role = 'ARGM-TMP'
                    elif pobj.ent_type_ == 'PERSON':
                        role = 'ARG2'  # 可能是接受者
                    else:
                        role = 'ARGM-MNR'  # 默认为方式
        
        # 提取介词宾语
        for pobj in prep_token.children:
            if pobj.dep_ == 'pobj' and role:
                span_start, span_end, text = self._get_argument_span(pobj, doc)
                # 包含介词本身
                full_text = prep_token.text + ' ' + text
                
                argument = SemanticRole(
                    role=role,
                    text=full_text,
                    span=(prep_token.i, span_end),
                    head_index=pobj.i
                )
                arguments.append(argument)
        
        return arguments
    
    def _get_argument_span(self, head_token, doc) -> Tuple[int, int, str]:
        """
        获取论元的完整跨度
        Get complete span of an argument
        
        Args:
            head_token: 论元头词token
            doc: spaCy Doc对象
            
        Returns:
            (起始索引, 结束索引, 文本)
        """
        # 获取子树中的所有token
        subtree_tokens = list(head_token.subtree)
        
        if not subtree_tokens:
            return head_token.i, head_token.i + 1, head_token.text
        
        # 按位置排序
        subtree_tokens.sort(key=lambda t: t.i)
        
        start = subtree_tokens[0].i
        end = subtree_tokens[-1].i + 1
        text = doc[start:end].text
        
        return start, end, text
    
    def _is_passive(self, verb_token) -> bool:
        """
        判断是否为被动语态
        Check if verb is in passive voice
        
        Args:
            verb_token: 动词token
            
        Returns:
            是否为被动语态
        """
        # 检查是否有被动主语
        for child in verb_token.children:
            if child.dep_ == 'nsubjpass':
                return True
        
        # 检查是否有aux be
        for child in verb_token.children:
            if child.dep_ == 'auxpass':
                return True
        
        return False
    
    def get_dependency_tree(self, sentence: str) -> Dict:
        """
        获取依存句法树
        Get dependency parse tree
        
        Args:
            sentence: 输入句子
            
        Returns:
            依存树的字典表示
        """
        doc = self.nlp(sentence)
        
        nodes = []
        for token in doc:
            node = {
                'id': token.i,
                'text': token.text,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.i,
                'children': [child.i for child in token.children]
            }
            nodes.append(node)
        
        return {
            'sentence': sentence,
            'nodes': nodes,
            'root': [token.i for token in doc if token.dep_ == 'ROOT']
        }
    
    def visualize_tree(self, sentence: str) -> str:
        """
        可视化依存句法树
        Visualize dependency parse tree
        
        Args:
            sentence: 输入句子
            
        Returns:
            文本格式的树结构
        """
        doc = self.nlp(sentence)
        
        lines = ["依存句法树 (Dependency Parse Tree):"]
        lines.append("-" * 50)
        
        for token in doc:
            indent = "  " * (token.i % 5)
            lines.append(
                f"{indent}{token.text} ({token.pos_}) "
                f"--{token.dep_}--> {token.head.text}"
            )
        
        return '\n'.join(lines)


class RuleBasedSRL(SRLBase):
    """
    基于规则的语义角色标注
    Rule-based SRL
    
    使用手工定义的语法规则进行语义角色标注。
    Uses manually defined grammar rules for SRL.
    """
    
    def __init__(self):
        """初始化规则SRL"""
        super().__init__(name="RuleBasedSRL")
        self._patterns = self._define_patterns()
    
    def _define_patterns(self) -> List[Dict]:
        """
        定义匹配模式
        Define matching patterns
        
        Returns:
            模式列表
        """
        # 简单的模式定义
        patterns = [
            {
                'name': 'SVO',  # Subject-Verb-Object
                'template': ['NOUN|PRON', 'VERB', 'NOUN|PRON'],
                'roles': ['ARG0', 'V', 'ARG1']
            },
            {
                'name': 'SVOO',  # Subject-Verb-Object-Object
                'template': ['NOUN|PRON', 'VERB', 'NOUN|PRON', 'NOUN|PRON'],
                'roles': ['ARG0', 'V', 'ARG2', 'ARG1']
            },
            {
                'name': 'SVA',  # Subject-Verb-Adjunct
                'template': ['NOUN|PRON', 'VERB', 'ADV|ADP'],
                'roles': ['ARG0', 'V', 'ARGM-MNR']
            },
        ]
        return patterns
    
    def predict(self, sentence: str) -> List[SRLResult]:
        """
        使用规则进行语义角色标注
        Perform SRL using rules
        
        Args:
            sentence: 输入句子
            
        Returns:
            SRLResult列表
        """
        doc = self._get_spacy_doc(sentence)
        words = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        
        results = []
        
        # 尝试匹配每个模式
        for pattern in self._patterns:
            matches = self._match_pattern(pos_tags, pattern['template'])
            
            for match_indices in matches:
                # 找到谓词
                verb_idx = -1
                for i, role in enumerate(pattern['roles']):
                    if role == 'V':
                        verb_idx = match_indices[i]
                        break
                
                if verb_idx < 0:
                    continue
                
                # 提取论元
                arguments = []
                for i, role in enumerate(pattern['roles']):
                    if role != 'V':
                        idx = match_indices[i]
                        argument = SemanticRole(
                            role=role,
                            text=words[idx],
                            span=(idx, idx + 1),
                            head_index=idx
                        )
                        arguments.append(argument)
                
                result = SRLResult(
                    sentence=sentence,
                    words=words,
                    predicate=words[verb_idx],
                    predicate_index=verb_idx,
                    arguments=arguments,
                    method=self.name
                )
                results.append(result)
        
        # 如果没有匹配到模式，回退到依存分析
        if not results:
            syntax_srl = SyntaxBasedSRL()
            results = syntax_srl.predict(sentence)
        
        return results
    
    def _match_pattern(self, pos_tags: List[str], 
                       template: List[str]) -> List[List[int]]:
        """
        匹配模式
        Match pattern
        
        Args:
            pos_tags: 词性标注列表
            template: 模式模板
            
        Returns:
            匹配的索引列表
        """
        matches = []
        
        # 简化的匹配：不要求连续
        def find_next_match(template_idx: int, start_pos: int, 
                           current_match: List[int]) -> None:
            if template_idx >= len(template):
                matches.append(current_match.copy())
                return
            
            target_pos = template[template_idx].split('|')
            
            for i in range(start_pos, len(pos_tags)):
                if pos_tags[i] in target_pos:
                    current_match.append(i)
                    find_next_match(template_idx + 1, i + 1, current_match)
                    current_match.pop()
        
        find_next_match(0, 0, [])
        return matches
