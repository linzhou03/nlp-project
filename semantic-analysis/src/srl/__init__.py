"""
语义角色标注模块
Semantic Role Labeling (SRL) Module

该模块提供多种语义角色标注方法的实现。
This module provides implementations of various SRL methods.
"""

from .base import SRLBase, SRLResult, SemanticRole
from .syntax_based import SyntaxBasedSRL
from .neural_srl import NeuralSRL, BiLSTMCRFSRL

__all__ = [
    "SRLBase",
    "SRLResult",
    "SemanticRole",
    "SyntaxBasedSRL",
    "NeuralSRL",
    "BiLSTMCRFSRL",
]
