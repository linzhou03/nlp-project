"""
词义消歧模块
Word Sense Disambiguation (WSD) Module

该模块提供多种词义消歧方法的实现。
This module provides implementations of various WSD methods.
"""

from .base import WSDBase, WSDResult
from .context_based import ContextBasedWSD, LeskWSD, BERTContextWSD
from .knowledge_enhanced import KnowledgeEnhancedWSD, GraphBasedWSD

__all__ = [
    "WSDBase",
    "WSDResult",
    "ContextBasedWSD",
    "LeskWSD",
    "BERTContextWSD",
    "KnowledgeEnhancedWSD",
    "GraphBasedWSD",
]
