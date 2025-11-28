"""
评估模块
Evaluation Module

该模块提供WSD和SRL的评估功能。
This module provides evaluation functionality for WSD and SRL.
"""

from .wsd_eval import WSDEvaluator, evaluate_wsd
from .srl_eval import SRLEvaluator, evaluate_srl

__all__ = [
    "WSDEvaluator",
    "SRLEvaluator",
    "evaluate_wsd",
    "evaluate_srl",
]
