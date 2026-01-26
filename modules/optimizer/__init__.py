#!/usr/bin/env python3
"""
components/optimizer/__init__.py
SuperBot - Optimizer Paketi
Author: SuperBot Team
Date: 2025-11-17
Versiyon: 2.0.0

Stage-by-stage optimization and AI training data system.

Exports:
    - MetricsCalculator: 30+ metric calculator
    - BacktestMetrics: Backtest metric dataclass
    - StageResult: Stage result dataclass
    - StageResultsManager: Stage result manager
    - DataExporter: AI training data exporter
    - Optimizer: Ana optimizer class
    - OptimizerConfig: Optimizer configuration

Usage:
    from components.optimizer import Optimizer, MetricsCalculator

    # Create optimizer
    optimizer = Optimizer(strategy=strategy, backtest_config=config)

    # Optimize all stages
    await optimizer.optimize_all_stages()
"""

from __future__ import annotations

from .metrics import MetricsCalculator, BacktestMetrics
from .stage_results import StageResult, StageResultsManager
from .data_exporter import DataExporter
from .optimizer import Optimizer, OptimizerConfig

__all__ = [
    'MetricsCalculator',
    'BacktestMetrics',
    'StageResult',
    'StageResultsManager',
    'DataExporter',
    'Optimizer',
    'OptimizerConfig',
]

__version__ = '2.0.0'
