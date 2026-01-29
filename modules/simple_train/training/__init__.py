#!/usr/bin/env python3
"""
modules/simple_train/training/__init__.py
SuperBot - Simple Train Training Module
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Training modules:
- EntryTrainer: Trains the Entry model.
"""

from __future__ import annotations

from .entry_trainer import EntryTrainer

__all__ = [
    "EntryTrainer",
]
