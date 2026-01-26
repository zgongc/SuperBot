#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
components/managers/__init__.py

SuperBot - Managers Package
Date: 2025-01-26

Manager components for data handling and connectivity.

Components:
    - SymbolsManager: Trading symbols management
    - WebSocketEngine: Real-time WebSocket connections
    - MultiTimeframeEngine: Multi-timeframe data handling
    - ParquetsEngine: Parquet file data management
"""

from components.managers.symbols_manager import SymbolsManager
from components.managers.websocket_engine import WebSocketEngine
from components.managers.multi_timeframe_engine import MultiTimeframeEngine
from components.managers.parquets_engine import ParquetsEngine

__all__ = [
    "SymbolsManager",
    "WebSocketEngine",
    "MultiTimeframeEngine",
    "ParquetsEngine",
]
