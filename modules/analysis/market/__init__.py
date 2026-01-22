#!/usr/bin/env python3
"""
modules/analysis/market/__init__.py
SuperBot - Market Intelligence Analysis Package

Market-level analysis modules:
- MarketSentimentAnalyzer: Fear & Greed Index, sentiment analysis
- CorrelationMatrixEngine: Coin correlations, BTC dominance
- AltcoinSeasonDetector: Altcoin season detection
- VolumeProfileAnalyzer: Market-wide volume analysis

Moved: components/analysis/ -> modules/analysis/market/
"""

__version__ = "1.0.0"
__author__ = "SuperBot Team"

from .market_sentiment_analyzer import MarketSentimentAnalyzer, SentimentLevel, MarketTrend
from .correlation_matrix_analyzer import CorrelationMatrixEngine
from .altcoin_season_detector import AltcoinSeasonDetector, SeasonPhase
from .volume_profile_analyzer import VolumeProfileAnalyzer

__all__ = [
    "MarketSentimentAnalyzer",
    "SentimentLevel",
    "MarketTrend",
    "CorrelationMatrixEngine",
    "AltcoinSeasonDetector",
    "SeasonPhase",
    "VolumeProfileAnalyzer",
]
