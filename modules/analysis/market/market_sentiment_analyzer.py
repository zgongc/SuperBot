#!/usr/bin/env python3
"""
analysis/market_sentiment_analyzer.py
SuperBot - Market Sentiment Analyzer

Market sentiment analizi:
- Fear & Greed Index (alternative.me API + local calculation)
- Market momentum analysis
- Sentiment scoring (0-100)
- Trend detection (bearish, neutral, bullish)

Features:
- API fallback with local calculation
- Historical sentiment tracking
- EventBus integration
- Cache support

Usage:
    analyzer = MarketSentimentAnalyzer(config, cache_manager, data_manager, event_bus)
    await analyzer.initialize()
    sentiment = await analyzer.get_current_sentiment()
    # Returns: {"value": 65, "classification": "greed", "trend": "bullish"}

Dependencies:
    - aiohttp>=3.8.0
    - CacheManager
    - DataManager
    - EventBus
"""

import asyncio
from core.logger_engine import LoggerEngine
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import aiohttp

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification levels"""
    EXTREME_FEAR = "extreme_fear"      # 0-24
    FEAR = "fear"                      # 25-44
    NEUTRAL = "neutral"                # 45-54
    GREED = "greed"                    # 55-74
    EXTREME_GREED = "extreme_greed"    # 75-100


class MarketTrend(Enum):
    """Market trend direction"""
    BEARISH = "bearish"        # Downtrend
    NEUTRAL = "neutral"        # Sideways
    BULLISH = "bullish"        # Uptrend


class MarketSentimentAnalyzer:
    """
    Market sentiment analyzer with Fear & Greed Index

    Features:
    - External API integration (alternative.me)
    - Local calculation fallback
    - Historical tracking
    - Trend detection
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_manager=None,
        data_manager=None,
        event_bus=None
    ):
        """
        Initialize MarketSentimentAnalyzer

        Args:
            config: Configuration dictionary
            cache_manager: CacheManager instance
            data_manager: DataManager instance
            event_bus: EventBus instance
        """
        self.config = config or {}
        self.cache = cache_manager
        self.data_manager = data_manager
        self.event_bus = event_bus

        # Configuration
        self.api_url = "https://api.alternative.me/fng/"
        self.api_timeout = self.config.get("sentiment_api_timeout", 10)
        self.cache_ttl = self.config.get("sentiment_cache_ttl", 300)  # 5 minutes
        self.use_api = self.config.get("sentiment_use_api", True)
        self.update_interval = self.config.get("sentiment_update_interval", 300)  # 5 minutes

        # State
        self.current_sentiment: Optional[Dict[str, Any]] = None
        self.sentiment_history: List[Dict[str, Any]] = []
        self.max_history = 100
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info("MarketSentimentAnalyzer initialized")

    async def initialize(self):
        """Initialize analyzer and load historical data"""
        try:
            # Load sentiment history from database
            if self.data_manager:
                # TODO: Load from database when implemented
                pass

            # Get initial sentiment
            sentiment = await self.get_current_sentiment()
            if sentiment:
                logger.info(f"Initial sentiment: {sentiment['classification']} ({sentiment['value']})")

            logger.info("MarketSentimentAnalyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MarketSentimentAnalyzer: {e}")
            raise

    async def start(self):
        """Start periodic sentiment updates"""
        if self.is_running:
            logger.warning("MarketSentimentAnalyzer already running")
            return

        self.is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("MarketSentimentAnalyzer started")

    async def stop(self):
        """Stop periodic updates"""
        self.is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("MarketSentimentAnalyzer stopped")

    async def _update_loop(self):
        """Periodic sentiment update loop"""
        while self.is_running:
            try:
                await self.get_current_sentiment(force_refresh=True)
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sentiment update loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def get_current_sentiment(
        self,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get current market sentiment

        Args:
            force_refresh: Force refresh from API

        Returns:
            {
                "value": 65,                    # 0-100 sentiment score
                "classification": "greed",       # Sentiment level
                "trend": "bullish",             # Market trend
                "timestamp": 1697462400,
                "source": "api",                # "api" or "local"
                "previous_value": 60,           # Previous sentiment
                "change": 5,                    # Change from previous
                "change_percent": 8.33          # % change
            }
        """
        try:
            # Check cache first
            if not force_refresh and self.cache:
                cached = self.cache.get("sentiment:current")
                if cached:
                    logger.debug("Sentiment retrieved from cache")
                    return cached

            # Try API first
            sentiment = None
            if self.use_api:
                sentiment = await self._fetch_from_api()

            # Fallback to local calculation
            if not sentiment:
                sentiment = await self._calculate_local_sentiment()

            if sentiment:
                # Add trend analysis
                sentiment["trend"] = self._analyze_trend()

                # Calculate change from previous
                if self.current_sentiment:
                    prev_value = self.current_sentiment["value"]
                    sentiment["previous_value"] = prev_value
                    sentiment["change"] = sentiment["value"] - prev_value
                    sentiment["change_percent"] = (
                        (sentiment["change"] / prev_value * 100) if prev_value > 0 else 0
                    )

                # Update state
                self.current_sentiment = sentiment
                self._add_to_history(sentiment)

                # Cache sentiment
                if self.cache:
                    self.cache.set("sentiment:current", sentiment, ttl=self.cache_ttl)

                # Publish event
                if self.event_bus:
                    await self.event_bus.publish("sentiment.updated", sentiment)

                logger.info(
                    f"Sentiment: {sentiment['classification']} ({sentiment['value']}) "
                    f"[{sentiment['source']}]"
                )

            return sentiment

        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            return None

    async def _fetch_from_api(self) -> Optional[Dict[str, Any]]:
        """
        Fetch Fear & Greed Index from alternative.me API

        Returns:
            Sentiment data or None on failure
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_url,
                    timeout=aiohttp.ClientTimeout(total=self.api_timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data and "data" in data and len(data["data"]) > 0:
                            fng_data = data["data"][0]
                            value = int(fng_data["value"])

                            return {
                                "value": value,
                                "classification": self._classify_sentiment(value),
                                "timestamp": int(fng_data.get("timestamp", datetime.now(timezone.utc).timestamp())),
                                "source": "api",
                                "value_classification": fng_data.get("value_classification", "").lower()
                            }
                    else:
                        logger.warning(f"API returned status {response.status}")
                        return None

        except asyncio.TimeoutError:
            logger.warning("API request timeout")
            return None
        except Exception as e:
            logger.error(f"Error fetching from API: {e}")
            return None

    async def _calculate_local_sentiment(self) -> Optional[Dict[str, Any]]:
        """
        Calculate sentiment locally using market data

        Calculation based on:
        - Price momentum (30%)
        - Volume trend (20%)
        - Volatility (20%)
        - Market breadth (30%)

        Returns:
            Sentiment data
        """
        try:
            # TODO: Implement local calculation when market data available
            # For now, return neutral sentiment

            value = 50  # Neutral

            return {
                "value": value,
                "classification": self._classify_sentiment(value),
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
                "source": "local"
            }

        except Exception as e:
            logger.error(f"Error calculating local sentiment: {e}")
            return None

    def _classify_sentiment(self, value: int) -> str:
        """
        Classify sentiment value into category

        Args:
            value: Sentiment value (0-100)

        Returns:
            Sentiment classification string
        """
        if value <= 24:
            return SentimentLevel.EXTREME_FEAR.value
        elif value <= 44:
            return SentimentLevel.FEAR.value
        elif value <= 54:
            return SentimentLevel.NEUTRAL.value
        elif value <= 74:
            return SentimentLevel.GREED.value
        else:
            return SentimentLevel.EXTREME_GREED.value

    def _analyze_trend(self) -> str:
        """
        Analyze sentiment trend from history

        Returns:
            Trend classification (bearish/neutral/bullish)
        """
        if len(self.sentiment_history) < 3:
            return MarketTrend.NEUTRAL.value

        try:
            # Get last 3 sentiments
            recent = self.sentiment_history[-3:]
            values = [s["value"] for s in recent]

            # Calculate trend
            avg_change = (values[-1] - values[0]) / len(values)

            if avg_change > 5:
                return MarketTrend.BULLISH.value
            elif avg_change < -5:
                return MarketTrend.BEARISH.value
            else:
                return MarketTrend.NEUTRAL.value

        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return MarketTrend.NEUTRAL.value

    def _add_to_history(self, sentiment: Dict[str, Any]):
        """Add sentiment to history"""
        self.sentiment_history.append(sentiment)

        # Keep only max_history items
        if len(self.sentiment_history) > self.max_history:
            self.sentiment_history = self.sentiment_history[-self.max_history:]

    def get_sentiment_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get sentiment history

        Args:
            limit: Maximum number of records to return

        Returns:
            List of sentiment records
        """
        if limit:
            return self.sentiment_history[-limit:]
        return self.sentiment_history.copy()

    def get_sentiment_statistics(self) -> Dict[str, Any]:
        """
        Get sentiment statistics

        Returns:
            {
                "average": 55.5,
                "min": 25,
                "max": 80,
                "current": 65,
                "trend": "bullish",
                "samples": 50
            }
        """
        if not self.sentiment_history:
            return {}

        values = [s["value"] for s in self.sentiment_history]

        return {
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "current": self.current_sentiment["value"] if self.current_sentiment else None,
            "trend": self._analyze_trend(),
            "samples": len(values)
        }

    def is_extreme_sentiment(self) -> bool:
        """
        Check if current sentiment is extreme (fear or greed)

        Returns:
            True if extreme sentiment
        """
        if not self.current_sentiment:
            return False

        classification = self.current_sentiment["classification"]
        return classification in [
            SentimentLevel.EXTREME_FEAR.value,
            SentimentLevel.EXTREME_GREED.value
        ]

    def should_be_cautious(self) -> bool:
        """
        Check if trading should be cautious based on sentiment

        Extreme fear or extreme greed = be cautious

        Returns:
            True if should be cautious
        """
        return self.is_extreme_sentiment()

    def get_trading_signal(self) -> Optional[str]:
        """
        Get trading signal based on sentiment

        Contrarian approach:
        - Extreme fear → potential buy opportunity
        - Extreme greed → potential sell/caution

        Returns:
            "buy", "sell", "neutral", or None
        """
        if not self.current_sentiment:
            return None

        classification = self.current_sentiment["classification"]

        if classification == SentimentLevel.EXTREME_FEAR.value:
            return "buy"  # Buy when others are fearful
        elif classification == SentimentLevel.EXTREME_GREED.value:
            return "sell"  # Sell when others are greedy
        else:
            return "neutral"


if __name__ == "__main__":
    # Test code
    async def test():
        analyzer = MarketSentimentAnalyzer()
        await analyzer.initialize()

        sentiment = await analyzer.get_current_sentiment()
        if sentiment:
            print(f"Current sentiment: {sentiment}")
            print(f"Trading signal: {analyzer.get_trading_signal()}")
            print(f"Should be cautious: {analyzer.should_be_cautious()}")

    asyncio.run(test())
