#!/usr/bin/env python3
"""
analysis/volume_profile_analyzer.py
SuperBot - Volume Profile Analyzer

Market-wide volume analizi:
- Total market volume tracking
- Volume distribution by symbol
- Volume spikes detection
- Volume trends (increasing/decreasing)
- High-volume zones identification

Features:
- Real-time volume aggregation
- Volume profile calculation
- POC (Point of Control) identification
- Value Area calculation
- EventBus integration

Usage:
    analyzer = VolumeProfileAnalyzer(config, data_manager, event_bus)
    await analyzer.initialize()

    # Total market volume
    total_vol = await analyzer.get_total_market_volume()

    # Volume profile for symbol
    profile = await analyzer.get_volume_profile("BTCUSDT")

    # Volume distribution
    distribution = await analyzer.get_volume_distribution()

    # Detect volume spike
    is_spike = await analyzer.is_volume_spike("BTCUSDT")

Dependencies:
    - numpy>=1.24.0
    - pandas>=2.0.0
    - DataManager
    - EventBus
"""

import asyncio
from core.logger_engine import LoggerEngine
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


class VolumeProfileAnalyzer:
    """
    Volume Profile Analyzer for market-wide volume analysis

    Features:
    - Total market volume tracking
    - Volume profile calculation
    - POC (Point of Control) identification
    - Volume spike detection
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_manager=None,
        event_bus=None
    ):
        """
        Initialize VolumeProfileAnalyzer

        Args:
            config: Configuration dictionary
            data_manager: DataManager instance
            event_bus: EventBus instance
        """
        self.config = config or {}
        self.data_manager = data_manager
        self.event_bus = event_bus

        # Configuration
        self.update_interval = self.config.get("volume_update_interval", 60)  # 1 minute
        self.profile_bins = self.config.get("volume_profile_bins", 50)
        self.value_area_percent = self.config.get("volume_value_area", 0.70)  # 70%
        self.spike_threshold = self.config.get("volume_spike_threshold", 2.0)  # 2x average
        self.lookback_periods = self.config.get("volume_lookback_periods", 100)

        # State
        self.symbol_volumes: Dict[str, List[float]] = defaultdict(list)
        self.symbol_prices: Dict[str, List[float]] = defaultdict(list)
        self.symbol_timestamps: Dict[str, List[int]] = defaultdict(list)
        self.total_market_volume: float = 0.0
        self.volume_profiles: Dict[str, Dict[str, Any]] = {}
        self.last_update: Optional[datetime] = None

        # Runtime
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info("VolumeProfileAnalyzer initialized")

    async def initialize(self):
        """Initialize analyzer and load historical data"""
        try:
            # Load historical volume data if available
            if self.data_manager:
                # TODO: Load from database when implemented
                pass

            logger.info("VolumeProfileAnalyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VolumeProfileAnalyzer: {e}")
            raise

    async def start(self):
        """Start periodic volume analysis"""
        if self.is_running:
            logger.warning("VolumeProfileAnalyzer already running")
            return

        self.is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("VolumeProfileAnalyzer started")

    async def stop(self):
        """Stop periodic updates"""
        self.is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("VolumeProfileAnalyzer stopped")

    async def _update_loop(self):
        """Periodic volume analysis loop"""
        while self.is_running:
            try:
                # Update market volume
                await self.update_market_volume()

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in volume analysis loop: {e}")
                await asyncio.sleep(60)

    async def add_volume_data(
        self,
        symbol: str,
        volume: float,
        price: float,
        timestamp: Optional[int] = None
    ):
        """
        Add volume data for a symbol

        Args:
            symbol: Trading symbol
            volume: Volume value
            price: Price at volume
            timestamp: Optional timestamp
        """
        try:
            if timestamp is None:
                timestamp = int(datetime.now(timezone.utc).timestamp())

            self.symbol_volumes[symbol].append(volume)
            self.symbol_prices[symbol].append(price)
            self.symbol_timestamps[symbol].append(timestamp)

            # Keep only recent data
            if len(self.symbol_volumes[symbol]) > self.lookback_periods:
                self.symbol_volumes[symbol] = self.symbol_volumes[symbol][-self.lookback_periods:]
                self.symbol_prices[symbol] = self.symbol_prices[symbol][-self.lookback_periods:]
                self.symbol_timestamps[symbol] = self.symbol_timestamps[symbol][-self.lookback_periods:]

            logger.debug(f"Added volume data for {symbol}: {volume:.2f}")

        except Exception as e:
            logger.error(f"Error adding volume data for {symbol}: {e}")

    async def get_total_market_volume(self) -> float:
        """
        Calculate total market volume

        Returns:
            Total volume across all symbols
        """
        try:
            total = 0.0

            for symbol, volumes in self.symbol_volumes.items():
                if volumes:
                    # Use latest volume
                    total += volumes[-1]

            self.total_market_volume = total
            return total

        except Exception as e:
            logger.error(f"Error calculating total market volume: {e}")
            return 0.0

    async def update_market_volume(self):
        """Update total market volume (wrapper for periodic updates)"""
        volume = await self.get_total_market_volume()

        if self.event_bus:
            await self.event_bus.publish("market.volume.updated", {
                "total_volume": volume,
                "timestamp": int(datetime.now(timezone.utc).timestamp())
            })

    async def get_volume_profile(
        self,
        symbol: str,
        bins: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate volume profile for a symbol

        Volume profile shows volume distribution across price levels

        Args:
            symbol: Trading symbol
            bins: Number of price bins

        Returns:
            {
                "poc": 50000.0,              # Point of Control (highest volume price)
                "value_area_high": 51000.0,  # Value Area High
                "value_area_low": 49000.0,   # Value Area Low
                "profile": [                  # Volume profile data
                    {"price": 50000, "volume": 1000},
                    {"price": 50100, "volume": 800},
                    ...
                ]
            }
        """
        try:
            if symbol not in self.symbol_volumes:
                logger.warning(f"No volume data for {symbol}")
                return None

            volumes = self.symbol_volumes[symbol]
            prices = self.symbol_prices[symbol]

            if len(volumes) == 0:
                return None

            bins = bins or self.profile_bins

            # Create DataFrame
            df = pd.DataFrame({
                "price": prices,
                "volume": volumes
            })

            # Bin prices and sum volumes
            df["price_bin"] = pd.cut(df["price"], bins=bins)
            profile_df = df.groupby("price_bin", observed=False)["volume"].sum().reset_index()

            # Get bin midpoints
            profile_df["price_mid"] = profile_df["price_bin"].apply(lambda x: x.mid)

            # Sort by volume (descending)
            profile_df = profile_df.sort_values("volume", ascending=False)

            # POC (Point of Control) - price level with highest volume
            poc_price = profile_df.iloc[0]["price_mid"]
            poc_volume = profile_df.iloc[0]["volume"]

            # Value Area - price range containing 70% of volume
            value_area = self._calculate_value_area(profile_df)

            # Create profile data
            profile_data = [
                {
                    "price": float(row["price_mid"]),
                    "volume": float(row["volume"])
                }
                for _, row in profile_df.iterrows()
            ]

            result = {
                "symbol": symbol,
                "poc": float(poc_price),
                "poc_volume": float(poc_volume),
                "value_area_high": float(value_area["high"]),
                "value_area_low": float(value_area["low"]),
                "profile": profile_data,
                "timestamp": int(datetime.now(timezone.utc).timestamp())
            }

            # Cache profile
            self.volume_profiles[symbol] = result

            logger.debug(f"Volume profile calculated for {symbol}: POC={poc_price:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error calculating volume profile for {symbol}: {e}")
            return None

    def _calculate_value_area(
        self,
        profile_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate Value Area (price range containing X% of volume)

        Args:
            profile_df: Volume profile DataFrame

        Returns:
            {"high": price_high, "low": price_low}
        """
        try:
            total_volume = profile_df["volume"].sum()
            target_volume = total_volume * self.value_area_percent

            # Sort by volume (descending)
            sorted_df = profile_df.sort_values("volume", ascending=False)

            # Accumulate volume until reaching target
            cumulative_volume = 0
            value_area_bins = []

            for _, row in sorted_df.iterrows():
                cumulative_volume += row["volume"]
                value_area_bins.append(row["price_mid"])

                if cumulative_volume >= target_volume:
                    break

            # Value area high/low
            va_high = max(value_area_bins)
            va_low = min(value_area_bins)

            return {"high": va_high, "low": va_low}

        except Exception as e:
            logger.error(f"Error calculating value area: {e}")
            return {"high": 0.0, "low": 0.0}

    async def get_volume_distribution(self) -> Dict[str, float]:
        """
        Get volume distribution across symbols

        Returns:
            {"BTCUSDT": 0.35, "ETHUSDT": 0.25, ...}  # Percentages
        """
        try:
            total_volume = await self.get_total_market_volume()

            if total_volume == 0:
                return {}

            distribution = {}

            for symbol, volumes in self.symbol_volumes.items():
                if volumes:
                    symbol_vol = volumes[-1]
                    distribution[symbol] = symbol_vol / total_volume

            return distribution

        except Exception as e:
            logger.error(f"Error calculating volume distribution: {e}")
            return {}

    async def is_volume_spike(
        self,
        symbol: str,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Detect if current volume is a spike

        Args:
            symbol: Trading symbol
            threshold: Spike threshold (multiplier of average)

        Returns:
            True if volume spike detected
        """
        try:
            if symbol not in self.symbol_volumes:
                return False

            volumes = self.symbol_volumes[symbol]

            if len(volumes) < 2:
                return False

            threshold = threshold or self.spike_threshold

            # Current volume
            current_vol = volumes[-1]

            # Average volume (excluding current)
            avg_vol = np.mean(volumes[:-1])

            # Spike if current > threshold * average
            is_spike = current_vol > (avg_vol * threshold)

            if is_spike:
                logger.info(f"Volume spike detected for {symbol}: {current_vol:.2f} vs avg {avg_vol:.2f}")

            return is_spike

        except Exception as e:
            logger.error(f"Error detecting volume spike for {symbol}: {e}")
            return False

    async def get_volume_trend(self, symbol: str) -> Optional[str]:
        """
        Get volume trend for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            "increasing", "decreasing", or "stable"
        """
        try:
            if symbol not in self.symbol_volumes:
                return None

            volumes = self.symbol_volumes[symbol]

            if len(volumes) < 10:
                return None

            # Get recent volumes
            recent = volumes[-10:]

            # Calculate linear regression slope
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]

            # Determine trend
            avg_vol = np.mean(recent)
            threshold = avg_vol * 0.05  # 5% threshold

            if slope > threshold:
                return "increasing"
            elif slope < -threshold:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Error getting volume trend for {symbol}: {e}")
            return None

    async def get_high_volume_symbols(
        self,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get symbols with highest current volume

        Args:
            top_n: Number of symbols to return

        Returns:
            List of (symbol, volume) tuples
        """
        try:
            symbol_current_volumes = []

            for symbol, volumes in self.symbol_volumes.items():
                if volumes:
                    symbol_current_volumes.append((symbol, volumes[-1]))

            # Sort by volume (descending)
            symbol_current_volumes.sort(key=lambda x: x[1], reverse=True)

            return symbol_current_volumes[:top_n]

        except Exception as e:
            logger.error(f"Error getting high volume symbols: {e}")
            return []

    async def get_volume_statistics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get volume statistics for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Statistics dictionary
        """
        try:
            if symbol not in self.symbol_volumes:
                return None

            volumes = self.symbol_volumes[symbol]

            if len(volumes) == 0:
                return None

            return {
                "symbol": symbol,
                "current": volumes[-1],
                "average": np.mean(volumes),
                "min": np.min(volumes),
                "max": np.max(volumes),
                "std": np.std(volumes),
                "trend": await self.get_volume_trend(symbol),
                "is_spike": await self.is_volume_spike(symbol),
                "samples": len(volumes)
            }

        except Exception as e:
            logger.error(f"Error calculating volume statistics for {symbol}: {e}")
            return None

    async def get_market_volume_summary(self) -> Dict[str, Any]:
        """
        Get market-wide volume summary

        Returns:
            Summary dictionary
        """
        try:
            total_volume = await self.get_total_market_volume()
            distribution = await self.get_volume_distribution()
            high_volume = await self.get_high_volume_symbols(5)

            return {
                "total_volume": total_volume,
                "tracked_symbols": len(self.symbol_volumes),
                "top_5_symbols": [
                    {"symbol": sym, "volume": vol}
                    for sym, vol in high_volume
                ],
                "volume_distribution": distribution,
                "timestamp": int(datetime.now(timezone.utc).timestamp())
            }

        except Exception as e:
            logger.error(f"Error getting market volume summary: {e}")
            return {}


if __name__ == "__main__":
    # Test code
    async def test():
        analyzer = VolumeProfileAnalyzer()
        await analyzer.initialize()

        # Simulate volume data
        symbol = "BTCUSDT"
        for i in range(100):
            volume = np.random.uniform(1000, 5000)
            price = 50000 + np.random.uniform(-1000, 1000)
            await analyzer.add_volume_data(symbol, volume, price)

        # Get volume profile
        profile = await analyzer.get_volume_profile(symbol)
        if profile:
            print(f"Volume Profile for {symbol}:")
            print(f"  POC: ${profile['poc']:.2f}")
            print(f"  Value Area: ${profile['value_area_low']:.2f} - ${profile['value_area_high']:.2f}")

        # Volume stats
        stats = await analyzer.get_volume_statistics(symbol)
        if stats:
            print(f"\nVolume Statistics:")
            print(f"  Current: {stats['current']:.2f}")
            print(f"  Average: {stats['average']:.2f}")
            print(f"  Trend: {stats['trend']}")
            print(f"  Spike: {stats['is_spike']}")

    asyncio.run(test())
