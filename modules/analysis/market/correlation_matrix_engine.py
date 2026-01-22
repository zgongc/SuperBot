#!/usr/bin/env python3
"""
analysis/correlation_matrix_engine.py
SuperBot - Correlation Matrix Engine

Coin correlation analysis:
- BTC dominance tracking
- Coin-coin correlation matrix
- Market leader detection
- Correlation-based trading signals

Features:
- Real-time correlation calculation
- Rolling window correlation (Pearson, Spearman)
- BTC dominance from market caps
- Correlation heatmap data
- EventBus integration

Usage:
    engine = CorrelationMatrixEngine(config, data_manager, event_bus)
    await engine.initialize()

    # BTC dominance
    dominance = await engine.get_btc_dominance()

    # Correlation matrix
    matrix = await engine.get_correlation_matrix(["BTCUSDT", "ETHUSDT", "BNBUSDT"])

    # Correlation with BTC
    corr = await engine.get_correlation_with_btc("ETHUSDT")

Dependencies:
    - numpy>=1.24.0
    - pandas>=2.0.0
    - DataManager
    - EventBus
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


class CorrelationMatrixEngine:
    """
    Correlation Matrix Engine for market analysis

    Features:
    - BTC dominance tracking
    - Coin correlation matrix
    - Market leader detection
    - Correlation-based signals
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_manager=None,
        event_bus=None
    ):
        """
        Initialize CorrelationMatrixEngine

        Args:
            config: Configuration dictionary
            data_manager: DataManager instance
            event_bus: EventBus instance
        """
        self.config = config or {}
        self.data_manager = data_manager
        self.event_bus = event_bus

        # Configuration
        self.window_size = self.config.get("correlation_window", 100)  # 100 candles
        self.update_interval = self.config.get("correlation_update_interval", 300)  # 5 minutes
        self.min_periods = self.config.get("correlation_min_periods", 30)
        self.correlation_method = self.config.get("correlation_method", "pearson")  # pearson, spearman

        # BTC dominance
        self.btc_dominance_history: List[Dict[str, Any]] = []
        self.current_btc_dominance: Optional[float] = None

        # Correlation data
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.price_data: Dict[str, pd.Series] = {}
        self.last_update: Optional[datetime] = None

        # State
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info("CorrelationMatrixEngine initialized")

    async def initialize(self):
        """Initialize engine and load historical data"""
        try:
            # Load historical correlation data if available
            if self.data_manager:
                # TODO: Load from database when implemented
                pass

            logger.info("CorrelationMatrixEngine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CorrelationMatrixEngine: {e}")
            raise

    async def start(self):
        """Start periodic correlation updates"""
        if self.is_running:
            logger.warning("CorrelationMatrixEngine already running")
            return

        self.is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("CorrelationMatrixEngine started")

    async def stop(self):
        """Stop periodic updates"""
        self.is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("CorrelationMatrixEngine stopped")

    async def _update_loop(self):
        """Periodic correlation update loop"""
        while self.is_running:
            try:
                # Update correlation matrix for tracked symbols
                # TODO: Get symbols from SymbolsManager
                symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Example

                await self.update_correlation_matrix(symbols)
                await self.update_btc_dominance()

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in correlation update loop: {e}")
                await asyncio.sleep(60)

    async def update_price_data(
        self,
        symbol: str,
        prices: List[float],
        timestamps: Optional[List[int]] = None
    ):
        """
        Update price data for a symbol

        Args:
            symbol: Trading symbol
            prices: List of prices
            timestamps: Optional timestamps
        """
        try:
            if timestamps:
                series = pd.Series(prices, index=pd.to_datetime(timestamps, unit='s'))
            else:
                series = pd.Series(prices)

            self.price_data[symbol] = series
            logger.debug(f"Updated price data for {symbol}: {len(prices)} data points")

        except Exception as e:
            logger.error(f"Error updating price data for {symbol}: {e}")

    async def get_correlation_matrix(
        self,
        symbols: List[str],
        method: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix for given symbols

        Args:
            symbols: List of trading symbols
            method: Correlation method ("pearson" or "spearman")

        Returns:
            Correlation matrix DataFrame or None
        """
        try:
            method = method or self.correlation_method

            # Check if we have price data for all symbols
            missing_symbols = [s for s in symbols if s not in self.price_data]
            if missing_symbols:
                logger.warning(f"Missing price data for symbols: {missing_symbols}")
                return None

            # Create DataFrame from price data
            df = pd.DataFrame({
                symbol: self.price_data[symbol].tail(self.window_size)
                for symbol in symbols
            })

            # Calculate returns (percentage change)
            returns = df.pct_change().dropna()

            if len(returns) < self.min_periods:
                logger.warning(f"Insufficient data for correlation: {len(returns)} < {self.min_periods}")
                return None

            # Calculate correlation matrix
            corr_matrix = returns.corr(method=method)

            self.correlation_matrix = corr_matrix
            self.last_update = datetime.now(timezone.utc)

            logger.info(f"Correlation matrix calculated for {len(symbols)} symbols")

            # Publish event
            if self.event_bus:
                await self.event_bus.publish("correlation.updated", {
                    "symbols": symbols,
                    "timestamp": int(self.last_update.timestamp()),
                    "method": method
                })

            return corr_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None

    async def update_correlation_matrix(self, symbols: List[str]):
        """Update correlation matrix (wrapper for periodic updates)"""
        await self.get_correlation_matrix(symbols)

    async def get_correlation_with_btc(
        self,
        symbol: str,
        method: Optional[str] = None
    ) -> Optional[float]:
        """
        Get correlation of a symbol with BTC

        Args:
            symbol: Trading symbol
            method: Correlation method

        Returns:
            Correlation coefficient or None
        """
        try:
            if symbol == "BTCUSDT":
                return 1.0  # BTC correlation with itself is 1

            # Calculate correlation with BTC
            symbols = ["BTCUSDT", symbol]
            corr_matrix = await self.get_correlation_matrix(symbols, method)

            if corr_matrix is not None:
                return corr_matrix.loc["BTCUSDT", symbol]

            return None

        except Exception as e:
            logger.error(f"Error calculating BTC correlation for {symbol}: {e}")
            return None

    async def get_btc_dominance(self) -> Optional[float]:
        """
        Calculate BTC dominance

        BTC Dominance = (BTC Market Cap / Total Crypto Market Cap) × 100

        Returns:
            BTC dominance percentage or None
        """
        try:
            # TODO: Implement actual market cap calculation
            # For now, return placeholder

            # This would typically fetch from CoinGecko or calculate from trading volumes
            dominance = 50.0  # Placeholder

            self.current_btc_dominance = dominance

            # Add to history
            self.btc_dominance_history.append({
                "value": dominance,
                "timestamp": int(datetime.now(timezone.utc).timestamp())
            })

            # Keep only last 1000 records
            if len(self.btc_dominance_history) > 1000:
                self.btc_dominance_history = self.btc_dominance_history[-1000:]

            return dominance

        except Exception as e:
            logger.error(f"Error calculating BTC dominance: {e}")
            return None

    async def update_btc_dominance(self):
        """Update BTC dominance (wrapper for periodic updates)"""
        dominance = await self.get_btc_dominance()

        if dominance and self.event_bus:
            await self.event_bus.publish("btc.dominance.updated", {
                "value": dominance,
                "timestamp": int(datetime.now(timezone.utc).timestamp())
            })

    def get_btc_dominance_trend(self) -> Optional[str]:
        """
        Get BTC dominance trend

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(self.btc_dominance_history) < 3:
            return None

        try:
            # Get last 3 dominance values
            recent = self.btc_dominance_history[-3:]
            values = [r["value"] for r in recent]

            # Calculate trend
            avg_change = (values[-1] - values[0]) / len(values)

            if avg_change > 0.5:
                return "increasing"
            elif avg_change < -0.5:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Error analyzing BTC dominance trend: {e}")
            return None

    def is_altcoin_season(self) -> bool:
        """
        Check if it's altcoin season based on BTC dominance

        Altcoin season = BTC dominance decreasing

        Returns:
            True if altcoin season
        """
        trend = self.get_btc_dominance_trend()
        return trend == "decreasing" if trend else False

    def get_highly_correlated_pairs(
        self,
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated symbol pairs

        Args:
            threshold: Correlation threshold (0-1)

        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if self.correlation_matrix is None:
            return []

        try:
            pairs = []
            symbols = self.correlation_matrix.columns

            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    corr = self.correlation_matrix.loc[sym1, sym2]
                    if abs(corr) >= threshold:
                        pairs.append((sym1, sym2, corr))

            # Sort by absolute correlation
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            return pairs

        except Exception as e:
            logger.error(f"Error finding correlated pairs: {e}")
            return []

    def get_market_leaders(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify market leaders (symbols with low correlation to others)

        Market leaders move independently

        Args:
            top_n: Number of leaders to return

        Returns:
            List of (symbol, avg_correlation) tuples
        """
        if self.correlation_matrix is None:
            return []

        try:
            # Calculate average correlation for each symbol (excluding self)
            avg_correlations = []

            for symbol in self.correlation_matrix.columns:
                # Get correlations with other symbols (exclude self)
                corr_values = self.correlation_matrix[symbol].drop(symbol)
                avg_corr = abs(corr_values).mean()
                avg_correlations.append((symbol, avg_corr))

            # Sort by average correlation (ascending - lower is more independent)
            avg_correlations.sort(key=lambda x: x[1])

            return avg_correlations[:top_n]

        except Exception as e:
            logger.error(f"Error finding market leaders: {e}")
            return []

    def get_correlation_signal(
        self,
        symbol: str,
        btc_correlation_threshold: float = 0.7
    ) -> Optional[str]:
        """
        Get trading signal based on BTC correlation

        Logic:
        - High BTC correlation + BTC uptrend → Follow BTC
        - Low BTC correlation → Independent move

        Args:
            symbol: Trading symbol
            btc_correlation_threshold: Threshold for high correlation

        Returns:
            "follow_btc", "independent", or None
        """
        if symbol == "BTCUSDT":
            return None

        try:
            if self.correlation_matrix is None:
                return None

            if "BTCUSDT" not in self.correlation_matrix.columns:
                return None

            if symbol not in self.correlation_matrix.columns:
                return None

            btc_corr = self.correlation_matrix.loc["BTCUSDT", symbol]

            if abs(btc_corr) >= btc_correlation_threshold:
                return "follow_btc"
            else:
                return "independent"

        except Exception as e:
            logger.error(f"Error getting correlation signal for {symbol}: {e}")
            return None

    def get_correlation_stats(self) -> Dict[str, Any]:
        """
        Get correlation statistics

        Returns:
            Statistics dictionary
        """
        if self.correlation_matrix is None:
            return {}

        try:
            # Get upper triangle of correlation matrix (exclude diagonal)
            corr_values = self.correlation_matrix.values[np.triu_indices_from(
                self.correlation_matrix.values, k=1
            )]

            return {
                "average_correlation": float(np.mean(corr_values)),
                "min_correlation": float(np.min(corr_values)),
                "max_correlation": float(np.max(corr_values)),
                "std_correlation": float(np.std(corr_values)),
                "btc_dominance": self.current_btc_dominance,
                "btc_dominance_trend": self.get_btc_dominance_trend(),
                "is_altcoin_season": self.is_altcoin_season(),
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "symbols_count": len(self.correlation_matrix.columns)
            }

        except Exception as e:
            logger.error(f"Error calculating correlation stats: {e}")
            return {}

    def get_heatmap_data(self) -> Optional[Dict[str, Any]]:
        """
        Get correlation matrix data formatted for heatmap visualization

        Returns:
            {
                "symbols": ["BTCUSDT", "ETHUSDT", ...],
                "matrix": [[1.0, 0.85, ...], ...],
                "timestamp": 1697462400
            }
        """
        if self.correlation_matrix is None:
            return None

        try:
            return {
                "symbols": self.correlation_matrix.columns.tolist(),
                "matrix": self.correlation_matrix.values.tolist(),
                "timestamp": int(self.last_update.timestamp()) if self.last_update else None
            }

        except Exception as e:
            logger.error(f"Error getting heatmap data: {e}")
            return None


if __name__ == "__main__":
    # Test code
    async def test():
        engine = CorrelationMatrixEngine()
        await engine.initialize()

        # Simulate price data
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        for symbol in symbols:
            prices = np.random.randn(100).cumsum() + 50000
            await engine.update_price_data(symbol, prices.tolist())

        # Calculate correlation
        matrix = await engine.get_correlation_matrix(symbols)
        if matrix is not None:
            print("Correlation Matrix:")
            print(matrix)

            print("\nCorrelation Stats:")
            print(engine.get_correlation_stats())

            print("\nHighly Correlated Pairs:")
            print(engine.get_highly_correlated_pairs(threshold=0.5))

    asyncio.run(test())
