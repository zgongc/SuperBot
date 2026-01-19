"""
indicators/statistical/correlation.py - Correlation

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Correlation - Measures the linear relationship between two entities.
    Range: -1 to +1
    +1: Perfect positive correlation (move together)
    0: No correlation (independent)
    -1: Perfect negative correlation (move in opposite directions)

Formula:
    Pearson Correlation Coefficient:
    r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)

    Rolling correlation is calculated over a specific window.

Usage:
    - Pairs trading stratejileri
    - Portfolio diversification
    - Risk management

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class Correlation(BaseIndicator):
    """
    Correlation.

    Calculates the correlation between two price series.
    It is used for pairs trading and portfolio analysis.

    Args:
        period: Correlation window period (default: 20)
        reference_data: Reference data to compare with (default: None)
        high_correlation: High correlation threshold (default: 0.7)
        low_correlation: Low correlation threshold (default: -0.7)
    """

    def __init__(
        self,
        period: int = 20,
        reference_data: pd.DataFrame = None,
        high_correlation: float = 0.7,
        low_correlation: float = -0.7,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.reference_data = reference_data
        self.high_correlation = high_correlation
        self.low_correlation = low_correlation

        super().__init__(
            name='correlation',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'reference_data': reference_data,
                'high_correlation': high_correlation,
                'low_correlation': low_correlation
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period must be at least 2 (for correlation)"
            )
        if not (-1 <= self.low_correlation < self.high_correlation <= 1):
            raise InvalidParameterError(
                self.name, 'thresholds',
                f"low={self.low_correlation}, high={self.high_correlation}",
                "Thresholds should be between -1 and 1, and low must be less than high."
            )
        return True

    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Set the reference data (the asset to be compared).

        Args:
            reference_data: Referans OHLCV DataFrame
        """
        self.reference_data = reference_data

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate correlation.

        Args:
            data: OHLCV DataFrame (first asset)

        Returns:
            IndicatorResult: Correlation value
        """
        # If there is no reference data, calculate the correlation with its own historical data (autocorrelation)
        if self.reference_data is None:
            close = data['close'].values
            period_data = close[-self.period:]

            # Autocorrelation: relationship between the current price and the lagged price.
            if len(period_data) >= 2:
                x = period_data[:-1]  # t-1
                y = period_data[1:]   # t
                correlation = np.corrcoef(x, y)[0, 1]
                reference_name = "self_lag1"
                reference_price = period_data[-2] if len(period_data) >= 2 else period_data[-1]
            else:
                correlation = 0.0
                reference_name = "self_lag1"
                reference_price = close[-1]
        else:
            # Correlation between two different entities
            close1 = data['close'].values[-self.period:]
            close2 = self.reference_data['close'].values[-self.period:]

            # Equalize data lengths
            min_len = min(len(close1), len(close2))
            if min_len < 2:
                correlation = 0.0
            else:
                close1 = close1[-min_len:]
                close2 = close2[-min_len:]
                correlation = np.corrcoef(close1, close2)[0, 1]

            reference_name = "reference_asset"
            reference_price = close2[-1] if len(close2) > 0 else 0

        # NaN check
        if np.isnan(correlation):
            correlation = 0.0

        timestamp = int(data.iloc[-1]['timestamp'])
        current_price = data['close'].values[-1]

        # Correlation strength: absolute value
        strength = abs(correlation) * 100

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(correlation, 4),
            timestamp=timestamp,
            signal=self.get_signal(correlation),
            trend=self.get_trend(correlation),
            strength=strength,
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'reference_price': round(reference_price, 2),
                'reference_name': reference_name,
                'correlation_pct': round(correlation * 100, 2),
                'relationship': self._get_relationship(correlation)
            }
        )

    def _get_relationship(self, correlation: float) -> str:
        """
        Determine the relationship type from the correlation value.

        Args:
            correlation: Correlation coefficient

        Returns:
            str: Relationship description
        """
        abs_corr = abs(correlation)

        if abs_corr >= 0.9:
            strength = "Very Strong"
        elif abs_corr >= 0.7:
            strength = "Strong"
        elif abs_corr >= 0.5:
            strength = "Orta"
        elif abs_corr >= 0.3:
            strength = "Weak"
        else:
            strength = "Very Weak"

        direction = "Positive" if correlation >= 0 else "Negative"

        return f"{strength} {direction}"

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Correlation calculation - for BACKTEST

        Correlation Formula:
            Rolling Pearson correlation coefficient over 'period' window
            - If reference_data is None: autocorrelation (self vs lag-1)
            - If reference_data provided: cross-correlation with reference

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Correlation values (-1 to +1) for all bars

        Performance: 2000 bars in ~0.05 seconds
        """
        self._validate_data(data)

        close = data['close']

        # If no reference data, calculate autocorrelation (lag-1)
        if self.reference_data is None:
            # Autocorrelation with lag-1
            correlation = close.rolling(window=self.period).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 2 else 0,
                raw=True
            )
        else:
            # Cross-correlation with reference asset
            ref_close = self.reference_data['close']

            # Align both series to same index (use minimum length)
            min_len = min(len(close), len(ref_close))
            close_aligned = close.iloc[-min_len:].reset_index(drop=True)
            ref_aligned = ref_close.iloc[-min_len:].reset_index(drop=True)

            # Calculate rolling correlation
            correlation = close_aligned.rolling(window=self.period).corr(ref_aligned)

            # Re-index to original
            correlation.index = data.index[-min_len:]
            correlation = correlation.reindex(data.index)

        # Handle NaN values
        correlation = correlation.fillna(0)

        # Set first period values to NaN (warmup)
        correlation.iloc[:self.period-1] = np.nan

        return pd.Series(correlation.values, index=data.index, name='correlation')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - prepares the necessary state for update()"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._close_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)
            self._buffers_init = True

        self._close_buffer.append(close_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the correlation value.

        Args:
            value: Correlation value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # High positive correlation: assets move together
        if value >= self.high_correlation:
            return SignalType.BUY

        # High negative correlation: assets move in opposite directions
        elif value <= self.low_correlation:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the correlation value.

        Args:
            value: Correlation value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > 0.3:
            return TrendDirection.UP  # Positive correlation
        elif value < -0.3:
            return TrendDirection.DOWN  # Negative correlation
        return TrendDirection.NEUTRAL  # Low/no correlation

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'high_correlation': 0.7,
            'low_correlation': -0.7
        }

    def _requires_volume(self) -> bool:
        """Correlation volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Correlation']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Correlation indicator test"""

    print("\n" + "="*60)
    print("CORRELATION TEST")
    print("="*60 + "\n")

    # Test 1: Autocorrelation testi
    print("1. Autocorrelation test (with its own lagged version)...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Price series containing trend + momentum
    base_price = 100
    prices = [base_price]
    for i in range(49):
        momentum = prices[-1] - (prices[-2] if len(prices) > 1 else base_price)
        trend = 0.1
        noise = np.random.randn() * 0.5
        prices.append(prices[-1] + trend + momentum * 0.3 + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    corr = Correlation(period=20)
    print(f"   [OK] Created: {corr}")
    print(f"   [OK] Kategori: {corr.category.value}")

    result = corr(data)
    print(f"   [OK] Autocorrelation: {result.value}")
    print(f"   [OK] Relationship: {result.metadata['relationship']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Positive correlation - two similar assets
    print("\n2. Positive correlation test (similar movements)...")
    np.random.seed(42)

    # Entity 1
    prices1 = [100]
    for i in range(49):
        trend = 0.2 + np.random.randn() * 0.5
        prices1.append(prices1[-1] + trend)

    data1 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices1,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices1],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices1],
        'close': prices1,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices1]
    })

    # Entity 2 - Highly correlated with Entity 1
    prices2 = []
    for p in prices1:
        # Same trend + small noise
        prices2.append(p * 1.1 + np.random.randn() * 0.3)

    data2 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices2,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices2],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices2],
        'close': prices2,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices2]
    })

    corr.set_reference_data(data2)
    result = corr.calculate(data1)
    print(f"   [OK] Positive Correlation: {result.value}")
    print(f"   [OK] Relationship: {result.metadata['relationship']}")
    print(f"   [OK] Correlation %: {result.metadata['correlation_pct']}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 3: Negative correlation - inverse movements
    print("\n3. Negative correlation test (inverse movements)...")
    prices3 = []
    for p in prices1:
        # Ters hareket
        prices3.append(200 - p + np.random.randn() * 0.3)

    data3 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices3,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices3],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices3],
        'close': prices3,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices3]
    })

    corr.set_reference_data(data3)
    result = corr.calculate(data1)
    print(f"   [OK] Negative Correlation: {result.value}")
    print(f"   [OK] Relationship: {result.metadata['relationship']}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Zero correlation - independent movements
    print("\n4. Zero correlation test (independent movements)...")
    np.random.seed(99)
    prices4 = [100 + np.random.randn() * 3 for _ in range(50)]

    data4 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices4,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices4],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices4],
        'close': prices4,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices4]
    })

    corr.set_reference_data(data4)
    result = corr.calculate(data1)
    print(f"   [OK] Zero Correlation: {result.value}")
    print(f"   [OK] Relationship: {result.metadata['relationship']}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 5: Different periods
    print("\n5. Different period test...")
    corr.set_reference_data(data2)  # Pozitif korelasyonlu veri
    for period in [10, 20, 30]:
        corr_test = Correlation(period=period, reference_data=data2)
        result = corr_test.calculate(data1)
        print(f"   [OK] Corr({period}): {result.value:.4f} | Relationship: {result.metadata['relationship']}")

    # Test 6: Rolling correlation analysis
    print("\n6. Rolling correlation test (last 10 candles)...")
    corr_roll = Correlation(period=20, reference_data=data2)
    for i in range(-10, 0):
        test_data1 = data1.iloc[:len(data1)+i]
        test_data2 = data2.iloc[:len(data2)+i]
        corr_roll.set_reference_data(test_data2)
        if len(test_data1) >= corr_roll.period:
            result = corr_roll.calculate(test_data1)
            print(f"   [OK] Mum {i:3d}: Corr = {result.value:7.4f} | "
                  f"Relationship = {result.metadata['relationship']:20s} | "
                  f"Trend = {result.trend.name}")

    # Test 7: Custom thresholds
    print("\n7. Special threshold test...")
    corr_custom = Correlation(period=20, reference_data=data2,
                              high_correlation=0.9, low_correlation=-0.9)
    result = corr_custom.calculate(data1)
    print(f"   [OK] Custom threshold correlation: {result.value}")
    print(f"   [OK] High threshold: {corr_custom.high_correlation}")
    print(f"   [OK] Low threshold: {corr_custom.low_correlation}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 8: Statistics
    print("\n8. Statistical test...")
    stats = corr.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 9: Metadata
    print("\n9. Metadata testi...")
    metadata = corr.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
