"""
indicators/trend/parabolic_sar.py - Parabolic SAR

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Parabolic SAR (Stop and Reverse) - Parabolic stop and reverse
    A trend-following indicator developed by J. Welles Wilder
    Creates parabolic points below or above the price

    Usage:
    - Determining the trend direction
    - Stop-loss seviyeleri
    - Entry/Exit signals (when the SAR position changes)

Formula:
    SAR(t+1) = SAR(t) + AF × (EP - SAR(t))

    EP (Extreme Point): The highest/lowest point in the current trend.
    AF (Acceleration Factor): Starts at 0.02, increases by 0.02 for each new EP, max 0.20.
    Uptrend: SAR price is below, EP = Highest.
    Downtrend: SAR price is above, EP = Lowest.

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import Any
from indicators.base_indicator import BaseIndicator
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class ParabolicSAR(BaseIndicator):
    """
    Parabolic SAR Indicator

    Tracks trends using parabolic points below or above the price.

    Args:
        af_start: Initial acceleration factor (default: 0.02)
        af_increment: AF increment amount (default: 0.02)
        af_max: Maximum AF (default: 0.20)
    """

    def __init__(
        self,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.20,
        logger=None,
        error_handler=None
    ):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        
        # State for incremental updates
        self._prev_sar = None
        self._prev_trend = None # 1: Up, -1: Down
        self._prev_af = None
        self._prev_ep = None
        self._prev_high = None
        self._prev_low = None
        self._prev_prev_high = None
        self._prev_prev_low = None

        super().__init__(
            name='parabolic_sar',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'af_start': af_start,
                'af_increment': af_increment,
                'af_max': af_max
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return 5  # At least 5 candles are required

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.af_start <= 0 or self.af_start > 1:
            raise InvalidParameterError(
                self.name, 'af_start', self.af_start,
                "AF must be between 0 and 1"
            )
        if self.af_increment <= 0 or self.af_increment > 1:
            raise InvalidParameterError(
                self.name, 'af_increment', self.af_increment,
                "AF increment should be between 0 and 1"
            )
        if self.af_max <= self.af_start or self.af_max > 1:
            raise InvalidParameterError(
                self.name, 'af_max', self.af_max,
                "AF max must be greater than AF start and less than 1"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: SAR, Trend, AF, EP values
        """
        high = data['high'].values
        low = data['low'].values
        
        sar, trend, af, ep = self._calculate_sar_series(high, low)
        
        return pd.DataFrame({
            'sar': sar,
            'trend': trend,
            'af': af,
            'ep': ep
        }, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        Parabolic SAR operates on a state-based system. This function uses historical data.
        son durumu (SAR, trend, AF, EP) hesaplar.

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        # Get the final state using batch calculation
        batch_result = self.calculate_batch(data)

        # Get the last values
        self._prev_sar = batch_result['sar'].iloc[-1]
        self._prev_trend = int(batch_result['trend'].iloc[-1])
        self._prev_af = batch_result['af'].iloc[-1]
        self._prev_ep = batch_result['ep'].iloc[-1]

        # The high/low values of the last 2 bars
        self._prev_high = data['high'].iloc[-1]
        self._prev_low = data['low'].iloc[-1]
        self._prev_prev_high = data['high'].iloc[-2] if len(data) > 1 else None
        self._prev_prev_low = data['low'].iloc[-2] if len(data) > 1 else None

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current SAR values
        """
            
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high = candle['high']
            low = candle['low']
            close = candle['close']
            timestamp_val = int(candle['timestamp']) if 'timestamp' in candle else 0
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high = candle[2] if len(candle) > 2 else 0
            low = candle[3] if len(candle) > 3 else 0
            close = candle[4] if len(candle) > 4 else 0        
        
        # If the state doesn't exist (first candle), we cannot start.
        if self._prev_sar is None:
            # Return None if there is not enough data
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        
        # Previous values
        prev_sar = self._prev_sar
        prev_trend = self._prev_trend
        prev_af = self._prev_af
        prev_ep = self._prev_ep
        
        # Yeni SAR hesapla
        new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
        
        # Trend control and update
        new_trend = prev_trend
        new_af = prev_af
        new_ep = prev_ep
        
        if prev_trend == 1:  # Uptrend
            # SAR should be below the low (the low of the previous 2 bars)
            if self._prev_low is not None:
                new_sar = min(new_sar, self._prev_low)
            if self._prev_prev_low is not None:
                new_sar = min(new_sar, self._prev_prev_low)
            
            # Check for trend reversal (if the current candle's low breaks the SAR)
            if low <= new_sar:
                new_trend = -1
                new_sar = prev_ep
                new_ep = low
                new_af = self.af_start
            else:
                # Uptrend devam
                if high > prev_ep:
                    new_ep = high
                    new_af = min(prev_af + self.af_increment, self.af_max)
                    
        else:  # Downtrend
            # It should stay above the SAR high.
            if self._prev_high is not None:
                new_sar = max(new_sar, self._prev_high)
            if self._prev_prev_high is not None:
                new_sar = max(new_sar, self._prev_prev_high)
                
            # Check for trend change
            if high >= new_sar:
                new_trend = 1
                new_sar = prev_ep
                new_ep = high
                new_af = self.af_start
            else:
                # Downtrend devam
                if low < prev_ep:
                    new_ep = low
                    new_af = min(prev_af + self.af_increment, self.af_max)
        
        # Update state
        self._prev_sar = new_sar
        self._prev_trend = new_trend
        self._prev_af = new_af
        self._prev_ep = new_ep
        
        # Shift history
        self._prev_prev_high = self._prev_high
        self._prev_prev_low = self._prev_low
        self._prev_high = high
        self._prev_low = low
        
        # Result
        current_trend = TrendDirection.UP if new_trend == 1 else TrendDirection.DOWN
        signal = self.get_signal(current_trend)
        
        return IndicatorResult(
            value=round(new_sar, 2),
            timestamp=timestamp_val,
            signal=signal,
            trend=current_trend,
            strength=self._calculate_strength(close, new_sar),
            metadata={
                'af': round(new_af, 4),
                'ep': round(new_ep, 2),
                'current_price': round(close, 2),
                'distance_pct': round(abs((close - new_sar) / new_sar) * 100, 2)
            }
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Parabolic SAR hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: SAR value and trend
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # SAR calculation
        sar, trend, af, ep = self._calculate_sar_series(high, low)

        # Last values
        current_sar = sar[-1]
        current_trend = TrendDirection.UP if trend[-1] == 1 else TrendDirection.DOWN
        current_af = af[-1]
        current_ep = ep[-1]
        
        # Update state (for incremental update)
        self._prev_sar = current_sar
        self._prev_trend = trend[-1]
        self._prev_af = current_af
        self._prev_ep = current_ep
        self._prev_high = high[-1]
        self._prev_low = low[-1]
        
        if len(high) > 1:
            self._prev_prev_high = high[-2]
            self._prev_prev_low = low[-2]
        else:
            self._prev_prev_high = high[-1]
            self._prev_prev_low = low[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal determination
        signal = self.get_signal(current_trend)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'sar': round(current_sar, 2),
                'trend': 1 if current_trend == TrendDirection.UP else -1,
                'af': round(current_af, 4),
                'ep': round(current_ep, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=current_trend,
            strength=self._calculate_strength(close[-1], current_sar),
            metadata={
                'current_price': round(close[-1], 2),
                'distance_pct': round(abs((close[-1] - current_sar) / current_sar) * 100, 2)
            }
        )

    def _calculate_sar_series(self, high: np.ndarray, low: np.ndarray):
        """
        SAR serisini hesapla

        Returns:
            sar, trend, af, ep arrays
        """
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 = up, -1 = down
        af = np.zeros(n)
        ep = np.zeros(n)

        # Initial values
        # Determine the trend based on the high/low of the first 5 candles
        if high[1] > high[0]:
            trend[0] = 1  # Uptrend
            sar[0] = low[0]
            ep[0] = high[0]
        else:
            trend[0] = -1  # Downtrend
            sar[0] = high[0]
            ep[0] = low[0]

        af[0] = self.af_start

        # SAR calculation loop
        for i in range(1, n):
            # Previous values
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]

            # Yeni SAR hesapla
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            # Check if the trend is continuing
            if prev_trend == 1:  # Uptrend
                # The SAR value should remain below the lower bound.
                new_sar = min(new_sar, low[i-1])
                if i > 1:
                    new_sar = min(new_sar, low[i-2])

                # Check for trend change
                if low[i] <= new_sar:
                    # Transition to downtrend
                    trend[i] = -1
                    new_sar = prev_ep  # EP olur yeni SAR
                    ep[i] = low[i]
                    af[i] = self.af_start
                else:
                    # Uptrend devam
                    trend[i] = 1
                    sar[i] = new_sar

                    # Update EP and AF
                    if high[i] > prev_ep:
                        ep[i] = high[i]
                        af[i] = min(prev_af + self.af_increment, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

            else:  # Downtrend (prev_trend == -1)
                # It should stay above the SAR high.
                new_sar = max(new_sar, high[i-1])
                if i > 1:
                    new_sar = max(new_sar, high[i-2])

                # Check for trend change
                if high[i] >= new_sar:
                    # Transition to uptrend
                    trend[i] = 1
                    new_sar = prev_ep
                    ep[i] = high[i]
                    af[i] = self.af_start
                else:
                    # Downtrend devam
                    trend[i] = -1
                    sar[i] = new_sar

                    # Update EP and AF
                    if low[i] < prev_ep:
                        ep[i] = low[i]
                        af[i] = min(prev_af + self.af_increment, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

            sar[i] = new_sar

        return sar, trend, af, ep

    def get_signal(self, trend: TrendDirection) -> SignalType:
        """
        Generate a signal from SAR.

        Args:
            trend: Current trend

        Returns:
            SignalType: BUY (uptrend), SELL (downtrend)
        """
        if trend == TrendDirection.UP:
            return SignalType.BUY
        elif trend == TrendDirection.DOWN:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: Any) -> TrendDirection:
        """Trend has already been calculated"""
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, sar: float) -> float:
        """Calculate signal strength (0-100)"""
        distance_pct = abs((price - sar) / sar * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'af_start': 0.02,
            'af_increment': 0.02,
            'af_max': 0.20
        }

    def _requires_volume(self) -> bool:
        """Parabolic SAR volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ParabolicSAR']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Parabolic SAR indicator test"""

    print("\n" + "="*60)
    print("PARABOLIC SAR TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend change simulation
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 25:
            trend = 0.8  # Increase
        else:
            trend = -0.6  # Decrease
        noise = np.random.randn() * 1.0
        prices.append(prices[-1] + trend + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    psar = ParabolicSAR()
    print(f"   [OK] Created: {psar}")
    print(f"   [OK] Kategori: {psar.category.value}")
    print(f"   [OK] Required period: {psar.get_required_periods()}")

    result = psar(data)
    print(f"   [OK] SAR Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: AF and EP analysis
    print("\n3. AF and EP analysis...")
    print(f"   [OK] Current AF: {result.metadata['af']}")
    print(f"   [OK] Extreme Point: {result.metadata['ep']}")
    print(f"   [OK] Current Price: {result.metadata['current_price']}")
    print(f"   [OK] Distance: {result.metadata['distance_pct']:.2f}%")

    # Test 3: Trend change test
    print("\n4. Trend change test...")
    for i in [15, 25, 35, 45]:
        data_slice = data.iloc[:i+1]
        result = psar.calculate(data_slice)
        print(f"   [OK] Mum {i}: SAR={result.value:.2f}, Trend={result.trend.name}")

    # Test 4: Different parameters
    print("\n5. Different parameter test...")
    for af_max in [0.10, 0.20, 0.30]:
        psar_test = ParabolicSAR(af_max=af_max)
        result = psar_test.calculate(data)
        print(f"   [OK] AF_max={af_max}: SAR={result.value:.2f}")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = psar.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = psar.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
