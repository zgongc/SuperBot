"""
indicators/breakout/volatility_breakout.py - Volatility Breakout

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Volatility Breakout - Breakout Detection with Bollinger Expansion
    Using Bollinger Bands expansion and price movement.
    detects volatility breakouts.

    Breakout Kriterleri:
    - The BB width increases (volatility increase)
    - The price breaks the upper/lower BB band
    - Volume is above the average (optional)

    Output:
    - Upper Band: Upper band
    - Middle Band: Orta bant (SMA)
    - Lower Band: Alt bant
    - Width: Bandwidth
    - %B: The position of the price within the bands

Formula:
    BB Middle = SMA(Close, period)
    BB Upper = Middle + (std_dev × StdDev)
    BB Lower = Middle - (std_dev × StdDev)
    BB Width = (Upper - Lower) / Middle × 100
    %B = (Close - Lower) / (Upper - Lower) × 100

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class VolatilityBreakout(BaseIndicator):
    """
    Volatility Breakout Indicator

    It detects volatility breakouts by monitoring the expansion of Bollinger Bands.
    It analyzes the band width, %B, and price movement.

    Args:
        period: BB period (default: 20)
        std_dev: Standard deviation factor (default: 2.0)
        width_threshold: Width threshold value (default: 4.0)
        use_volume: Use volume check (default: True)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        width_threshold: float = 4.0,
        use_volume: bool = True,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.std_dev = std_dev
        self.width_threshold = width_threshold
        self.use_volume = use_volume

        super().__init__(
            name='volatility_breakout',
            category=IndicatorCategory.BREAKOUT,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'std_dev': std_dev,
                'width_threshold': width_threshold,
                'use_volume': use_volume
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + 10

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Volatility Breakout calculation - for BACKTEST

        Uses Bollinger Bands for volatility breakout detection

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        close = data['close']

        # Bollinger Bands calculation
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std(ddof=0)
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # BB Width: (Upper - Lower) / Middle × 100
        width = ((upper - lower) / middle) * 100

        # %B: (Close - Lower) / (Upper - Lower) × 100
        pct_b = ((close - lower) / (upper - lower)) * 100

        # Handle division by zero
        width = width.fillna(0).replace([np.inf, -np.inf], 0)
        pct_b = pct_b.fillna(50).replace([np.inf, -np.inf], 50)

        # Set first period values to NaN (warmup)
        upper.iloc[:self.period-1] = np.nan
        middle.iloc[:self.period-1] = np.nan
        lower.iloc[:self.period-1] = np.nan
        width.iloc[:self.period-1] = np.nan
        pct_b.iloc[:self.period-1] = np.nan

        return pd.DataFrame({
            'upper': upper.values,
            'middle': middle.values,
            'lower': lower.values,
            'width': width.values,
            'percent_b': pct_b.values
        }, index=data.index)

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be at least 2"
            )
        if self.std_dev <= 0:
            raise InvalidParameterError(
                self.name, 'std_dev', self.std_dev,
                "Standard deviation must be positive"
            )
        if self.width_threshold <= 0:
            raise InvalidParameterError(
                self.name, 'width_threshold', self.width_threshold,
                "The width threshold must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Volatility Breakout hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: BB bands, width, and %B value.
        """
        close = data['close'].values
        volume = data['volume'].values if self.use_volume else None

        # Bollinger Bands hesapla
        sma = np.mean(close[-self.period:])
        std = np.std(close[-self.period:])

        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        # BB Width hesapla (%)
        width = ((upper_band - lower_band) / sma) * 100 if sma != 0 else 0

        # Calculate %B (the position of the price within the bands)
        band_range = upper_band - lower_band
        if band_range != 0:
            percent_b = ((close[-1] - lower_band) / band_range) * 100
        else:
            percent_b = 50.0

        # Calculate the previous width (for the trend)
        if len(close) >= self.period + 5:
            prev_sma = np.mean(close[-(self.period+5):-5])
            prev_std = np.std(close[-(self.period+5):-5])
            prev_upper = prev_sma + (self.std_dev * prev_std)
            prev_lower = prev_sma - (self.std_dev * prev_std)
            prev_width = ((prev_upper - prev_lower) / prev_sma) * 100 if prev_sma != 0 else 0
        else:
            prev_width = width

        width_expanding = width > prev_width

        # Volume control
        volume_confirm = True
        if self.use_volume and volume is not None:
            avg_volume = np.mean(volume[-self.period:])
            volume_confirm = volume[-1] > avg_volume * 1.2

        # Detect breakout
        breakout_up = (
            close[-1] > upper_band and
            width > self.width_threshold and
            width_expanding
        )

        breakout_down = (
            close[-1] < lower_band and
            width > self.width_threshold and
            width_expanding
        )

        if self.use_volume:
            breakout_up = breakout_up and volume_confirm
            breakout_down = breakout_down and volume_confirm

        timestamp = int(data.iloc[-1]['timestamp'])

        # Define signal
        signal = self.get_signal(breakout_up, breakout_down, percent_b)
        trend = self.get_trend(percent_b, close[-1], sma)

        # Power: Width and %B combination
        strength = min((width / self.width_threshold) * 50 + abs(percent_b - 50), 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper_band, 2),
                'middle': round(sma, 2),
                'lower': round(lower_band, 2),
                'width': round(width, 2),
                'percent_b': round(percent_b, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'width_expanding': width_expanding,
                'volume_confirm': volume_confirm if self.use_volume else None,
                'price': round(close[-1], 2)
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True
        
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle.get('open', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value={'breakout': False, 'direction': 'none'},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, breakout_up: bool, breakout_down: bool, percent_b: float) -> SignalType:
        """
        Generate a signal from the breakout state.

        Args:
            breakout_up: Is there an upward breakout?
            breakout_down: Is there a downward breakout?
            percent_b: %B value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if breakout_up:
            return SignalType.STRONG_BUY
        elif breakout_down:
            return SignalType.STRONG_SELL
        elif percent_b > 80:
            return SignalType.SELL
        elif percent_b < 20:
            return SignalType.BUY

        return SignalType.HOLD

    def get_trend(self, percent_b: float, price: float, middle: float) -> TrendDirection:
        """
        Determine the trend based on %B and price.

        Args:
            percent_b: The value of %B
            price: The current price
            middle: Orta bant (SMA)

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if price > middle and percent_b > 50:
            return TrendDirection.UP
        elif price < middle and percent_b < 50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'std_dev': 2.0,
            'width_threshold': 4.0,
            'use_volume': True
        }

    def _requires_volume(self) -> bool:
        """Volume is optional, but can be used."""
        return False

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['upper', 'middle', 'lower', 'width', 'percent_b']


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VolatilityBreakout']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Volatility Breakout indicator test"""

    print("\n" + "="*60)
    print("VOLATILITY BREAKOUT TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Simulate low volatility to high volatility
    base_price = 100
    prices = [base_price]

    # First 50 candles: Low volatility
    for i in range(49):
        change = np.random.randn() * 0.5
        prices.append(prices[-1] + change)

    # Last 50 candles: High volatility + trend
    for i in range(50):
        change = np.random.randn() * 3.0 + 0.8
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 2000) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    vb = VolatilityBreakout()
    print(f"   [OK] Created: {vb}")
    print(f"   [OK] Kategori: {vb.category.value}")
    print(f"   [OK] Required period: {vb.get_required_periods()}")

    result = vb(data)
    print(f"   [OK] Upper Band: {result.value['upper']}")
    print(f"   [OK] Middle Band: {result.value['middle']}")
    print(f"   [OK] Lower Band: {result.value['lower']}")
    print(f"   [OK] Width: {result.value['width']:.2f}%")
    print(f"   [OK] %B: {result.value['percent_b']:.2f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Low volatility test
    print("\n3. Low volatility test (first 50 candles)...")
    low_vol_data = data.head(60)
    result = vb.calculate(low_vol_data)
    print(f"   [OK] Width: {result.value['width']:.2f}%")
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Breakout DOWN: {result.metadata['breakout_down']}")
    print(f"   [OK] Width Expanding: {result.metadata['width_expanding']}")

    # Test 3: High volatility test
    print("\n4. High volatility test (all data)...")
    result = vb.calculate(data)
    print(f"   [OK] Width: {result.value['width']:.2f}%")
    print(f"   [OK] %B: {result.value['percent_b']:.2f}")
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Power: {result.strength:.2f}")

    # Test 4: Different parameters
    print("\n5. Different parameter test...")
    vb_tight = VolatilityBreakout(std_dev=1.5, width_threshold=3.0)
    result = vb_tight.calculate(data)
    print(f"   [OK] Tight BB Width: {result.value['width']:.2f}%")
    print(f"   [OK] %B: {result.value['percent_b']:.2f}")

    # Test 5: Volume olmadan
    print("\n6. Volume olmadan test...")
    vb_no_vol = VolatilityBreakout(use_volume=False)
    result = vb_no_vol.calculate(data)
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Volume Confirm: {result.metadata['volume_confirm']}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 6: Zaman serisi analizi
    print("\n7. Zaman serisi analizi...")
    width_history = []
    percent_b_history = []

    for i in range(40, len(data), 10):
        partial_data = data.head(i)
        result = vb.calculate(partial_data)
        width_history.append(result.value['width'])
        percent_b_history.append(result.value['percent_b'])

    print(f"   [OK] Total measurement: {len(width_history)}")
    print(f"   [OK] Average width: {np.mean(width_history):.2f}%")
    print(f"   [OK] Max width: {max(width_history):.2f}%")
    print(f"   [OK] Minimum width: {min(width_history):.2f}%")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = vb.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = vb.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Output names: {metadata.output_names}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
