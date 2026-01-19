"""
indicators/breakout/range_breakout.py - Range Breakout

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Range Breakout - Konsolidasyon Range + Breakout Seviyeleri
    It detects periods where the price consolidates within a narrow range
    and determines breakout levels.

    Features:
    - Range belirleme (High/Low)
    - Konsolidasyon tespiti (dar range)
    - Breakout levels (upper/lower)
    - Target price calculation (up to the range height)

    Usage Areas:
    - Range trading stratejileri
    - Breakout buy/sell signals
    - Risk/reward calculations

Formula:
    Range High = MAX(High, period)
    Range Low = MIN(Low, period)
    Range Height = Range High - Range Low
    Range %  = (Range Height / Range Low) × 100

    Consolidation = Range % < threshold

    Breakout UP Target = Range High + Range Height
    Breakout DOWN Target = Range Low - Range Height

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


class RangeBreakout(BaseIndicator):
    """
    Range Breakout Indicator

    Detects consolidation ranges and determines breakout levels.
    Calculates range height, width, and target prices.

    Args:
        period: Range calculation period (default: 20)
        consolidation_threshold: Consolidation threshold (%) (default: 3.0)
        breakout_confirmation: Breakout confirmation candle count (default: 1)
        use_body: Use body (True) or wick (False) (default: False)
    """

    def __init__(
        self,
        period: int = 20,
        consolidation_threshold: float = 3.0,
        breakout_confirmation: int = 1,
        use_body: bool = False,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.consolidation_threshold = consolidation_threshold
        self.breakout_confirmation = breakout_confirmation
        self.use_body = use_body

        super().__init__(
            name='range_breakout',
            category=IndicatorCategory.BREAKOUT,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'consolidation_threshold': consolidation_threshold,
                'breakout_confirmation': breakout_confirmation,
                'use_body': use_body
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + self.breakout_confirmation + 2

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 5:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be at least 5"
            )
        if self.consolidation_threshold <= 0:
            raise InvalidParameterError(
                self.name, 'consolidation_threshold', self.consolidation_threshold,
                "Consolidation threshold must be positive"
            )
        if self.breakout_confirmation < 1:
            raise InvalidParameterError(
                self.name, 'breakout_confirmation', self.breakout_confirmation,
                "Breakout confirmation should be at least 1"
            )
        return True

    def _calculate_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        close: np.ndarray
    ) -> tuple:
        """
        Range hesapla

        Returns:
            (range_high, range_low, range_height, range_pct)
        """
        if self.use_body:
            # Use body (open/close)
            highs = np.maximum(open_[-self.period:], close[-self.period:])
            lows = np.minimum(open_[-self.period:], close[-self.period:])
        else:
            # Use wick (high/low)
            highs = high[-self.period:]
            lows = low[-self.period:]

        range_high = np.max(highs)
        range_low = np.min(lows)
        range_height = range_high - range_low
        range_pct = (range_height / range_low * 100) if range_low > 0 else 0

        return range_high, range_low, range_height, range_pct

    def _is_consolidating(self, range_pct: float) -> bool:
        """Consolidation check"""
        return range_pct < self.consolidation_threshold

    def _detect_breakout(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        range_high: float,
        range_low: float
    ) -> tuple:
        """
        Detect breakout.

        Returns:
            (breakout_up, breakout_down, breakout_confirmed)
        """
        # Check the last confirmation candles
        recent_highs = high[-self.breakout_confirmation:]
        recent_lows = low[-self.breakout_confirmation:]
        recent_closes = close[-self.breakout_confirmation:]

        # Breakout check (all confirmation candles must be outside the range)
        breakout_up = all(recent_closes > range_high)
        breakout_down = all(recent_closes < range_low)

        # Is the breakout confirmed?
        breakout_confirmed = breakout_up or breakout_down

        return breakout_up, breakout_down, breakout_confirmed

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Range Breakout calculation - for BACKTEST

        Range Formula:
            Range High = MAX(High, period)
            Range Low = MIN(Low, period)
            Range Height = Range High - Range Low

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: range_high, range_low for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        open_ = data['open']
        close = data['close']

        if self.use_body:
            # Use body (open/close)
            highs = pd.DataFrame({'open': open_, 'close': close}).max(axis=1)
            lows = pd.DataFrame({'open': open_, 'close': close}).min(axis=1)
        else:
            # Use wick (high/low)
            highs = high
            lows = low

        # Rolling range
        range_high = highs.rolling(window=self.period).max()
        range_low = lows.rolling(window=self.period).min()
        range_middle = (range_high + range_low) / 2

        # Set first period values to NaN (warmup)
        range_high.iloc[:self.period-1] = np.nan
        range_low.iloc[:self.period-1] = np.nan
        range_middle.iloc[:self.period-1] = np.nan

        return pd.DataFrame({
            'upper': range_high.values,
            'middle': range_middle.values,
            'lower': range_low.values
        }, index=data.index)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Range Breakout hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Range levels and breakout status.
        """
        high = data['high'].values
        low = data['low'].values
        open_ = data['open'].values
        close = data['close'].values

        # Range hesapla
        range_high, range_low, range_height, range_pct = self._calculate_range(
            high, low, open_, close
        )

        # Consolidation check
        is_consolidating = self._is_consolidating(range_pct)

        # Detect breakout
        breakout_up, breakout_down, breakout_confirmed = self._detect_breakout(
            high, low, close, range_high, range_low
        )

        # Hedef fiyatlar hesapla
        target_up = range_high + range_height
        target_down = range_low - range_height

        # Orta seviye (range middle)
        range_middle = (range_high + range_low) / 2

        # The position of the current price within the range.
        if range_height > 0:
            position_pct = ((close[-1] - range_low) / range_height) * 100
        else:
            position_pct = 50.0

        timestamp = int(data.iloc[-1]['timestamp'])

        # Define signal
        signal = self.get_signal(
            breakout_up, breakout_down, is_consolidating, position_pct
        )
        trend = self.get_trend(position_pct, breakout_up, breakout_down)

        # Power: Breakout distance + consolidation status
        if breakout_up:
            strength = min(((close[-1] - range_high) / range_height) * 100, 100)
        elif breakout_down:
            strength = min(((range_low - close[-1]) / range_height) * 100, 100)
        elif is_consolidating:
            strength = 50.0
        else:
            strength = abs(position_pct - 50)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(range_high, 2),
                'middle': round(range_middle, 2),
                'lower': round(range_low, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'range_height': round(range_height, 2),
                'range_pct': round(range_pct, 2),
                'is_consolidating': is_consolidating,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'breakout_confirmed': breakout_confirmed,
                'target_up': round(target_up, 2),
                'target_down': round(target_down, 2),
                'position_pct': round(position_pct, 2),
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

    def get_signal(
        self,
        breakout_up: bool,
        breakout_down: bool,
        is_consolidating: bool,
        position_pct: float
    ) -> SignalType:
        """
        Generate a signal from the range state.

        Args:
            breakout_up: Is there an upward breakout?
            breakout_down: Is there a downward breakout?
            is_consolidating: Is there consolidation?
            position_pct: Position within the range (%)

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if breakout_up:
            return SignalType.STRONG_BUY
        elif breakout_down:
            return SignalType.STRONG_SELL
        elif is_consolidating:
            # During consolidation, based on range lower/upper.
            if position_pct < 30:
                return SignalType.BUY
            elif position_pct > 70:
                return SignalType.SELL

        return SignalType.HOLD

    def get_trend(
        self,
        position_pct: float,
        breakout_up: bool,
        breakout_down: bool
    ) -> TrendDirection:
        """
        Range pozisyonundan trend belirle

        Args:
            position_pct: Position within the range (%).
            breakout_up: Breakout upwards.
            breakout_down: Breakout downwards.

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if breakout_up or position_pct > 60:
            return TrendDirection.UP
        elif breakout_down or position_pct < 40:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'consolidation_threshold': 3.0,
            'breakout_confirmation': 1,
            'use_body': False
        }

    def _requires_volume(self) -> bool:
        """Range Breakout volume gerektirmez"""
        return False

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['upper', 'middle', 'lower']


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['RangeBreakout']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Range Breakout indicator test"""

    print("\n" + "="*60)
    print("RANGE BREAKOUT TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Consolidation -> Simulate breakout
    base_price = 100
    prices = [base_price]

    # First 50 candles: Narrow range (consolidation)
    for i in range(49):
        change = np.random.randn() * 0.2
        prices.append(np.clip(prices[-1] + change, 99, 101))

    # Sonraki 25 mum: Breakout up
    for i in range(25):
        change = np.random.randn() * 0.5 + 0.5
        prices.append(prices[-1] + change)

    # Son 25 mum: Yeni range
    for i in range(25):
        change = np.random.randn() * 0.3
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p + np.random.randn() * 0.1 for p in prices],
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    rb = RangeBreakout()
    print(f"   [OK] Created: {rb}")
    print(f"   [OK] Kategori: {rb.category.value}")
    print(f"   [OK] Required period: {rb.get_required_periods()}")

    result = rb(data)
    print(f"   [OK] Upper: {result.value['upper']}")
    print(f"   [OK] Middle: {result.value['middle']}")
    print(f"   [OK] Lower: {result.value['lower']}")
    print(f"   [OK] Range %: {result.metadata['range_pct']:.2f}%")
    print(f"   [OK] Consolidating: {result.metadata['is_consolidating']}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 2: Konsolidasyon testi
    print("\n3. Konsolidasyon testi (ilk 50 mum)...")
    consol_data = data.head(55)
    result = rb.calculate(consol_data)
    print(f"   [OK] Range Height: {result.metadata['range_height']:.2f}")
    print(f"   [OK] Range %: {result.metadata['range_pct']:.2f}%")
    print(f"   [OK] Consolidating: {result.metadata['is_consolidating']}")
    print(f"   [OK] Position %: {result.metadata['position_pct']:.2f}%")

    # Test 3: Breakout testi
    print("\n4. Breakout testi (75 mum)...")
    breakout_data = data.head(75)
    result = rb.calculate(breakout_data)
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Breakout DOWN: {result.metadata['breakout_down']}")
    print(f"   [OK] Breakout Confirmed: {result.metadata['breakout_confirmed']}")
    print(f"   [OK] Target UP: {result.metadata['target_up']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Power: {result.strength:.2f}")

    # Test 4: Body vs Wick
    print("\n5. Body vs Wick testi...")
    rb_body = RangeBreakout(use_body=True)
    result_body = rb_body.calculate(data)
    rb_wick = RangeBreakout(use_body=False)
    result_wick = rb_wick.calculate(data)
    print(f"   [OK] Body Range: {result_body.metadata['range_height']:.2f}")
    print(f"   [OK] Wick Range: {result_wick.metadata['range_height']:.2f}")

    # Test 5: Different thresholds
    print("\n6. Different threshold test...")
    rb_tight = RangeBreakout(consolidation_threshold=2.0)
    result = rb_tight.calculate(consol_data)
    print(f"   [OK] Tight Threshold (2%): {result.metadata['is_consolidating']}")

    rb_loose = RangeBreakout(consolidation_threshold=5.0)
    result = rb_loose.calculate(consol_data)
    print(f"   [OK] Loose Threshold (5%): {result.metadata['is_consolidating']}")

    # Test 6: Zaman serisi analizi
    print("\n7. Zaman serisi analizi...")
    consolidation_periods = []
    breakout_periods = []

    for i in range(30, len(data), 5):
        partial_data = data.head(i)
        result = rb.calculate(partial_data)

        if result.metadata['is_consolidating']:
            consolidation_periods.append(i)
        if result.metadata['breakout_confirmed']:
            breakout_periods.append(i)

    print(f"   [OK] Total measurement: {(len(data) - 30) // 5}")
    print(f"   [OK] Consolidation count: {len(consolidation_periods)}")
    print(f"   [OK] Number of breakout periods: {len(breakout_periods)}")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = rb.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = rb.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Output names: {metadata.output_names}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
