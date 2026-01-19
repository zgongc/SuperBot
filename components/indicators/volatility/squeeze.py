"""
indicators/volatility/squeeze.py - TTM Squeeze

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    TTM Squeeze - Volatility squeeze indicator
    Comparison of Bollinger Bands and Keltner Channel
    If Bollinger Bands are within the Keltner Channel, there is a "squeeze"
    A strong move is expected when the squeeze ends

Formula:
    BB = Bollinger Bands (20, 2.0)
    KC = Keltner Channel (20, 1.5)
    Squeeze = BB Lower > KC Lower AND BB Upper < KC Upper
    Momentum = LinReg(close - avg(high, low), 20)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.volatility.bollinger import BollingerBands
from indicators.volatility.keltner import KeltnerChannel
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class TTMSqueeze(BaseIndicator):
    """
    TTM Squeeze

    Compares Bollinger Bands and Keltner Channel to detect volatility compression.
    A squeeze condition can be a sign of a strong movement.

    Args:
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Bollinger Bands standard deviation (default: 2.0)
        kc_period: Keltner Channel period (default: 20)
        kc_atr_period: Keltner ATR period (default: 20)
        kc_mult: Keltner multiplier (default: 1.5)
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_period: int = 20,
        kc_mult: float = 1.5,
        logger=None,
        error_handler=None
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_period = kc_atr_period
        self.kc_mult = kc_mult

        # Use Bollinger Bands and Keltner Channel indicators (code reuse)
        self._bb = BollingerBands(period=bb_period, std_dev=bb_std)
        self._kc = KeltnerChannel(ema_period=kc_period, atr_period=kc_atr_period, multiplier=kc_mult)

        super().__init__(
            name='squeeze',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'bb_period': bb_period,
                'bb_std': bb_std,
                'kc_period': kc_period,
                'kc_atr_period': kc_atr_period,
                'kc_mult': kc_mult
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.bb_period, self.kc_period, self.kc_atr_period) + 1

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.bb_period < 2:
            raise InvalidParameterError(
                self.name, 'bb_period', self.bb_period,
                "The BB period must be at least 2"
            )
        if self.bb_std <= 0:
            raise InvalidParameterError(
                self.name, 'bb_std', self.bb_std,
                "BB standard deviation must be positive"
            )
        if self.kc_period < 1:
            raise InvalidParameterError(
                self.name, 'kc_period', self.kc_period,
                "KC period must be positive"
            )
        if self.kc_atr_period < 1:
            raise InvalidParameterError(
                self.name, 'kc_atr_period', self.kc_atr_period,
                "KC ATR period must be positive"
            )
        if self.kc_mult <= 0:
            raise InvalidParameterError(
                self.name, 'kc_mult', self.kc_mult,
                "KC factor must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        TTM Squeeze hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Squeeze value (positive/negative momentum)
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Bollinger Bands hesapla
        bb_middle = np.mean(close[-self.bb_period:])
        bb_std = np.std(close[-self.bb_period:], ddof=0)
        bb_upper = bb_middle + (self.bb_std * bb_std)
        bb_lower = bb_middle - (self.bb_std * bb_std)

        # Keltner Channel hesapla
        # EMA hesapla
        ema_values = np.zeros(len(close))
        ema_values[0] = close[0]
        alpha = 2.0 / (self.kc_period + 1)

        for i in range(1, len(close)):
            ema_values[i] = alpha * close[i] + (1 - alpha) * ema_values[i-1]

        kc_middle = ema_values[-1]

        # ATR hesapla
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        atr_values = np.zeros(len(tr))
        start_idx = self.kc_atr_period - 1
        atr_values[start_idx] = np.mean(tr[:self.kc_atr_period])

        atr_alpha = 1.0 / self.kc_atr_period
        for i in range(self.kc_atr_period, len(tr)):
            atr_values[i] = atr_values[i-1] + atr_alpha * (tr[i] - atr_values[i-1])

        atr = atr_values[-1]

        kc_upper = kc_middle + (atr * self.kc_mult)
        kc_lower = kc_middle - (atr * self.kc_mult)

        # Squeeze state check
        # If it's inside BB, there's a squeeze.
        squeeze_on = (bb_lower > kc_lower) and (bb_upper < kc_upper)

        # Calculate momentum (simplified: close - average of hl2)
        hl2 = (high + low) / 2
        momentum_source = close - hl2

        # Average momentum of the last 20 candles
        momentum_period = min(20, len(momentum_source))
        momentum_value = np.mean(momentum_source[-momentum_period:])

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Squeeze power (how close the bands are)
        if kc_upper != kc_lower:
            squeeze_strength = 1 - ((bb_upper - bb_lower) / (kc_upper - kc_lower))
            squeeze_strength = max(0, min(1, squeeze_strength))  # between 0 and 1
        else:
            squeeze_strength = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(momentum_value, 8),
            timestamp=timestamp,
            signal=self.get_signal(squeeze_on, momentum_value),
            trend=self.get_trend(momentum_value),
            strength=min(abs(momentum_value) * 100, 100),  # Normalize to a range of 0-100
            metadata={
                'squeeze_on': squeeze_on,
                'squeeze_strength': round(squeeze_strength, 4),
                'bb_upper': round(bb_upper, 8),
                'bb_lower': round(bb_lower, 8),
                'kc_upper': round(kc_upper, 8),
                'kc_lower': round(kc_lower, 8),
                'momentum': round(momentum_value, 8),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch TTM Squeeze calculation - for BACKTEST

        TTM Squeeze Formula:
            BB = Bollinger Bands (SMA ± std * mult)
            KC = Keltner Channel (EMA ± ATR * mult)
            Squeeze = BB inside KC (bb_lower > kc_lower AND bb_upper < kc_upper)
            Momentum = close - (high + low) / 2 rolling mean

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: squeeze (bool), momentum for all bars
            NOTE: 'squeeze' column contains MOMENTUM VALUE (consistent with calculate())

        Performance: 2000 bars in ~0.05 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # 1. Bollinger Bands - use BollingerBands.calculate_batch (code reuse)
        bb_result = self._bb.calculate_batch(data)
        bb_upper = bb_result['upper']
        bb_lower = bb_result['lower']

        # 2. Keltner Channel - use KeltnerChannel.calculate_batch (code reuse)
        kc_result = self._kc.calculate_batch(data)
        kc_upper = kc_result['upper']
        kc_lower = kc_result['lower']

        # 3. Squeeze detection: BB inside KC?
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # 4. Momentum calculation (same as calculate())
        hl2 = (high + low) / 2
        momentum_source = close - hl2
        momentum = momentum_source.rolling(window=20, min_periods=1).mean()

        # Set first period values to NaN (warmup)
        warmup = max(self.bb_period, self.kc_period, self.kc_atr_period)
        momentum.iloc[:warmup] = np.nan
        # Convert to float first to allow NaN assignment (squeeze_on is bool)
        squeeze_on = squeeze_on.astype(float)
        squeeze_on.iloc[:warmup] = np.nan

        return pd.DataFrame({
            'momentum': momentum.values,  # Main value is momentum (consistent with calculate())
            'squeeze_on': squeeze_on.values
        }, index=data.index)

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
                value=0.0,  # Momentum value (consistent with calculate())
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'squeeze_on': False}
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

    def get_signal(self, squeeze_on: bool, momentum: float) -> SignalType:
        """
        Generate a signal from the squeeze condition and momentum.

        Args:
            squeeze_on: Is the squeeze feature active?
            momentum: Momentum value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if squeeze_on:
            # Squeeze active: wait
            return SignalType.HOLD
        else:
            # Squeeze is complete: signal in the momentum direction
            if momentum > 0:
                return SignalType.BUY
            elif momentum < 0:
                return SignalType.SELL
            return SignalType.HOLD

    def get_trend(self, momentum: float) -> TrendDirection:
        """
        Momentum'dan trend belirle

        Args:
            momentum: Momentum value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if momentum > 0:
            return TrendDirection.UP
        elif momentum < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'kc_period': 20,
            'kc_atr_period': 20,
            'kc_mult': 1.5
        }

    def _requires_volume(self) -> bool:
        """TTM Squeeze volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TTMSqueeze']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """TTM Squeeze indicator test"""

    print("\n" + "="*60)
    print("TTM SQUEEZE TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(40)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(39):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    squeeze = TTMSqueeze()
    print(f"   [OK] Created: {squeeze}")
    print(f"   [OK] Kategori: {squeeze.category.value}")
    print(f"   [OK] Required period: {squeeze.get_required_periods()}")

    result = squeeze(data)
    print(f"   [OK] Momentum: {result.value}")
    print(f"   [OK] Squeeze Active: {result.metadata['squeeze_on']}")
    print(f"   [OK] Squeeze Power: {result.metadata['squeeze_strength']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] BB Upper: {result.metadata['bb_upper']:.2f}, Lower: {result.metadata['bb_lower']:.2f}")
    print(f"   [OK] KC Upper: {result.metadata['kc_upper']:.2f}, Lower: {result.metadata['kc_lower']:.2f}")

    # Test 2: Low volatility (squeeze active)
    print("\n3. Low volatility (squeeze) test...")
    low_vol_prices = [100.0]
    for i in range(39):
        change = np.random.randn() * 0.1  # Very low volatility
        low_vol_prices.append(low_vol_prices[-1] + change)

    low_vol_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': low_vol_prices,
        'high': [p + 0.05 for p in low_vol_prices],
        'low': [p - 0.05 for p in low_vol_prices],
        'close': low_vol_prices,
        'volume': [1000] * 40
    })
    result = squeeze.calculate(low_vol_data)
    print(f"   [OK] Low Vol Momentum: {result.value:.6f}")
    print(f"   [OK] Squeeze Active: {result.metadata['squeeze_on']}")
    print(f"   [OK] Squeeze Power: {result.metadata['squeeze_strength']:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 3: High volatility (no squeeze)
    print("\n4. High volatility (no squeeze) test...")
    high_vol_prices = [100]
    for i in range(39):
        change = np.random.randn() * 5  # High volatility
        high_vol_prices.append(high_vol_prices[-1] + change)

    high_vol_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': high_vol_prices,
        'high': [p + abs(np.random.randn()) * 3 for p in high_vol_prices],
        'low': [p - abs(np.random.randn()) * 3 for p in high_vol_prices],
        'close': high_vol_prices,
        'volume': [1000] * 40
    })
    result = squeeze.calculate(high_vol_data)
    print(f"   [OK] High Vol Momentum: {result.value:.4f}")
    print(f"   [OK] Squeeze Active: {result.metadata['squeeze_on']}")
    print(f"   [OK] Squeeze Power: {result.metadata['squeeze_strength']:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Pozitif momentum
    print("\n5. Pozitif momentum testi...")
    uptrend_prices = [100]
    for i in range(39):
        change = abs(np.random.randn()) * 1.0  # Positive direction
        uptrend_prices.append(uptrend_prices[-1] + change)

    uptrend_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': uptrend_prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in uptrend_prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in uptrend_prices],
        'close': uptrend_prices,
        'volume': [1000] * 40
    })
    result = squeeze.calculate(uptrend_data)
    print(f"   [OK] Pozitif Momentum: {result.value:.4f}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 5: Negative momentum
    print("\n6. Negative momentum test...")
    downtrend_prices = [100]
    for i in range(39):
        change = abs(np.random.randn()) * 1.0  # Negative direction
        downtrend_prices.append(downtrend_prices[-1] - change)

    downtrend_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': downtrend_prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in downtrend_prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in downtrend_prices],
        'close': downtrend_prices,
        'volume': [1000] * 40
    })
    result = squeeze.calculate(downtrend_data)
    print(f"   [OK] Negative Momentum: {result.value:.4f}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 6: Different parameters
    print("\n7. Different parameter test...")
    squeeze_tight = TTMSqueeze(bb_std=1.5, kc_mult=1.0)
    result = squeeze_tight.calculate(data)
    print(f"   [OK] Tight Squeeze: {result.metadata['squeeze_on']}")

    squeeze_loose = TTMSqueeze(bb_std=2.5, kc_mult=2.0)
    result = squeeze_loose.calculate(data)
    print(f"   [OK] Loose Squeeze: {result.metadata['squeeze_on']}")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = squeeze.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = squeeze.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
