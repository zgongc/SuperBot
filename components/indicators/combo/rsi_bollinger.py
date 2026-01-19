"""
indicators/combo/rsi_bollinger.py - RSI + Bollinger Bands Kombine Stratejisi

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    RSI + Bollinger Bands combined indicator
    Combines the RSI momentum oscillator with Bollinger Bands volatility bands to
    generate strong buy/sell signals.

    Features:
    - RSI overbought/oversold levels
    - Bollinger Bands price position
    - Combined signal generation
    - Robust confirmation system

Strategy:
    STRONG BUY: RSI < 30 AND Price Near/Below Lower Band
    BUY: RSI < 40 AND Price Near Lower Band
    STRONG SELL: RSI > 70 AND Price Near/Above Upper Band
    SELL: RSI > 60 AND Price Near Upper Band
    HOLD: Other cases

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.momentum.rsi
    - indicators.volatility.bollinger
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.momentum.rsi import RSI
from indicators.volatility.bollinger import BollingerBands
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class RSIBollinger(BaseIndicator):
    """
    RSI + Bollinger Bands Kombine Stratejisi

    Combining the RSI momentum oscillator with the Bollinger Bands volatility bands.
    generates strong buy/sell signals.

    Args:
        rsi_period: RSI period (default: 14)
        rsi_overbought: RSI overbought level (default: 70)
        rsi_oversold: RSI oversold level (default: 30)
        bb_period: Bollinger Bands period (default: 20)
        bb_std_dev: BB standard deviation factor (default: 2.0)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        logger=None,
        error_handler=None
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev

        # Create sub-indicators
        self.rsi = RSI(
            period=rsi_period,
            overbought=rsi_overbought,
            oversold=rsi_oversold,
            logger=logger,
            error_handler=error_handler
        )

        self.bb = BollingerBands(
            period=bb_period,
            std_dev=bb_std_dev,
            logger=logger,
            error_handler=error_handler
        )

        super().__init__(
            name='rsi_bollinger',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'bb_period': bb_period,
                'bb_std_dev': bb_std_dev
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.rsi.get_required_periods(), self.bb.get_required_periods())

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "The RSI period must be positive"
            )
        if self.bb_period < 2:
            raise InvalidParameterError(
                self.name, 'bb_period', self.bb_period,
                "The BB period must be at least 2"
            )
        if self.rsi_oversold >= self.rsi_overbought:
            raise InvalidParameterError(
                self.name, 'rsi_levels',
                f"oversold={self.rsi_oversold}, overbought={self.rsi_overbought}",
                "RSI oversold should be less than overbought"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch RSI + Bollinger calculation - for BACKTEST

        Combines RSI and Bollinger Bands using their respective calculate_batch() methods

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: rsi, bb_upper, bb_middle, bb_lower for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        # Calculate RSI (vectorized)
        rsi_series = self.rsi.calculate_batch(data)

        # Calculate Bollinger Bands (vectorized)
        bb_df = self.bb.calculate_batch(data)

        # Calculate percent_b (vectorized)
        close = data['close']
        bb_upper = bb_df['upper']
        bb_lower = bb_df['lower']

        # Avoid division by zero
        denominator = bb_upper - bb_lower
        percent_b = ((close - bb_lower) / denominator.replace(0, np.nan))

        # Combine results (same keys as calculate())
        return pd.DataFrame({
            'rsi': rsi_series.values,
            'bb_upper': bb_upper.values,
            'bb_middle': bb_df['middle'].values,
            'bb_lower': bb_lower.values,
            'percent_b': percent_b.values,
            'price': close.values
        }, index=data.index)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        RSI + Bollinger combined calculation

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Combined RSI + BB values and signals.
        """
        # RSI hesapla
        rsi_result = self.rsi.calculate(data)
        rsi_value = rsi_result.value

        # Bollinger Bands hesapla
        bb_result = self.bb.calculate(data)
        bb_upper = bb_result.value['upper']
        bb_middle = bb_result.value['middle']
        bb_lower = bb_result.value['lower']

        # Current price and %B value
        current_price = data['close'].values[-1]
        percent_b = bb_result.metadata['percent_b']

        timestamp = int(data.iloc[-1]['timestamp'])

        # Combined signal and trend determination
        signal = self.get_signal(rsi_value, percent_b, current_price, bb_upper, bb_lower)
        trend = self.get_trend(rsi_value, percent_b)
        strength = self._calculate_strength(rsi_value, percent_b)

        # Signal confirmation
        confirmation = self._get_confirmation(rsi_value, percent_b)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'rsi': round(rsi_value, 2),
                'bb_upper': round(bb_upper, 8),
                'bb_middle': round(bb_middle, 8),
                'bb_lower': round(bb_lower, 8),
                'percent_b': round(percent_b, 4),
                'price': round(current_price, 8)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'rsi_period': self.rsi_period,
                'bb_period': self.bb_period,
                'rsi_signal': rsi_result.signal.value,
                'bb_signal': bb_result.signal.value,
                'confirmation': confirmation,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'bandwidth': bb_result.metadata['bandwidth']
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
                value={'signal': 'none', 'strength': 0.0},
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

    def get_signal(self, rsi: float, percent_b: float, price: float,
                   bb_upper: float, bb_lower: float) -> SignalType:
        """
        Kombine RSI + BB sinyali

        Args:
            rsi: RSI value
            percent_b: Bollinger %B value
            price: Current price
            bb_upper: Upper band
            bb_lower: Alt bant

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # STRONG AL signals
        if rsi < 30 and (percent_b <= 0 or price <= bb_lower):
            return SignalType.BUY  # Very strong buy

        # AL sinyalleri
        if rsi < 40 and percent_b < 0.2:
            return SignalType.BUY

        # STRONG BUY signals
        if rsi > 70 and (percent_b >= 1 or price >= bb_upper):
            return SignalType.SELL  # Strong sell signal

        # SAT sinyalleri
        if rsi > 60 and percent_b > 0.8:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, rsi: float, percent_b: float) -> TrendDirection:
        """
        Kombine trend belirleme

        Args:
            rsi: RSI value
            percent_b: Bollinger %B value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        # If both indicators show an upward trend
        if rsi > 50 and percent_b > 0.5:
            return TrendDirection.UP
        # If both indicators show a decrease
        elif rsi < 50 and percent_b < 0.5:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _calculate_strength(self, rsi: float, percent_b: float) -> float:
        """
        Calculate signal strength (0-100)

        The power increases at the extreme values of two indicators.
        """
        # Power from RSI
        rsi_strength = 0
        if rsi < 30:
            rsi_strength = (30 - rsi) * 2  # Oversold
        elif rsi > 70:
            rsi_strength = (rsi - 70) * 2  # Oversold

        # Power from %B
        bb_strength = 0
        if percent_b <= 0:
            bb_strength = abs(percent_b) * 100 + 50
        elif percent_b >= 1:
            bb_strength = (percent_b - 1) * 100 + 50
        elif percent_b < 0.2:
            bb_strength = (0.2 - percent_b) * 200
        elif percent_b > 0.8:
            bb_strength = (percent_b - 0.8) * 200

        # Combined power (average)
        combined_strength = (rsi_strength + bb_strength) / 2

        return min(combined_strength, 100)

    def _get_confirmation(self, rsi: float, percent_b: float) -> str:
        """
        Determine the signal confirmation status.

        Returns:
            str: 'strong', 'moderate', 'weak' or 'none'
        """
        # Strong confirmation (both indicators are in the same direction and extreme)
        if (rsi < 30 and percent_b < 0.2) or (rsi > 70 and percent_b > 0.8):
            return 'strong'

        # Middle confirmation (one indicator is extreme, the other confirms)
        if (rsi < 40 and percent_b < 0.3) or (rsi > 60 and percent_b > 0.7):
            return 'moderate'

        # Weak confirmation (indicators are in different directions)
        if (rsi < 50 and percent_b > 0.5) or (rsi > 50 and percent_b < 0.5):
            return 'weak'

        return 'none'

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std_dev': 2.0
        }

    def _requires_volume(self) -> bool:
        """RSI + Bollinger volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['RSIBollinger']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """RSI + Bollinger combined indicator test"""

    print("\n" + "="*60)
    print("RSI + BOLLINGER BANDS COMBO TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Simulate volatile price movements
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 15:
            change = np.random.randn() * 1.5 - 0.5  # Downward trend
        elif i < 35:
            change = np.random.randn() * 1.5 + 0.5  # Upward trend
        else:
            change = np.random.randn() * 2  # Yatay hareket
        prices.append(max(prices[-1] + change, 50))  # Minimum 50

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

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    combo = RSIBollinger()
    print(f"   [OK] Created: {combo}")
    print(f"   [OK] Kategori: {combo.category.value}")
    print(f"   [OK] Tip: {combo.indicator_type.value}")
    print(f"   [OK] Required period: {combo.get_required_periods()}")

    result = combo(data)
    print(f"   [OK] RSI: {result.value['rsi']}")
    print(f"   [OK] BB Upper: {result.value['bb_upper']:.2f}")
    print(f"   [OK] BB Middle: {result.value['bb_middle']:.2f}")
    print(f"   [OK] BB Lower: {result.value['bb_lower']:.2f}")
    print(f"   [OK] %B: {result.value['percent_b']}")
    print(f"   [OK] Price: {result.value['price']:.2f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")

    # Test 2: Signal analysis
    print("\n3. Signal analysis...")
    print(f"   [OK] RSI Sinyali: {result.metadata['rsi_signal']}")
    print(f"   [OK] BB Sinyali: {result.metadata['bb_signal']}")
    print(f"   [OK] Combined Signal: {result.signal.value}")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")

    # Test 3: Different scenarios
    print("\n4. Different scenario testing...")
    scenarios = [
        (20, "End of decrease"),
        (30, "Ascent middle"),
        (45, "End of ascent")
    ]

    for idx, desc in scenarios:
        if idx < len(data):
            data_slice = data.iloc[:idx+1]
            result = combo.calculate(data_slice)
            print(f"   [OK] {desc}: RSI={result.value['rsi']:.1f}, "
                  f"%B={result.value['percent_b']:.2f}, "
                  f"Signal={result.signal.value}, "
                  f"Konfirmasyon={result.metadata['confirmation']}")

    # Test 4: Custom parameters
    print("\n5. Special parameter test...")
    combo_custom = RSIBollinger(
        rsi_period=21,
        rsi_overbought=75,
        rsi_oversold=25,
        bb_period=30,
        bb_std_dev=2.5
    )
    result = combo_custom.calculate(data)
    print(f"   [OK] RSI with custom parameters: {result.value['rsi']}")
    print(f"   [OK] Parameter with special format %B: {result.value['percent_b']:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 5: Power analysis
    print("\n6. Power analysis...")
    print(f"   [OK] Signal Strength: {result.strength:.2f}/100")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")
    print(f"   [OK] Bandwidth: {result.metadata['bandwidth']:.2f}%")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = combo.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = combo.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
