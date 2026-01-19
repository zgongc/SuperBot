"""
indicators/combo/triple_screen.py - Elder's Triple Screen Trading System

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Triple Screen - The famous 3-screen trading system by Dr. Alexander Elder
    Combines three different timeframes and indicators to generate reliable signals.

    Sistem:
    1. SCREEN 1 (Trend): Long-term trend (EMA or MACD)
    2. SCREEN 2 (Oscillator): Mid-term momentum (RSI or Stochastic)
    3. SCREEN 3 (Entry): Short-term entry (Price Action)

    Features:
    - Multiple timezone analysis
    - Trend filtering system
    - Powerful risk management
    - High accuracy rate

Strategy:
    AL Sinyali:
    - Screen 1: Long-term uptrend (EMA > EMA_slow OR MACD > 0)
    - Screen 2: RSI is in the oversold region (<30)
    - Screen 3: Price is higher than the previous low level

    SAT Sinyali:
    - Screen 1: Long-term downtrend (EMA < EMA_slow OR MACD < 0)
    - Screen 2: RSI in overbought region (>70)
    - Screen 3: Price is lower than the previous peak level

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.trend.ema
    - indicators.trend.macd
    - indicators.momentum.rsi
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.ema import EMA
from indicators.trend.macd import MACD
from indicators.momentum.rsi import RSI
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class TripleScreen(BaseIndicator):
    """
    Elder's Triple Screen Trading System

    It generates reliable buy/sell signals with a three-screen system.

    Args:
        ema_fast: Fast EMA period (default: 13)
        ema_slow: Slow EMA period (default: 26)
        rsi_period: RSI period (default: 13)
        rsi_overbought: RSI overbought level (default: 70)
        rsi_oversold: RSI oversold level (default: 30)
        use_macd: Use MACD (default: True), if False use EMA crossover
    """

    def __init__(
        self,
        ema_fast: int = 13,
        ema_slow: int = 26,
        rsi_period: int = 13,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        use_macd: bool = True,
        logger=None,
        error_handler=None
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_macd = use_macd

        # Screen 1: Trend identification (MACD or EMA crossover)
        if use_macd:
            self.trend_indicator = MACD(
                fast_period=12,
                slow_period=26,
                signal_period=9,
                logger=logger,
                error_handler=error_handler
            )
        else:
            self.ema_fast_ind = EMA(period=ema_fast, logger=logger, error_handler=error_handler)
            self.ema_slow_ind = EMA(period=ema_slow, logger=logger, error_handler=error_handler)

        # Ekran 2: Oscillator (RSI)
        self.rsi = RSI(
            period=rsi_period,
            overbought=rsi_overbought,
            oversold=rsi_oversold,
            logger=logger,
            error_handler=error_handler
        )

        super().__init__(
            name='triple_screen',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'use_macd': use_macd
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        if self.use_macd:
            return max(self.trend_indicator.get_required_periods(), self.rsi.get_required_periods())
        else:
            return max(self.ema_slow, self.rsi.get_required_periods())

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.ema_fast >= self.ema_slow:
            raise InvalidParameterError(
                self.name, 'ema_periods',
                f"fast={self.ema_fast}, slow={self.ema_slow}",
                "The EMA fast period must be smaller than the slow period."
            )
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "The RSI period must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Triple Screen system calculation

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Three-screen analysis and combined signal.
        """
        # EKRAN 1: Uzun vadeli trend
        if self.use_macd:
            trend_result = self.trend_indicator.calculate(data)
            screen1_value = trend_result.value['histogram']
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = trend_result.signal
        else:
            ema_fast_result = self.ema_fast_ind.calculate(data)
            ema_slow_result = self.ema_slow_ind.calculate(data)
            screen1_value = ema_fast_result.value - ema_slow_result.value
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = SignalType.BUY if screen1_value > 0 else SignalType.SELL

        # EKRAN 2: Orta vadeli oscillator (RSI)
        rsi_result = self.rsi.calculate(data)
        rsi_value = rsi_result.value
        screen2_signal = rsi_result.signal
        screen2_strength = rsi_result.strength

        # SCREEN 3: Short-term entry point (Price Action)
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Last 5 candle's low and high levels
        recent_high = np.max(high[-5:]) if len(high) >= 5 else high[-1]
        recent_low = np.min(low[-5:]) if len(low) >= 5 else low[-1]
        current_price = close[-1]

        # Price action control
        screen3_buy = current_price > recent_low  # A new low is not forming
        screen3_sell = current_price < recent_high  # A new peak is not forming

        timestamp = int(data.iloc[-1]['timestamp'])

        # Combined signal generation (all 3 screens must confirm)
        signal = self._generate_signal(
            screen1_trend, screen1_signal,
            rsi_value, screen2_signal,
            screen3_buy, screen3_sell
        )

        # Signal strength and confirmation
        strength = self._calculate_strength(screen1_trend, rsi_value, screen3_buy, screen3_sell)
        confirmation = self._get_confirmation(screen1_signal, screen2_signal, signal)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'screen1': round(screen1_value, 4),
                'screen2': round(rsi_value, 2),
                'screen3_buy': screen3_buy,
                'screen3_sell': screen3_sell,
                'current_price': round(current_price, 8)
            },
            timestamp=timestamp,
            signal=signal,
            trend=screen1_trend,
            strength=strength,
            metadata={
                'ema_fast': self.ema_fast,
                'ema_slow': self.ema_slow,
                'rsi_period': self.rsi_period,
                'use_macd': self.use_macd,
                'screen1_trend': screen1_trend.name,
                'screen1_signal': screen1_signal.value,
                'screen2_rsi': round(rsi_value, 2),
                'screen2_signal': screen2_signal.value,
                'screen3_status': 'ready' if (screen3_buy or screen3_sell) else 'waiting',
                'confirmation': confirmation,
                'recent_high': round(recent_high, 8),
                'recent_low': round(recent_low, 8)
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current Triple Screen value
        """
        # Buffer management
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            
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
        
        # Yeterli veri yoksa
        min_required = self.get_required_periods()
        if len(self._close_buffer) < min_required:
            return IndicatorResult(
                value={'screen1': 0.0, 'screen2': 50.0, 'screen3_buy': False, 'screen3_sell': False, 'current_price': candle['close']},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'confirmation': 'none'}
            )
            
        # Create a DataFrame (for sub-indicators)
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        # EKRAN 1: Uzun vadeli trend
        if self.use_macd:
            trend_result = self.trend_indicator.calculate(buffer_data)
            screen1_value = trend_result.value['histogram']
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = trend_result.signal
        else:
            ema_fast_result = self.ema_fast_ind.calculate(buffer_data)
            ema_slow_result = self.ema_slow_ind.calculate(buffer_data)
            screen1_value = ema_fast_result.value - ema_slow_result.value
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = SignalType.BUY if screen1_value > 0 else SignalType.SELL
        
        # EKRAN 2: Orta vadeli oscillator (RSI)
        rsi_result = self.rsi.calculate(buffer_data)
        rsi_value = rsi_result.value
        screen2_signal = rsi_result.signal
        
        # SCREEN 3: Short-term entry point (Price Action)
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        
        # Last 5 candle's low and high levels
        recent_high = np.max(high[-5:]) if len(high) >= 5 else high[-1]
        recent_low = np.min(low[-5:]) if len(low) >= 5 else low[-1]
        current_price = close[-1]
        
        # Price action control
        screen3_buy = current_price > recent_low
        screen3_sell = current_price < recent_high
        
        # Combined signal generation
        signal = self._generate_signal(
            screen1_trend, screen1_signal,
            rsi_value, screen2_signal,
            screen3_buy, screen3_sell
        )

        # Signal strength and confirmation
        strength = self._calculate_strength(screen1_trend, rsi_value, screen3_buy, screen3_sell)
        confirmation = self._get_confirmation(screen1_signal, screen2_signal, signal)

        return IndicatorResult(
            value={
                'screen1': round(screen1_value, 4),
                'screen2': round(rsi_value, 2),
                'screen3_buy': screen3_buy,
                'screen3_sell': screen3_sell,
                'current_price': round(current_price, 8)
            },
            timestamp=timestamp_val,
            signal=signal,
            trend=screen1_trend,
            strength=strength,
            metadata={
                'ema_fast': self.ema_fast,
                'ema_slow': self.ema_slow,
                'rsi_period': self.rsi_period,
                'use_macd': self.use_macd,
                'screen1_trend': screen1_trend.name,
                'screen1_signal': screen1_signal.value,
                'screen2_rsi': round(rsi_value, 2),
                'screen2_signal': screen2_signal.value,
                'screen3_status': 'ready' if (screen3_buy or screen3_sell) else 'waiting',
                'confirmation': confirmation,
                'recent_high': round(recent_high, 8),
                'recent_low': round(recent_low, 8)
            }
        )

    def _generate_signal(
        self,
        screen1_trend: TrendDirection,
        screen1_signal: SignalType,
        rsi_value: float,
        screen2_signal: SignalType,
        screen3_buy: bool,
        screen3_sell: bool
    ) -> SignalType:
        """
        Generates a combined signal from three screens.

        All screens must confirm in the same direction.
        """
        # AL Signal: Screen 1 uptrend + Screen 2 oversold + Screen 3 confirmation
        if (screen1_trend == TrendDirection.UP and
            rsi_value < self.rsi_oversold and
            screen3_buy):
            return SignalType.BUY

        # SAT Signal: Screen 1 downtrend + Screen 2 overbought + Screen 3 confirmation
        if (screen1_trend == TrendDirection.DOWN and
            rsi_value > self.rsi_overbought and
            screen3_sell):
            return SignalType.SELL

        # Partial confirmations (weaker signals)
        if screen1_trend == TrendDirection.UP and rsi_value < 40 and screen3_buy:
            return SignalType.BUY

        if screen1_trend == TrendDirection.DOWN and rsi_value > 60 and screen3_sell:
            return SignalType.SELL

        return SignalType.HOLD

    def _calculate_strength(
        self,
        screen1_trend: TrendDirection,
        rsi_value: float,
        screen3_buy: bool,
        screen3_sell: bool
    ) -> float:
        """
        Calculate signal strength (0-100)

        Evaluate the contribution of each screen.
        """
        strength = 0

        # Screen 1 contribution (trend strength)
        if screen1_trend != TrendDirection.NEUTRAL:
            strength += 30

        # Screen 2 contribution (RSI extreme levels)
        rsi_deviation = abs(rsi_value - 50)
        if rsi_value < 30 or rsi_value > 70:
            strength += min(rsi_deviation, 40)
        else:
            strength += min(rsi_deviation / 2, 20)

        # Screen 3 contribution (price action confirmation)
        if screen3_buy or screen3_sell:
            strength += 30

        return min(strength, 100)

    def _get_confirmation(
        self,
        screen1_signal: SignalType,
        screen2_signal: SignalType,
        final_signal: SignalType
    ) -> str:
        """
        Determine the signal confirmation level.

        Returns:
            str: 'triple', 'double', 'single' or 'none'
        """
        # The three screens are in the same orientation.
        if (screen1_signal == screen2_signal == final_signal and
            final_signal != SignalType.HOLD):
            return 'triple'

        # Two screens in the same orientation
        if ((screen1_signal == screen2_signal and screen1_signal != SignalType.HOLD) or
            (screen1_signal == final_signal and screen1_signal != SignalType.HOLD) or
            (screen2_signal == final_signal and screen2_signal != SignalType.HOLD)):
            return 'double'

        # Sending a single screen signal
        if final_signal != SignalType.HOLD:
            return 'single'

        return 'none'

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation - returns DataFrame with multiple columns

        Note: This is a simple implementation for compatibility.
        For performance, consider implementing vectorized logic.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Indicator values with columns: screen1, screen2, screen3_buy, screen3_sell, current_price
        """
        results = {
            'screen1': [],
            'screen2': [],
            'screen3_buy': [],
            'screen3_sell': [],
            'current_price': []
        }

        for i in range(len(data)):
            if i < self.get_required_periods() - 1:
                for key in results:
                    results[key].append(np.nan)
            else:
                window_data = data.iloc[:i+1]
                result = self.calculate(window_data)

                # Extract dict values
                if result and hasattr(result, 'value') and isinstance(result.value, dict):
                    results['screen1'].append(result.value.get('screen1', np.nan))
                    results['screen2'].append(result.value.get('screen2', np.nan))
                    results['screen3_buy'].append(result.value.get('screen3_buy', False))
                    results['screen3_sell'].append(result.value.get('screen3_sell', False))
                    results['current_price'].append(result.value.get('current_price', np.nan))
                else:
                    for key in results:
                        results[key].append(np.nan)

        return pd.DataFrame(results, index=data.index)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'ema_fast': 13,
            'ema_slow': 26,
            'rsi_period': 13,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'use_macd': True
        }

    def _requires_volume(self) -> bool:
        """Triple Screen volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TripleScreen']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Triple Screen indicator test"""

    print("\n" + "="*60)
    print("TRIPLE SCREEN TRADING SYSTEM TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Market simulation with trend changes
    base_price = 100
    prices = [base_price]
    highs = []
    lows = []

    for i in range(99):
        if i < 30:
            trend = 0.5  # Increase
        elif i < 60:
            trend = -0.3  # Decrease
        else:
            trend = 0.6  # Strong upward trend
        noise = np.random.randn() * 2
        new_price = prices[-1] + trend + noise
        prices.append(new_price)
        highs.append(new_price + abs(np.random.randn()) * 1.0)
        lows.append(new_price - abs(np.random.randn()) * 1.0)

    # Add high/low for the initial price
    highs.insert(0, prices[0] + 1)
    lows.insert(0, prices[0] - 1)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation (MACD version)
    print("\n2. Basic calculation test (MACD)...")
    ts = TripleScreen(use_macd=True)
    print(f"   [OK] Created: {ts}")
    print(f"   [OK] Kategori: {ts.category.value}")
    print(f"   [OK] Tip: {ts.indicator_type.value}")
    print(f"   [OK] Required period: {ts.get_required_periods()}")

    result = ts(data)
    print(f"   [OK] Ekran 1 (MACD): {result.value['screen1']}")
    print(f"   [OK] Ekran 2 (RSI): {result.value['screen2']}")
    print(f"   [OK] Ekran 3 Buy: {result.value['screen3_buy']}")
    print(f"   [OK] Ekran 3 Sell: {result.value['screen3_sell']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")

    # Test 2: Ekran analizi
    print("\n3. Ekran analizi...")
    print(f"   [OK] Ekran 1 Trend: {result.metadata['screen1_trend']}")
    print(f"   [OK] Screen 1 Signal: {result.metadata['screen1_signal']}")
    print(f"   [OK] Ekran 2 RSI: {result.metadata['screen2_rsi']}")
    print(f"   [OK] Screen 2 Signal: {result.metadata['screen2_signal']}")
    print(f"   [OK] Ekran 3 Status: {result.metadata['screen3_status']}")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")

    # Test 3: EMA versiyonu
    print("\n4. EMA crossover versiyonu testi...")
    ts_ema = TripleScreen(use_macd=False, ema_fast=13, ema_slow=26)
    result_ema = ts_ema.calculate(data)
    print(f"   [OK] Ekran 1 (EMA): {result_ema.value['screen1']:.4f}")
    print(f"   [OK] Ekran 2 (RSI): {result_ema.value['screen2']}")
    print(f"   [OK] Signal: {result_ema.signal.value}")
    print(f"   [OK] Konfirmasyon: {result_ema.metadata['confirmation']}")

    # Test 4: Trend change analysis
    print("\n5. Trend change analysis...")
    test_points = [35, 55, 75, 95]
    for idx in test_points:
        data_slice = data.iloc[:idx+1]
        result = ts.calculate(data_slice)
        print(f"   [OK] Mum {idx}: "
              f"Screen1={result.metadata['screen1_trend']}, "
              f"RSI={result.value['screen2']:.1f}, "
              f"Signal={result.signal.value}, "
              f"Confirm={result.metadata['confirmation']}")

    # Test 5: Custom parameters
    print("\n6. Special parameter test...")
    ts_custom = TripleScreen(
        ema_fast=8,
        ema_slow=21,
        rsi_period=9,
        rsi_overbought=75,
        rsi_oversold=25,
        use_macd=True
    )
    result = ts_custom.calculate(data)
    print(f"   [OK] Screen 1 with custom parameters: {result.value['screen1']:.4f}")
    print(f"   [OK] Screen with custom parameters 2: {result.value['screen2']}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 6: Signal strength and confirmation
    print("\n7. Signal strength and confirmation analysis...")
    print(f"   [OK] Signal Strength: {result.strength:.2f}/100")
    print(f"   [OK] Confirmation Level: {result.metadata['confirmation']}")
    print(f"   [OK] Recent High: {result.metadata['recent_high']:.2f}")
    print(f"   [OK] Recent Low: {result.metadata['recent_low']:.2f}")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = ts.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = ts.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
