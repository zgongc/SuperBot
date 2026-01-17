# components/indicators/combo/smart_grok.py
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
class SmartGrok(BaseIndicator):
    """
    Smart Money Concept (SMC) - Geliştirilmiş versiyon
    FVG, Order Blocks, BOS/CHoCH, Market Structure analizi
    """

    def __init__(
        self,
        obv_period: int = 20,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        structure_lookback: int = 20,
        fvg_min_gap_percent: float = 0.015,
        ob_lookback: int = 15,
        logger=None,
        error_handler=None
    ):
        self.obv_period = obv_period
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.structure_lookback = structure_lookback
        self.fvg_min_gap = fvg_min_gap_percent
        self.ob_lookback = ob_lookback

        super().__init__(
            name='smart_grok',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'obv_period': obv_period,
                'rsi_period': rsi_period,
                'adx_period': adx_period,
                'adx_threshold': adx_threshold,
                'structure_lookback': structure_lookback,
                'fvg_min_gap_percent': fvg_min_gap_percent,
                'ob_lookback': ob_lookback
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.structure_lookback, self.adx_period, self.rsi_period) + 10

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate Smart Grok indicators"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # 1. FVG (Fair Value Gap) - son 3 mum
        fvg_bull = 0
        fvg_bear = 0
        if len(close) >= 3:
            if low[-2] > high[-3] and (low[-2] - high[-3]) / close[-3] >= self.fvg_min_gap:
                fvg_bull = 1
            if high[-2] < low[-3] and (low[-3] - high[-2]) / close[-3] >= self.fvg_min_gap:
                fvg_bear = 1

        # 2. Order Blocks
        ob_bull = 0
        ob_bear = 0
        if len(close) >= self.ob_lookback:
            window_high = np.max(high[-self.ob_lookback:-1])
            window_low = np.min(low[-self.ob_lookback:-1])
            window_vol_mean = np.mean(volume[-self.ob_lookback:-1])

            if close[-1] > window_high and volume[-1] > window_vol_mean * 1.5:
                ob_bull = 1
            if close[-1] < window_low and volume[-1] > window_vol_mean * 1.5:
                ob_bear = 1

        # 3. BOS (Break of Structure)
        bos = 0
        if len(close) >= self.structure_lookback:
            structure_high = np.max(high[-self.structure_lookback:-1])
            structure_low = np.min(low[-self.structure_lookback:-1])

            if high[-1] > structure_high and close[-1] > close[-2]:
                bos = 1
            elif low[-1] < structure_low and close[-1] < close[-2]:
                bos = -1

        # 4. RSI hesapla (basit)
        if len(close) >= self.rsi_period + 1:
            deltas = np.diff(close[-self.rsi_period-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50

        # 5. Market Structure
        market_structure = 0
        if bos == 1:
            market_structure = 1
        elif bos == -1:
            market_structure = -1

        # 6. SMC Score
        score = 0
        if fvg_bull == 1: score += 25
        if fvg_bear == 1: score -= 25
        if ob_bull == 1: score += 20
        if ob_bear == 1: score -= 20
        if bos == 1: score += 20
        if bos == -1: score -= 20
        if market_structure == 1: score += 15
        if market_structure == -1: score -= 15
        if rsi < 30: score += 10
        if rsi > 70: score -= 10

        # Signal
        if score >= 50:
            smc_signal = 1
        elif score <= -50:
            smc_signal = -1
        else:
            smc_signal = 0

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'fvg_bull': fvg_bull,
                'fvg_bear': fvg_bear,
                'ob_bull': ob_bull,
                'ob_bear': ob_bear,
                'bos': bos,
                'choch': 0,
                'market_structure': market_structure,
                'smc_score': score,
                'smc_signal': smc_signal
            },
            timestamp=timestamp,
            signal=SignalType.BUY if smc_signal == 1 else SignalType.SELL if smc_signal == -1 else SignalType.HOLD,
            trend=TrendDirection.UP if market_structure == 1 else TrendDirection.DOWN if market_structure == -1 else TrendDirection.NEUTRAL,
            strength=abs(score),
            metadata={
                'rsi': round(rsi, 2),
                'fvg_min_gap': self.fvg_min_gap
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Batch calculation for backtesting"""
        n = len(data)
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        results = {
            'fvg_bull': np.zeros(n, dtype=int),
            'fvg_bear': np.zeros(n, dtype=int),
            'ob_bull': np.zeros(n, dtype=int),
            'ob_bear': np.zeros(n, dtype=int),
            'bos': np.zeros(n, dtype=int),
            'choch': np.zeros(n, dtype=int),
            'market_structure': np.zeros(n, dtype=int),
            'smc_score': np.zeros(n),
            'smc_signal': np.zeros(n, dtype=int)
        }

        # FVG calculation
        for i in range(2, n):
            if low[i-1] > high[i-2] and (low[i-1] - high[i-2]) / close[i-2] >= self.fvg_min_gap:
                results['fvg_bull'][i] = 1
            if high[i-1] < low[i-2] and (low[i-2] - high[i-1]) / close[i-2] >= self.fvg_min_gap:
                results['fvg_bear'][i] = 1

        # Order Blocks
        for i in range(self.ob_lookback, n):
            window_high = np.max(high[i-self.ob_lookback:i])
            window_low = np.min(low[i-self.ob_lookback:i])
            window_vol_mean = np.mean(volume[i-self.ob_lookback:i])

            if close[i] > window_high and volume[i] > window_vol_mean * 1.5:
                results['ob_bull'][i] = 1
            if close[i] < window_low and volume[i] > window_vol_mean * 1.5:
                results['ob_bear'][i] = 1

        # BOS
        for i in range(self.structure_lookback, n):
            structure_high = np.max(high[i-self.structure_lookback:i])
            structure_low = np.min(low[i-self.structure_lookback:i])

            if high[i] > structure_high and close[i] > close[i-1]:
                results['bos'][i] = 1
            elif low[i] < structure_low and close[i] < close[i-1]:
                results['bos'][i] = -1

        # Market structure based on BOS
        results['market_structure'] = results['bos']

        # SMC Score and Signal
        for i in range(n):
            score = 0
            if results['fvg_bull'][i] == 1: score += 25
            if results['fvg_bear'][i] == 1: score -= 25
            if results['ob_bull'][i] == 1: score += 20
            if results['ob_bear'][i] == 1: score -= 20
            if results['bos'][i] == 1: score += 20
            if results['bos'][i] == -1: score -= 20
            if results['market_structure'][i] == 1: score += 15
            if results['market_structure'][i] == -1: score -= 15

            results['smc_score'][i] = score

            if score >= 50:
                results['smc_signal'][i] = 1
            elif score <= -50:
                results['smc_signal'][i] = -1

        return pd.DataFrame(results, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Uses BaseIndicator's _buffers[symbol] (populated by warmup_buffer())

        Args:
            candle: New candle data
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Updated SmartGrok values
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        if buffer_key not in self._buffers:
            self._buffers[buffer_key] = deque(maxlen=self.get_required_periods())

        # Normalize candle to dict format
        if isinstance(candle, dict):
            candle_dict = candle
        elif hasattr(candle, 'to_dict'):
            # Candle object with to_dict() method
            candle_dict = candle.to_dict()
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            candle_dict = {
                'timestamp': candle[0] if len(candle) > 0 else 0,
                'open': candle[1] if len(candle) > 1 else 0,
                'high': candle[2] if len(candle) > 2 else 0,
                'low': candle[3] if len(candle) > 3 else 0,
                'close': candle[4] if len(candle) > 4 else 0,
                'volume': candle[5] if len(candle) > 5 else 1000
            }

        # Add normalized candle to symbol's buffer
        self._buffers[buffer_key].append(candle_dict)

        # Extract timestamp for result
        timestamp_val = int(candle_dict.get('timestamp', 0))

        # Check minimum data
        if len(self._buffers[buffer_key]) < self.get_required_periods():
            return IndicatorResult(
                value={
                    'fvg_bull': 0,
                    'fvg_bear': 0,
                    'ob_bull': 0,
                    'ob_bear': 0,
                    'bos': 0,
                    'choch': 0,
                    'market_structure': 0,
                    'smc_score': 0,
                    'smc_signal': 0
                },
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'rsi': 50.0, 'fvg_min_gap': self.fvg_min_gap}
            )

        # Convert to DataFrame and calculate
        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))
        return self.calculate(buffer_data)