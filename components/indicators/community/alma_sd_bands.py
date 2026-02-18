"""
indicators/community/alma_sd_bands.py - ALMA Standard Deviation Bands

Version: 1.0.0
Date: 2026-02-18
Author: SuperBot Team

Description:
    ALMA SD Bands - Arnaud Legoux Moving Average with Standard Deviation Bands
    Similar to Bollinger Bands but uses ALMA instead of SMA for smoother,
    lower-lag results.

    Pine Script Reference: "ALMA SD Bands | RakoQuant" by RakoQuant

Formula:
    Basis  = ALMA(source, period, offset, sigma)
    Vol    = StdDev(source, period)
    SmVol  = ALMA(Vol, vol_smooth, offset, sigma)    # Smoothed volatility
    Upper  = Basis + mult * SmVol
    Lower  = Basis - mult * SmVol

    Regime (with deadband):
        BULL  = close > basis + dead
        BEAR  = close < basis - dead
        dead  = deadband_mult * SmVol

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.trend.alma (ALMA)
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.alma import ALMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class ALMAStdDevBands(BaseIndicator):
    """
    ALMA Standard Deviation Bands

    ALMA-based volatility bands with smoothed standard deviation.
    Provides smoother bands than Bollinger with less lag.

    Args:
        period: ALMA/StdDev period (default: 111)
        offset: ALMA Gaussian offset 0..1 (default: 0.85)
        sigma: ALMA Gaussian width (default: 6.0)
        mult: Band multiplier in sigma units (default: 0.4)
        source: Price source (default: 'high')
        vol_smooth: Volatility smoothing period (default: 75)
        deadband_mult: Deadband multiplier for regime detection (default: 0.35)
    """

    def __init__(
        self,
        period: int = 111,
        offset: float = 0.85,
        sigma: float = 6.0,
        mult: float = 0.4,
        source: str = 'high',
        vol_smooth: int = 75,
        deadband_mult: float = 0.35,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.offset = offset
        self.sigma = sigma
        self.mult = mult
        self.source = source
        self.vol_smooth = vol_smooth
        self.deadband_mult = deadband_mult

        # Internal ALMA for basis
        self._alma_basis = ALMA(period=period, offset=offset, sigma=sigma, source=source)
        # Internal ALMA for volatility smoothing
        self._alma_vol = ALMA(period=vol_smooth, offset=offset, sigma=sigma, source='close')

        super().__init__(
            name='alma_sd_bands',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'offset': offset,
                'sigma': sigma,
                'mult': mult,
                'source': source,
                'vol_smooth': vol_smooth,
                'deadband_mult': deadband_mult
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.period, self.vol_smooth) + 10

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be at least 2"
            )
        if self.mult <= 0:
            raise InvalidParameterError(
                self.name, 'mult', self.mult,
                "Band multiplier must be positive"
            )
        return True

    def _get_source(self, data: pd.DataFrame) -> pd.Series:
        """Get price source series"""
        if self.source == 'high':
            return data['high']
        elif self.source == 'low':
            return data['low']
        elif self.source == 'hl2':
            return (data['high'] + data['low']) / 2
        elif self.source == 'hlc3':
            return (data['high'] + data['low'] + data['close']) / 3
        else:
            return data['close']

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate ALMA SD Bands

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: upper, middle, lower values
        """
        src = self._get_source(data).values
        close = data['close'].values

        # 1. Basis = ALMA(source)
        alma_weights = self._alma_basis._alma_weights(self.period)
        window = src[-self.period:]
        basis = np.sum(alma_weights * window)

        # 2. Raw volatility = StdDev(source, period)
        raw_vol = np.std(src[-self.period:], ddof=0)

        # 3. Smoothed volatility (simplified - use raw_vol for single calculation)
        vol = raw_vol

        # 4. Bands
        upper = basis + self.mult * vol
        lower = basis - self.mult * vol

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Regime detection
        dead = self.deadband_mult * vol
        if current_price > basis + dead:
            regime = 'bull'
        elif current_price < basis - dead:
            regime = 'bear'
        else:
            regime = 'neutral'

        # Warmup buffer for update()
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper, 8),
                'middle': round(basis, 8),
                'lower': round(lower, 8)
            },
            timestamp=timestamp,
            signal=SignalType.BUY if regime == 'bull' else (SignalType.SELL if regime == 'bear' else SignalType.HOLD),
            trend=TrendDirection.UP if regime == 'bull' else (TrendDirection.DOWN if regime == 'bear' else TrendDirection.NEUTRAL),
            strength=min(abs((current_price - basis) / vol * 50), 100) if vol > 0 else 0,
            metadata={
                'period': self.period,
                'mult': self.mult,
                'source': self.source,
                'vol_smooth': self.vol_smooth,
                'deadband_mult': self.deadband_mult,
                'regime': regime,
                'volatility': round(vol, 8),
                'bandwidth': round((upper - lower) / basis * 100, 4) if basis != 0 else 0,
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        VECTORIZED batch ALMA SD Bands calculation - for BACKTEST

        Returns:
            pd.DataFrame: 3 columns (upper, middle, lower)
        """
        self._validate_data(data)

        src = self._get_source(data).values
        n = len(src)

        # 1. Basis = ALMA(source, period) - vectorized
        alma_weights = self._alma_basis._alma_weights(self.period)
        basis = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = src[i - self.period + 1:i + 1]
            basis[i] = np.sum(alma_weights * window)

        # 2. Raw volatility = rolling StdDev(source, period)
        src_series = pd.Series(src)
        raw_vol = src_series.rolling(window=self.period).std(ddof=0).values

        # 3. Smoothed volatility = ALMA(raw_vol, vol_smooth)
        vol_weights = self._alma_vol._alma_weights(self.vol_smooth)
        smooth_vol = np.full(n, np.nan)
        for i in range(self.period + self.vol_smooth - 2, n):
            vol_window = raw_vol[i - self.vol_smooth + 1:i + 1]
            if not np.any(np.isnan(vol_window)):
                smooth_vol[i] = np.sum(vol_weights * vol_window)

        # Fallback: use raw_vol where smooth_vol is NaN
        vol = np.where(np.isnan(smooth_vol), raw_vol, smooth_vol)

        # 4. Bands
        upper = basis + self.mult * vol
        lower = basis - self.mult * vol

        result = pd.DataFrame({
            'upper': upper,
            'middle': basis,
            'lower': lower
        }, index=data.index)

        return result

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer for update() method"""
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._src_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)

        src = self._get_source(data)
        for val in src.tail(max_len).values:
            self._src_buffer.append(val)
        for val in data['close'].tail(max_len).values:
            self._close_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_src_buffer'):
            from collections import deque
            self._src_buffer = deque(maxlen=self.get_required_periods() + 50)
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Extract source value
        if isinstance(candle, dict):
            if self.source == 'high':
                src_val = float(candle['high'])
            elif self.source == 'low':
                src_val = float(candle['low'])
            elif self.source == 'hl2':
                src_val = (float(candle['high']) + float(candle['low'])) / 2
            elif self.source == 'hlc3':
                src_val = (float(candle['high']) + float(candle['low']) + float(candle['close'])) / 3
            else:
                src_val = float(candle['close'])
            close_val = float(candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            src_val = float(candle[4]) if len(candle) > 4 else 0.0
            close_val = src_val

        self._src_buffer.append(src_val)
        self._close_buffer.append(close_val)

        if len(self._src_buffer) < self.period:
            return IndicatorResult(
                value={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        # Calculate from buffer
        src_arr = np.array(list(self._src_buffer))

        # Basis
        alma_weights = self._alma_basis._alma_weights(self.period)
        window = src_arr[-self.period:]
        basis = np.sum(alma_weights * window)

        # Volatility
        raw_vol = np.std(src_arr[-self.period:], ddof=0)
        vol = raw_vol

        # Bands
        upper = basis + self.mult * vol
        lower = basis - self.mult * vol

        # Regime
        dead = self.deadband_mult * vol
        if close_val > basis + dead:
            regime = 'bull'
        elif close_val < basis - dead:
            regime = 'bear'
        else:
            regime = 'neutral'

        return IndicatorResult(
            value={
                'upper': round(upper, 8),
                'middle': round(basis, 8),
                'lower': round(lower, 8)
            },
            timestamp=timestamp_val,
            signal=SignalType.BUY if regime == 'bull' else (SignalType.SELL if regime == 'bear' else SignalType.HOLD),
            trend=TrendDirection.UP if regime == 'bull' else (TrendDirection.DOWN if regime == 'bear' else TrendDirection.NEUTRAL),
            strength=min(abs((close_val - basis) / vol * 50), 100) if vol > 0 else 0,
            metadata={
                'period': self.period,
                'mult': self.mult,
                'source': self.source,
                'vol_smooth': self.vol_smooth,
                'deadband_mult': self.deadband_mult,
                'regime': regime,
                'volatility': round(vol, 8),
                'bandwidth': round((upper - lower) / basis * 100, 4) if basis != 0 else 0,
                'price': round(close_val, 8)
            }
        )

    def get_signal(self, percent_b: float) -> SignalType:
        """Signal from %B position"""
        if percent_b > 1.0:
            return SignalType.BUY
        elif percent_b < 0.0:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, middle: float) -> TrendDirection:
        """Trend from price vs middle band"""
        if price > middle:
            return TrendDirection.UP
        elif price < middle:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 111,
            'offset': 0.85,
            'sigma': 6.0,
            'mult': 0.4,
            'source': 'high',
            'vol_smooth': 75,
            'deadband_mult': 0.35
        }

    def _requires_volume(self) -> bool:
        """Volume not required"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ALMAStdDevBands']
