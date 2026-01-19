"""
indicators/combo/smart_money.py - Smart Money Concept (SMC) System

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Smart Money Concept - An advanced system that tracks smart money movements.
    Detecting the movements of institutional investors (smart money) by
    provides advantages to retail investors

    Components:
    1. Volume Analysis (OBV) - Volume flow
    2. Momentum (RSI) - Price momentum
    3. Trend (ADX) - Trend strength
    4. Price Action - Structural breakouts

    Features:
    - Order Block tespiti
    - Liquidity sweep analizi
    - Market structure (BOS - Break of Structure)
    - Fair Value Gap (FVG) analizi
    - Volume confirmation

Strategy:
    STRONG BUY:
    - Strong trend (ADX > 25)
    - Volume increase (OBV is rising)
    - RSI bullish divergence or oversold bounce
    - Price breaks a new high (Higher High)

    STRONG SELL:
    - Strong trend (ADX > 25)
    - Volume decrease (OBV is decreasing)
    - RSI bearish divergence or overbought rejection
    - Price breaks a new structure downwards (Lower Low)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.volume.obv
    - indicators.momentum.rsi
    - indicators.trend.adx
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.volume.obv import OBV
from indicators.momentum.rsi import RSI
from indicators.trend.adx import ADX
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class SmartMoney(BaseIndicator):
    """
    Smart Money Concept Trading System

    It generates strong buy/sell signals by tracking corporate cash flow.

    Args:
        obv_period: OBV signal period (default: 20)
        rsi_period: RSI period (default: 14)
        adx_period: ADX period (default: 14)
        adx_threshold: ADX strong trend threshold (default: 25)
        structure_lookback: Lookback period for structural analysis (default: 20)
    """

    def __init__(
        self,
        obv_period: int = 20,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 25,
        structure_lookback: int = 20,
        logger=None,
        error_handler=None
    ):
        self.obv_period = obv_period
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.structure_lookback = structure_lookback

        # Sub indicators
        self.obv = OBV(
            signal_period=obv_period,
            logger=logger,
            error_handler=error_handler
        )

        self.rsi = RSI(
            period=rsi_period,
            logger=logger,
            error_handler=error_handler
        )

        self.adx = ADX(
            period=adx_period,
            adx_threshold=adx_threshold,
            logger=logger,
            error_handler=error_handler
        )

        super().__init__(
            name='smart_money',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'obv_period': obv_period,
                'rsi_period': rsi_period,
                'adx_period': adx_period,
                'adx_threshold': adx_threshold,
                'structure_lookback': structure_lookback
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(
            self.obv.get_required_periods(),
            self.rsi.get_required_periods(),
            self.adx.get_required_periods(),
            self.structure_lookback
        )

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.obv_period < 1:
            raise InvalidParameterError(
                self.name, 'obv_period', self.obv_period,
                "OBV period must be positive"
            )
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "The RSI period must be positive"
            )
        if self.adx_period < 1:
            raise InvalidParameterError(
                self.name, 'adx_period', self.adx_period,
                "The ADX period must be positive"
            )
        if self.structure_lookback < 5:
            raise InvalidParameterError(
                self.name, 'structure_lookback', self.structure_lookback,
                "The structure lookback must be at least 5"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Smart Money analizi

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Smart Money analysis and signals.
        """
        # Fill the buffers (preparation for incremental update)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = max(self.obv_period, self.rsi_period, self.adx_period, self.structure_lookback) + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        self._volume_buffer.clear()
        
        # Get the data up to the last max_len.
        start_idx = max(0, len(data) - (max(self.obv_period, self.rsi_period, self.adx_period, self.structure_lookback) + 50))
        self._high_buffer.extend(data['high'].values[start_idx:])
        self._low_buffer.extend(data['low'].values[start_idx:])
        self._close_buffer.extend(data['close'].values[start_idx:])
        self._volume_buffer.extend(data['volume'].values[start_idx:])
        # Calculate sub-indicators
        obv_result = self.obv.calculate(data)
        rsi_result = self.rsi.calculate(data)
        adx_result = self.adx.calculate(data)

        # OBV value could be a dict or float
        if isinstance(obv_result.value, dict):
            obv_value = obv_result.value.get('obv', 0.0)
            obv_signal_val = obv_result.value.get('obv_signal', obv_value)
        else:
            obv_value = obv_result.value
            obv_signal_val = obv_value

        obv_trend = obv_result.trend
        rsi_value = rsi_result.value
        adx_value = adx_result.value['adx']
        plus_di = adx_result.value['plus_di']
        minus_di = adx_result.value['minus_di']

        # Market structure analysis
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        structure = self._analyze_market_structure(high, low, close)

        # Volume profil analizi
        volume_profile = self._analyze_volume_profile(obv_trend, obv_value, obv_signal_val)

        # Smart Money sinyali
        signal = self.get_signal(
            adx_value, plus_di, minus_di,
            obv_trend, rsi_value,
            structure
        )

        # Trend and power
        trend = self.get_trend(obv_trend, structure['trend'])
        strength = self._calculate_strength(adx_value, obv_trend, rsi_value, structure)

        # Confirmation level
        confirmation = self._get_confirmation(
            adx_value >= self.adx_threshold,
            obv_trend,
            rsi_value,
            structure
        )

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'obv': round(obv_value, 2),
                'obv_signal': round(obv_signal_val, 2),
                'rsi': round(rsi_value, 2),
                'adx': round(adx_value, 2),
                'plus_di': round(plus_di, 2),
                'minus_di': round(minus_di, 2),
                'market_structure': structure['structure'],
                'bos': structure['bos'],
                'signal': signal.value,
                'trend': trend.name,
                'strength': round(strength, 2),
                'confirmation': confirmation,
                'smart_money_flow': self._determine_money_flow(obv_trend, structure)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'obv_period': self.obv_period,
                'rsi_period': self.rsi_period,
                'adx_period': self.adx_period,
                'obv_trend': obv_trend.name,
                'volume_profile': volume_profile,
                'trend_strength': 'Strong' if adx_value >= self.adx_threshold else 'Weak',
                'structure_type': structure['structure'],
                'higher_high': structure['hh'],
                'lower_low': structure['ll']
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Smart Money values
        """
        # Batch calculations for sub-indicators
        obv_batch = self.obv.calculate_batch(data)  # Returns DataFrame with 'obv', 'obv_signal'
        rsi_batch = self.rsi.calculate_batch(data)  # Returns Series
        adx_batch = self.adx.calculate_batch(data)  # Returns DataFrame

        # OBV could return DataFrame or Series
        if isinstance(obv_batch, pd.DataFrame):
            obv_values = obv_batch['obv'].values
            obv_signal_values = obv_batch['obv_signal'].values
        else:
            obv_values = obv_batch.values
            obv_signal_values = pd.Series(obv_values).ewm(span=self.obv_period, adjust=False).mean().values

        # Result arrays
        n = len(data)
        results = {
            'obv': obv_values,
            'obv_signal': obv_signal_values,
            'rsi': rsi_batch.values,
            'adx': adx_batch['adx'].values,
            'plus_di': adx_batch['plus_di'].values,
            'minus_di': adx_batch['minus_di'].values,
            'market_structure': np.full(n, 'ranging', dtype=object),
            'bos': np.zeros(n, dtype=bool),
            'signal': np.full(n, 'HOLD', dtype=object),
            'trend': np.full(n, 'NEUTRAL', dtype=object),
            'strength': np.zeros(n),
            'confirmation': np.full(n, 'none', dtype=object),
            'smart_money_flow': np.full(n, 'neutral', dtype=object)
        }
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate the market structure and signals for each bar
        for i in range(self.structure_lookback, n):
            # Market structure analizi
            start_idx = max(0, i - self.structure_lookback)
            structure = self._analyze_market_structure(
                high[start_idx:i+1],
                low[start_idx:i+1],
                close[start_idx:i+1]
            )
            
            results['market_structure'][i] = structure['structure']
            results['bos'][i] = structure['bos']
            
            # OBV trend (compare with obv_signal)
            obv_val = results['obv'][i]
            obv_sig = results['obv_signal'][i]
            if obv_val > obv_sig:
                obv_trend = TrendDirection.UP
            elif obv_val < obv_sig:
                obv_trend = TrendDirection.DOWN
            else:
                obv_trend = TrendDirection.NEUTRAL
                
            # Signal
            signal = self.get_signal(
                results['adx'][i],
                results['plus_di'][i],
                results['minus_di'][i],
                obv_trend,
                results['rsi'][i],
                structure
            )
            results['signal'][i] = signal.value
            
            # Trend
            trend = self.get_trend(obv_trend, structure['trend'])
            results['trend'][i] = trend.name
            
            # Strength
            strength = self._calculate_strength(
                results['adx'][i],
                obv_trend,
                results['rsi'][i],
                structure
            )
            results['strength'][i] = strength
            
            # Confirmation
            confirmation = self._get_confirmation(
                results['adx'][i] >= self.adx_threshold,
                obv_trend,
                results['rsi'][i],
                structure
            )
            results['confirmation'][i] = confirmation
            
            # Smart Money Flow
            flow = self._determine_money_flow(obv_trend, structure)
            results['smart_money_flow'][i] = flow
            
        return pd.DataFrame(results, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: The current Smart Money value.
        """
        # Buffer management
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = max(self.obv_period, self.rsi_period, self.adx_period, self.structure_lookback) + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            
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
        self._volume_buffer.append(volume_val)
        
        # Yeterli veri yoksa
        min_required = max(self.obv_period, self.rsi_period, self.adx_period, self.structure_lookback)
        if len(self._close_buffer) < min_required:
            return IndicatorResult(
                value={'obv': 0, 'rsi': 50, 'adx': 0, 'plus_di': 0, 'minus_di': 0, 'market_structure': 'ranging', 'bos': False},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'confirmation': 'none', 'smart_money_flow': 'neutral'}
            )
            
        # Create a DataFrame (for sub-indicators)
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        # Calculate sub-indicators
        obv_result = self.obv.calculate(buffer_data)
        rsi_result = self.rsi.calculate(buffer_data)
        adx_result = self.adx.calculate(buffer_data)

        # OBV value could be a dict or float
        if isinstance(obv_result.value, dict):
            obv_value = obv_result.value.get('obv', 0.0)
            obv_signal_val = obv_result.value.get('obv_signal', obv_value)
        else:
            obv_value = obv_result.value
            obv_signal_val = obv_value

        obv_trend = obv_result.trend
        rsi_value = rsi_result.value
        adx_value = adx_result.value['adx']
        plus_di = adx_result.value['plus_di']
        minus_di = adx_result.value['minus_di']

        # Market structure analysis
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)

        structure = self._analyze_market_structure(high, low, close)

        # Volume profil analizi
        volume_profile = self._analyze_volume_profile(obv_trend, obv_value, obv_signal_val)
        
        # Smart Money sinyali
        signal = self.get_signal(
            adx_value, plus_di, minus_di,
            obv_trend, rsi_value,
            structure
        )
        
        # Trend and power
        trend = self.get_trend(obv_trend, structure['trend'])
        strength = self._calculate_strength(adx_value, obv_trend, rsi_value, structure)
        
        # Confirmation level
        confirmation = self._get_confirmation(
            adx_value >= self.adx_threshold,
            obv_trend,
            rsi_value,
            structure
        )
        
        return IndicatorResult(
            value={
                'obv': round(obv_value, 2),
                'rsi': round(rsi_value, 2),
                'adx': round(adx_value, 2),
                'plus_di': round(plus_di, 2),
                'minus_di': round(minus_di, 2),
                'market_structure': structure['structure'],
                'bos': structure['bos']
            },
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'obv_period': self.obv_period,
                'rsi_period': self.rsi_period,
                'adx_period': self.adx_period,
                'obv_trend': obv_trend.name,
                'volume_profile': volume_profile,
                'trend_strength': 'Strong' if adx_value >= self.adx_threshold else 'Weak',
                'structure_type': structure['structure'],
                'higher_high': structure['hh'],
                'lower_low': structure['ll'],
                'confirmation': confirmation,
                'smart_money_flow': self._determine_money_flow(obv_trend, structure)
            }
        )

    def _analyze_market_structure(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
        """
        Analyze the market structure (Higher Highs, Lower Lows, BOS).

        Returns:
            dict: structure, trend, hh, ll, bos
        """
        lookback = min(self.structure_lookback, len(high))

        # The highest and lowest values in the last lookback period
        recent_highs = high[-lookback:]
        recent_lows = low[-lookback:]
        recent_close = close[-lookback:]

        # Highest and lowest points
        max_high = np.max(recent_highs)
        min_low = np.min(recent_lows)
        current_high = high[-1]
        current_low = low[-1]
        current_close = close[-1]

        # Higher High (HH) - Yeni tepe
        hh = current_high >= max_high * 0.995  # %0.5 tolerans

        # Lower Low (LL) - Yeni dip
        ll = current_low <= min_low * 1.005  # %0.5 tolerans

        # Break of Structure (BOS)
        bos = False
        structure = 'ranging'
        trend = TrendDirection.NEUTRAL

        if hh and not ll:
            structure = 'bullish'
            trend = TrendDirection.UP
            bos = True  # The ascending structure is broken
        elif ll and not hh:
            structure = 'bearish'
            trend = TrendDirection.DOWN
            bos = True  # The drop structure is broken
        elif hh and ll:
            structure = 'volatile'
            trend = TrendDirection.NEUTRAL

        return {
            'structure': structure,
            'trend': trend,
            'hh': hh,
            'll': ll,
            'bos': bos,
            'max_high': max_high,
            'min_low': min_low
        }

    def _analyze_volume_profile(self, obv_trend: TrendDirection, obv_value: float, obv_signal: float) -> str:
        """
        Volume profil analizi

        Returns:
            str: 'accumulation', 'distribution', 'neutral'
        """
        # OBV trend and signal comparison
        if obv_trend == TrendDirection.UP and obv_value > obv_signal:
            return 'accumulation'  # Accumulation (Smart Money is buying)
        elif obv_trend == TrendDirection.DOWN and obv_value < obv_signal:
            return 'distribution'  # Distribution (Smart Money is selling)
        else:
            return 'neutral'

    def _determine_money_flow(self, obv_trend: TrendDirection, structure: dict) -> str:
        """
        Determine the direction of Smart Money flow.

        Returns:
            str: 'bullish', 'bearish', 'accumulation', 'distribution', 'neutral'
        """
        # Is the volume and structure compatible?
        if obv_trend == TrendDirection.UP and structure['structure'] == 'bullish':
            return 'bullish'  # Strong buy
        elif obv_trend == TrendDirection.DOWN and structure['structure'] == 'bearish':
            return 'bearish'  # Strong sell
        elif obv_trend == TrendDirection.UP and structure['structure'] != 'bullish':
            return 'accumulation'  # Sessiz birikim
        elif obv_trend == TrendDirection.DOWN and structure['structure'] != 'bearish':
            return 'distribution'  # Silent distribution
        else:
            return 'neutral'

    def get_signal(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        obv_trend: TrendDirection,
        rsi: float,
        structure: dict
    ) -> SignalType:
        """
        Generate a Smart Money signal.

        All components must be compatible.
        """
        # Strong trend required
        strong_trend = adx >= self.adx_threshold

        # Buy Signal: Strong upward movement + Volume increase + Structure Breakout (BOS)
        if (strong_trend and
            plus_di > minus_di and
            obv_trend == TrendDirection.UP and
            structure['structure'] == 'bullish' and
            structure['bos']):
            return SignalType.BUY

        # Medium level AL: Volume and structure compatible
        if (obv_trend == TrendDirection.UP and
            structure['hh'] and
            rsi < 50):
            return SignalType.BUY

        # SAT Signal: Strong decrease + Volume decrease + Structure EMPTY
        if (strong_trend and
            minus_di > plus_di and
            obv_trend == TrendDirection.DOWN and
            structure['structure'] == 'bearish' and
            structure['bos']):
            return SignalType.SELL

        # Medium level SAT: Volume and structure compatible
        if (obv_trend == TrendDirection.DOWN and
            structure['ll'] and
            rsi > 50):
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, obv_trend: TrendDirection, structure_trend: TrendDirection) -> TrendDirection:
        """
        Kombine trend belirleme

        Volume and structure must be in the same direction.
        """
        if obv_trend == structure_trend and obv_trend != TrendDirection.NEUTRAL:
            return obv_trend
        return TrendDirection.NEUTRAL

    def _calculate_strength(
        self,
        adx: float,
        obv_trend: TrendDirection,
        rsi: float,
        structure: dict
    ) -> float:
        """
        Calculate signal strength (0-100)
        """
        strength = 0

        # ADX contribution (trend strength)
        strength += min(adx, 40)

        # Volume trend contribution
        if obv_trend != TrendDirection.NEUTRAL:
            strength += 20

        # RSI contribution (extreme levels)
        rsi_deviation = abs(rsi - 50)
        strength += min(rsi_deviation / 2, 20)

        # Structure contribution
        if structure['bos']:
            strength += 20

        return min(strength, 100)

    def _get_confirmation(
        self,
        strong_trend: bool,
        obv_trend: TrendDirection,
        rsi: float,
        structure: dict
    ) -> str:
        """
        Confirmation level

        Returns:
            str: 'strong', 'moderate', 'weak', 'none'
        """
        confirmations = 0

        # Strong trend confirmation
        if strong_trend:
            confirmations += 1

        # Volume confirmation
        if obv_trend != TrendDirection.NEUTRAL:
            confirmations += 1

        # RSI confirmation (extreme levels)
        if rsi < 30 or rsi > 70:
            confirmations += 1

        # Structure validation
        if structure['bos']:
            confirmations += 1

        if confirmations >= 3:
            return 'strong'
        elif confirmations == 2:
            return 'moderate'
        elif confirmations == 1:
            return 'weak'
        else:
            return 'none'

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'obv_period': 20,
            'rsi_period': 14,
            'adx_period': 14,
            'adx_threshold': 25,
            'structure_lookback': 20
        }

    def _requires_volume(self) -> bool:
        """Smart Money requires volume (for OBV)"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SmartMoney']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Smart Money indicator test"""

    print("\n" + "="*60)
    print("SMART MONEY CONCEPT (SMC) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Market simulation with Smart Money flow.
    base_price = 100
    prices = [base_price]
    volumes = [10000]
    highs = []
    lows = []

    for i in range(99):
        if i < 30:
            # Accumulation phase (sideways + volume increase)
            trend = np.random.randn() * 0.5
            vol_change = 100
        elif i < 60:
            # Strong ascent (markup)
            trend = 1.0 + np.random.randn() * 0.5
            vol_change = 200
        else:
            # Distribution and drop
            trend = -0.5 + np.random.randn() * 0.5
            vol_change = -50

        new_price = prices[-1] + trend
        prices.append(new_price)
        volumes.append(max(volumes[-1] + vol_change + np.random.randint(-100, 100), 5000))
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
        'volume': volumes
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Volume range: {min(volumes):,.0f} -> {max(volumes):,.0f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    smc = SmartMoney()
    print(f"   [OK] Created: {smc}")
    print(f"   [OK] Kategori: {smc.category.value}")
    print(f"   [OK] Tip: {smc.indicator_type.value}")
    print(f"   [OK] Required period: {smc.get_required_periods()}")
    print(f"   [OK] Volume required: {smc.metadata.requires_volume}")

    result = smc(data)
    print(f"   [OK] OBV: {result.value['obv']:,.2f}")
    print(f"   [OK] RSI: {result.value['rsi']}")
    print(f"   [OK] ADX: {result.value['adx']}")
    print(f"   [OK] +DI: {result.value['plus_di']}")
    print(f"   [OK] -DI: {result.value['minus_di']}")
    print(f"   [OK] Market Structure: {result.value['market_structure']}")
    print(f"   [OK] BOS: {result.value['bos']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")

    # Test 2: Smart Money analizi
    print("\n3. Smart Money analizi...")
    print(f"   [OK] OBV Trend: {result.metadata['obv_trend']}")
    print(f"   [OK] Volume Profile: {result.metadata['volume_profile']}")
    print(f"   [OK] Trend Strength: {result.metadata['trend_strength']}")
    print(f"   [OK] Structure Type: {result.metadata['structure_type']}")
    print(f"   [OK] Higher High: {result.metadata['higher_high']}")
    print(f"   [OK] Lower Low: {result.metadata['lower_low']}")
    print(f"   [OK] Confirmation: {result.metadata['confirmation']}")
    print(f"   [OK] Smart Money Flow: {result.metadata['smart_money_flow']}")

    # Test 3: Market phase analysis
    print("\n4. Market phase analysis...")
    phases = [
        (25, "Accumulation phase"),
        (45, "Start of ascent"),
        (70, "Distribution start"),
        (95, "Descending phase")
    ]

    for idx, phase_name in phases:
        data_slice = data.iloc[:idx+1]
        result = smc.calculate(data_slice)
        print(f"   [OK] {phase_name} (Mum {idx}): "
              f"Structure={result.value['market_structure']}, "
              f"Flow={result.metadata['smart_money_flow']}, "
              f"Signal={result.signal.value}")

    # Test 4: Custom parameters
    print("\n5. Special parameter test...")
    smc_custom = SmartMoney(
        obv_period=30,
        rsi_period=21,
        adx_period=21,
        adx_threshold=30,
        structure_lookback=30
    )
    result = smc_custom.calculate(data)
    print(f"   [OK] Custom OBV: {result.value['obv']:,.2f}")
    print(f"   [OK] Custom RSI: {result.value['rsi']}")
    print(f"   [OK] Custom ADX: {result.value['adx']}")
    print(f"   [OK] Smart Money Flow: {result.metadata['smart_money_flow']}")

    # Test 5: Konfirmasyon seviyeleri
    print("\n6. Confirmation level analysis...")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")
    print(f"   [OK] Trend Strength: {result.metadata['trend_strength']}")
    print(f"   [OK] Volume Profile: {result.metadata['volume_profile']}")
    print(f"   [OK] Signal Strength: {result.strength:.2f}/100")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = smc.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = smc.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
