"""
indicators/structure/smc.py - SMC (Smart Money Concepts) Indicator

Version: 1.0.0
Date: 2025-12-08
Author: SuperBot Team

Açıklama:
    Smart Money Concepts (SMC) - Tüm bileşenler tek indicator'de

    Bileşenler:
    1. Swing Points - Internal hesaplama (left/right bar analizi)
    2. BOS (Break of Structure) - Yapı kırılması
    3. CHoCH (Change of Character) - Trend değişimi
    4. FVG (Fair Value Gap) - Boşluk bölgeleri
    5. Order Blocks - Kurumsal emir bölgeleri (opsiyonel)

    Output:
    - smc_signal: 1=bullish, -1=bearish, 0=neutral
    - bos: 1=bullish BOS, -1=bearish BOS, 0=none
    - choch: 1=bullish CHoCH, -1=bearish CHoCH, 0=none
    - fvg: >0 bullish dominance, <0 bearish dominance
    - swing_high: En son swing high seviyesi
    - swing_low: En son swing low seviyesi
    - ob_bullish: Aktif bullish order block (top, bottom)
    - ob_bearish: Aktif bearish order block (top, bottom)

Avantajlar:
    - Tek indicator ekleme ile tüm SMC analizi
    - calculate_batch() ile optimize backtest
    - Internal hesaplamalar - ayrı indicator dependency yok
    - custom_parameters entegrasyonu

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class SMC(BaseIndicator):
    """
    Smart Money Concepts (SMC) Indicator

    Tüm SMC bileşenlerini tek bir indicator'de birleştirir.
    Backtest için optimize edilmiş calculate_batch() methodu ile.

    Args:
        left_bars: Swing point için sol bar sayısı (varsayılan: 5)
        right_bars: Swing point için sağ bar sayısı (varsayılan: 5)
        max_levels: Takip edilecek maksimum swing seviyesi (varsayılan: 5)
        trend_strength: Trend tespiti için minimum swing sayısı (varsayılan: 2)
        fvg_min_size_pct: Minimum FVG boyutu % (varsayılan: 0.1)
        fvg_max_age: FVG maksimum yaşı (bar) (varsayılan: 50)
        ob_strength_threshold: Order Block güç eşiği % (varsayılan: 1.0)
        ob_max_blocks: Maksimum aktif OB sayısı (varsayılan: 3)
        ob_enabled: Order Block tespiti aktif mi (varsayılan: True)
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        max_levels: int = 5,
        trend_strength: int = 2,
        fvg_min_size_pct: float = 0.1,
        fvg_max_age: int = 50,
        ob_strength_threshold: float = 1.0,
        ob_max_blocks: int = 3,
        ob_enabled: bool = True,
        logger=None,
        error_handler=None
    ):
        # Swing Points params
        self.left_bars = left_bars
        self.right_bars = right_bars

        # BOS/CHoCH params
        self.max_levels = max_levels
        self.trend_strength = trend_strength

        # FVG params
        self.fvg_min_size_pct = fvg_min_size_pct
        self.fvg_max_age = fvg_max_age

        # Order Block params
        self.ob_strength_threshold = ob_strength_threshold
        self.ob_max_blocks = ob_max_blocks
        self.ob_enabled = ob_enabled

        super().__init__(
            name='smc',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'max_levels': max_levels,
                'trend_strength': trend_strength,
                'fvg_min_size_pct': fvg_min_size_pct,
                'fvg_max_age': fvg_max_age,
                'ob_strength_threshold': ob_strength_threshold,
                'ob_max_blocks': ob_max_blocks,
                'ob_enabled': ob_enabled
            },
            logger=logger,
            error_handler=error_handler
        )

        # Buffers for real-time updates
        self._buffers_init = False

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(
            self.left_bars + self.right_bars + 5,
            self.trend_strength * 3,
            self.fvg_max_age
        )

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.left_bars < 1:
            raise InvalidParameterError(
                self.name, 'left_bars', self.left_bars,
                "Left bars en az 1 olmalı"
            )
        if self.right_bars < 1:
            raise InvalidParameterError(
                self.name, 'right_bars', self.right_bars,
                "Right bars en az 1 olmalı"
            )
        if self.max_levels < 1:
            raise InvalidParameterError(
                self.name, 'max_levels', self.max_levels,
                "Max levels en az 1 olmalı"
            )
        return True

    # =========================================================================
    # INTERNAL: SWING POINTS
    # =========================================================================

    def _detect_swing_points(
        self,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Swing high ve swing low noktalarını tespit et

        Returns:
            Tuple[List[Dict], List[Dict]]: (swing_highs, swing_lows)
        """
        swing_highs = []
        swing_lows = []

        n = len(highs)

        for i in range(self.left_bars, n - self.right_bars):
            # Swing High kontrolü
            is_swing_high = True
            for j in range(1, self.left_bars + 1):
                if highs[i] <= highs[i - j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                for j in range(1, self.right_bars + 1):
                    if highs[i] <= highs[i + j]:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'value': highs[i]
                })

            # Swing Low kontrolü
            is_swing_low = True
            for j in range(1, self.left_bars + 1):
                if lows[i] >= lows[i - j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                for j in range(1, self.right_bars + 1):
                    if lows[i] >= lows[i + j]:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'value': lows[i]
                })

        return swing_highs, swing_lows

    # =========================================================================
    # INTERNAL: TREND DETECTION
    # =========================================================================

    def _detect_trend(
        self,
        recent_highs: List[Dict],
        recent_lows: List[Dict]
    ) -> str:
        """
        Mevcut trendi tespit et (swing yapısına göre)

        Returns:
            str: 'uptrend', 'downtrend', 'ranging'
        """
        if len(recent_highs) < self.trend_strength or len(recent_lows) < self.trend_strength:
            return 'ranging'

        # Son swing high'ları karşılaştır (Higher Highs?)
        last_highs = [h['value'] for h in recent_highs[-self.trend_strength:]]
        higher_highs = all(last_highs[i] > last_highs[i-1] for i in range(1, len(last_highs)))

        # Son swing low'ları karşılaştır (Higher Lows?)
        last_lows = [l['value'] for l in recent_lows[-self.trend_strength:]]
        higher_lows = all(last_lows[i] > last_lows[i-1] for i in range(1, len(last_lows)))

        # Lower Lows ve Lower Highs?
        lower_highs = all(last_highs[i] < last_highs[i-1] for i in range(1, len(last_highs)))
        lower_lows = all(last_lows[i] < last_lows[i-1] for i in range(1, len(last_lows)))

        if higher_highs and higher_lows:
            return 'uptrend'
        elif lower_highs and lower_lows:
            return 'downtrend'
        else:
            return 'ranging'

    # =========================================================================
    # INTERNAL: BOS DETECTION
    # =========================================================================

    def _detect_bos_batch(
        self,
        closes: np.ndarray,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> np.ndarray:
        """
        Break of Structure batch hesaplama (ONLY NEW BREAKS)

        Returns:
            np.ndarray: BOS sinyali (1=bullish, -1=bearish, 0=none)
        """
        n = len(closes)
        bos_signal = np.zeros(n, dtype=int)

        # Convert to dict for fast lookup
        swing_high_dict = {s['index']: s['value'] for s in swing_highs}
        swing_low_dict = {s['index']: s['value'] for s in swing_lows}

        # Track recent swing levels
        recent_highs = []
        recent_lows = []

        # Track broken levels (for NEW break detection)
        broken_highs = set()
        broken_lows = set()

        start_idx = self.left_bars + self.right_bars

        for i in range(start_idx, n):
            # Update recent swings
            if i in swing_high_dict:
                recent_highs.append({'index': i, 'value': swing_high_dict[i]})
                if len(recent_highs) > self.max_levels:
                    recent_highs.pop(0)

            if i in swing_low_dict:
                recent_lows.append({'index': i, 'value': swing_low_dict[i]})
                if len(recent_lows) > self.max_levels:
                    recent_lows.pop(0)

            current_close = closes[i]
            prev_close = closes[i - 1] if i > 0 else current_close

            # Bullish BOS: NEW break above swing high
            for swing in reversed(recent_highs[-self.max_levels:]):
                swing_idx = swing['index']
                swing_val = swing['value']

                if current_close > swing_val and prev_close <= swing_val:
                    if swing_idx not in broken_highs:
                        bos_signal[i] = 1  # Bullish BOS
                        broken_highs.add(swing_idx)
                        break
                elif current_close > swing_val:
                    broken_highs.add(swing_idx)

            # Bearish BOS: NEW break below swing low
            if bos_signal[i] == 0:
                for swing in reversed(recent_lows[-self.max_levels:]):
                    swing_idx = swing['index']
                    swing_val = swing['value']

                    if current_close < swing_val and prev_close >= swing_val:
                        if swing_idx not in broken_lows:
                            bos_signal[i] = -1  # Bearish BOS
                            broken_lows.add(swing_idx)
                            break
                    elif current_close < swing_val:
                        broken_lows.add(swing_idx)

        return bos_signal

    # =========================================================================
    # INTERNAL: CHoCH DETECTION
    # =========================================================================

    def _detect_choch_batch(
        self,
        closes: np.ndarray,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> np.ndarray:
        """
        Change of Character batch hesaplama (ONLY NEW BREAKS)

        CHoCH: Trend değişimi - uptrend'de swing low kırılması veya
               downtrend'de swing high kırılması

        Returns:
            np.ndarray: CHoCH sinyali (1=bullish, -1=bearish, 0=none)
        """
        n = len(closes)
        choch_signal = np.zeros(n, dtype=int)

        # Convert to dict for fast lookup
        swing_high_dict = {s['index']: s['value'] for s in swing_highs}
        swing_low_dict = {s['index']: s['value'] for s in swing_lows}

        # Track recent swings for trend detection
        recent_highs = []
        recent_lows = []

        # Track broken levels
        broken_highs = set()
        broken_lows = set()

        start_idx = self.left_bars + self.right_bars + self.trend_strength

        for i in range(start_idx, n):
            # Update recent swings
            if i in swing_high_dict:
                recent_highs.append({'index': i, 'value': swing_high_dict[i]})
                if len(recent_highs) > self.trend_strength * 2:
                    recent_highs = recent_highs[-self.trend_strength * 2:]

            if i in swing_low_dict:
                recent_lows.append({'index': i, 'value': swing_low_dict[i]})
                if len(recent_lows) > self.trend_strength * 2:
                    recent_lows = recent_lows[-self.trend_strength * 2:]

            # Detect trend
            trend = self._detect_trend(recent_highs, recent_lows)

            current_close = closes[i]
            prev_close = closes[i - 1] if i > 0 else current_close

            # Uptrend -> Bearish CHoCH (NEW break below swing low)
            if trend == 'uptrend' and recent_lows:
                for swing in reversed(recent_lows[-self.max_levels:]):
                    swing_idx = swing['index']
                    swing_val = swing['value']

                    if current_close < swing_val and prev_close >= swing_val:
                        if swing_idx not in broken_lows:
                            choch_signal[i] = -1  # Bearish CHoCH
                            broken_lows.add(swing_idx)
                            break
                    elif current_close < swing_val:
                        broken_lows.add(swing_idx)

            # Downtrend -> Bullish CHoCH (NEW break above swing high)
            elif trend == 'downtrend' and recent_highs:
                for swing in reversed(recent_highs[-self.max_levels:]):
                    swing_idx = swing['index']
                    swing_val = swing['value']

                    if current_close > swing_val and prev_close <= swing_val:
                        if swing_idx not in broken_highs:
                            choch_signal[i] = 1  # Bullish CHoCH
                            broken_highs.add(swing_idx)
                            break
                    elif current_close > swing_val:
                        broken_highs.add(swing_idx)

        return choch_signal

    # =========================================================================
    # INTERNAL: FVG DETECTION
    # =========================================================================

    def _detect_fvg_batch(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> np.ndarray:
        """
        Fair Value Gap batch hesaplama

        FVG: 3 mum arasındaki boşluk
        - Bullish FVG: Mum 1'in high'ı < Mum 3'ün low'u
        - Bearish FVG: Mum 1'in low'u > Mum 3'ün high'ı

        Returns:
            np.ndarray: FVG count (positive=bullish dominance, negative=bearish)
        """
        n = len(closes)
        fvg_signal = np.zeros(n, dtype=int)

        # Track active FVGs
        active_fvgs = []

        for i in range(2, n):
            current_price = closes[i]

            # Detect new FVGs
            candle1_high = highs[i - 2]
            candle1_low = lows[i - 2]
            candle3_high = highs[i]
            candle3_low = lows[i]

            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if candle3_low > candle1_high:
                gap_size = candle3_low - candle1_high
                gap_pct = (gap_size / candle1_high) * 100 if candle1_high > 0 else 0

                if gap_pct >= self.fvg_min_size_pct:
                    active_fvgs.append({
                        'type': 'bullish',
                        'top': candle3_low,
                        'bottom': candle1_high,
                        'created_at': i,
                        'filled': False
                    })

            # Bearish FVG: Gap between candle 1 low and candle 3 high
            if candle1_low > candle3_high:
                gap_size = candle1_low - candle3_high
                gap_pct = (gap_size / candle3_high) * 100 if candle3_high > 0 else 0

                if gap_pct >= self.fvg_min_size_pct:
                    active_fvgs.append({
                        'type': 'bearish',
                        'top': candle1_low,
                        'bottom': candle3_high,
                        'created_at': i,
                        'filled': False
                    })

            # Update FVG status (check if filled or too old)
            updated_fvgs = []
            for fvg in active_fvgs:
                age = i - fvg['created_at']

                # Check if filled
                if fvg['type'] == 'bullish':
                    if current_price <= fvg['bottom']:
                        fvg['filled'] = True
                else:
                    if current_price >= fvg['top']:
                        fvg['filled'] = True

                # Keep if not filled and not too old
                if not fvg['filled'] and age < self.fvg_max_age:
                    updated_fvgs.append(fvg)

            active_fvgs = updated_fvgs

            # Count active FVGs
            bullish_count = sum(1 for f in active_fvgs if f['type'] == 'bullish')
            bearish_count = sum(1 for f in active_fvgs if f['type'] == 'bearish')

            fvg_signal[i] = bullish_count - bearish_count

        return fvg_signal

    # =========================================================================
    # INTERNAL: ORDER BLOCK DETECTION
    # =========================================================================

    def _detect_orderblocks_batch(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Order Block batch hesaplama

        Returns:
            Tuple: (ob_bullish_top, ob_bullish_bottom, ob_bearish_top, ob_bearish_bottom)
        """
        n = len(closes)
        ob_bullish_top = np.full(n, np.nan)
        ob_bullish_bottom = np.full(n, np.nan)
        ob_bearish_top = np.full(n, np.nan)
        ob_bearish_bottom = np.full(n, np.nan)

        if not self.ob_enabled:
            return ob_bullish_top, ob_bullish_bottom, ob_bearish_top, ob_bearish_bottom

        # Track active order blocks
        bullish_obs = []
        bearish_obs = []

        for i in range(1, n):
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Calculate candle change %
            if opens[i] != 0:
                change_pct = ((closes[i] - opens[i]) / opens[i]) * 100
            else:
                change_pct = 0

            # Detect strong bullish move -> find previous bearish candle as OB
            if change_pct >= self.ob_strength_threshold:
                # Look back for the last bearish candle
                for j in range(i - 1, max(0, i - 10), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        bullish_obs.append({
                            'top': highs[j],
                            'bottom': lows[j],
                            'created_at': i,
                            'strength': change_pct
                        })
                        break

            # Detect strong bearish move -> find previous bullish candle as OB
            elif change_pct <= -self.ob_strength_threshold:
                for j in range(i - 1, max(0, i - 10), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        bearish_obs.append({
                            'top': highs[j],
                            'bottom': lows[j],
                            'created_at': i,
                            'strength': abs(change_pct)
                        })
                        break

            # Update OB status (remove if broken)
            bullish_obs = [
                ob for ob in bullish_obs
                if current_close > ob['bottom']  # Not broken below
            ][-self.ob_max_blocks:]

            bearish_obs = [
                ob for ob in bearish_obs
                if current_close < ob['top']  # Not broken above
            ][-self.ob_max_blocks:]

            # Set current values (most recent OB)
            if bullish_obs:
                ob_bullish_top[i] = bullish_obs[-1]['top']
                ob_bullish_bottom[i] = bullish_obs[-1]['bottom']

            if bearish_obs:
                ob_bearish_top[i] = bearish_obs[-1]['top']
                ob_bearish_bottom[i] = bearish_obs[-1]['bottom']

        return ob_bullish_top, ob_bullish_bottom, ob_bearish_top, ob_bearish_bottom

    # =========================================================================
    # INTERNAL: SMC SIGNAL GENERATION
    # =========================================================================

    def _generate_smc_signal(
        self,
        bos: np.ndarray,
        choch: np.ndarray,
        fvg: np.ndarray
    ) -> np.ndarray:
        """
        Kombine SMC sinyal üret

        Signal Logic:
        - Bullish BOS veya Bullish CHoCH + FVG >= 0 -> 1 (bullish)
        - Bearish BOS veya Bearish CHoCH + FVG <= 0 -> -1 (bearish)
        - Aksi halde -> 0 (neutral)

        Returns:
            np.ndarray: SMC sinyali
        """
        n = len(bos)
        smc_signal = np.zeros(n, dtype=int)

        for i in range(n):
            # Bullish signal
            if (bos[i] == 1 or choch[i] == 1) and fvg[i] >= 0:
                smc_signal[i] = 1
            # Bearish signal
            elif (bos[i] == -1 or choch[i] == -1) and fvg[i] <= 0:
                smc_signal[i] = -1

        return smc_signal

    # =========================================================================
    # PUBLIC: CALCULATE (Real-time)
    # =========================================================================

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Real-time hesaplama (son bar için)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: SMC analizi
        """
        # Batch hesapla ve son değerleri döndür
        batch_result = self.calculate_batch(data)

        timestamp = int(data.iloc[-1].get('timestamp', 0))

        # Son değerleri al
        last_idx = len(batch_result) - 1

        smc_signal = batch_result['smc_signal'].iloc[last_idx]
        bos = batch_result['bos'].iloc[last_idx]
        choch = batch_result['choch'].iloc[last_idx]
        fvg = batch_result['fvg'].iloc[last_idx]
        swing_high = batch_result['swing_high'].iloc[last_idx]
        swing_low = batch_result['swing_low'].iloc[last_idx]

        # Signal type
        if smc_signal == 1:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif smc_signal == -1:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Strength calculation
        strength = 0.0
        if bos != 0:
            strength += 40
        if choch != 0:
            strength += 30
        if abs(fvg) > 0:
            strength += min(abs(fvg) * 10, 30)

        return IndicatorResult(
            value={
                'smc_signal': int(smc_signal),
                'bos': int(bos),
                'choch': int(choch),
                'fvg': int(fvg),
                'swing_high': float(swing_high) if not np.isnan(swing_high) else None,
                'swing_low': float(swing_low) if not np.isnan(swing_low) else None
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(strength, 100),
            metadata={
                'left_bars': self.left_bars,
                'right_bars': self.right_bars,
                'ob_enabled': self.ob_enabled
            }
        )

    # =========================================================================
    # PUBLIC: CALCULATE_BATCH (Backtest)
    # =========================================================================

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için optimize edilmiş)

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Tüm SMC değerleri
        """
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        n = len(data)

        # 1. Swing Points
        swing_highs, swing_lows = self._detect_swing_points(highs, lows)

        # 2. BOS
        bos = self._detect_bos_batch(closes, swing_highs, swing_lows)

        # 3. CHoCH
        choch = self._detect_choch_batch(closes, swing_highs, swing_lows)

        # 4. FVG
        fvg = self._detect_fvg_batch(highs, lows, closes)

        # 5. Order Blocks
        ob_bull_top, ob_bull_bot, ob_bear_top, ob_bear_bot = \
            self._detect_orderblocks_batch(opens, highs, lows, closes)

        # 6. SMC Signal
        smc_signal = self._generate_smc_signal(bos, choch, fvg)

        # 7. Current swing levels (son aktif swing high/low)
        swing_high_values = np.full(n, np.nan)
        swing_low_values = np.full(n, np.nan)

        swing_high_dict = {s['index']: s['value'] for s in swing_highs}
        swing_low_dict = {s['index']: s['value'] for s in swing_lows}

        current_high = np.nan
        current_low = np.nan

        for i in range(n):
            if i in swing_high_dict:
                current_high = swing_high_dict[i]
            if i in swing_low_dict:
                current_low = swing_low_dict[i]

            swing_high_values[i] = current_high
            swing_low_values[i] = current_low

        # Result DataFrame
        result = pd.DataFrame({
            'smc_signal': smc_signal,
            'bos': bos,
            'choch': choch,
            'fvg': fvg,
            'swing_high': swing_high_values,
            'swing_low': swing_low_values,
            'ob_bullish_top': ob_bull_top,
            'ob_bullish_bottom': ob_bull_bot,
            'ob_bearish_top': ob_bear_top,
            'ob_bearish_bottom': ob_bear_bot
        }, index=data.index)

        return result

    # =========================================================================
    # PUBLIC: WARMUP_BUFFER
    # =========================================================================

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup SMC buffers with historical data

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier (unused)
        """
        max_len = self.get_required_periods() + 50

        self._open_buffer = deque(maxlen=max_len)
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Fill buffers with historical data
        for _, row in data.tail(max_len).iterrows():
            self._open_buffer.append(row['open'])
            self._high_buffer.append(row['high'])
            self._low_buffer.append(row['low'])
            self._close_buffer.append(row['close'])

    # =========================================================================
    # PUBLIC: UPDATE (Real-time incremental)
    # =========================================================================

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier

        Returns:
            IndicatorResult: Güncel SMC değeri
        """
        # Initialize buffers if needed
        if not self._buffers_init:
            max_len = self.get_required_periods() + 50
            self._open_buffer = deque(maxlen=max_len)
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Parse candle data
        if isinstance(candle, dict):
            open_val = candle.get('open', candle.get('close', 0))
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        # Add to buffers
        self._open_buffer.append(open_val)
        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)

        # Check if enough data
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value={
                    'smc_signal': 0,
                    'bos': 0,
                    'choch': 0,
                    'fvg': 0,
                    'swing_high': None,
                    'swing_low': None
                },
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={
                    'warmup': True,
                    'required': self.get_required_periods(),
                    'current': len(self._close_buffer)
                }
            )

        # Create DataFrame from buffers
        buffer_data = pd.DataFrame({
            'open': list(self._open_buffer),
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer)
        })

        # Calculate using batch method
        return self.calculate(buffer_data)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'max_levels': 5,
            'trend_strength': 2,
            'fvg_min_size_pct': 0.1,
            'fvg_max_age': 50,
            'ob_strength_threshold': 1.0,
            'ob_max_blocks': 3,
            'ob_enabled': True
        }

    def _requires_volume(self) -> bool:
        """SMC volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SMC']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """SMC indicator testi"""

    print("\n" + "="*60)
    print("SMC INDICATOR TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    n = 200

    # Trend simülasyonu
    base_price = 100
    prices = [base_price]

    for i in range(n - 1):
        if i < 50:
            trend = 0.3 + np.random.randn() * 0.5  # Uptrend
        elif i < 100:
            trend = -0.2 + np.random.randn() * 0.5  # Downtrend
        elif i < 150:
            trend = 0.4 + np.random.randn() * 0.5  # Strong uptrend
        else:
            trend = -0.3 + np.random.randn() * 0.5  # Downtrend

        prices.append(prices[-1] + trend)

    prices = np.array(prices)
    highs = prices + np.abs(np.random.randn(n)) * 0.5
    lows = prices - np.abs(np.random.randn(n)) * 0.5
    opens = prices + np.random.randn(n) * 0.2

    data = pd.DataFrame({
        'timestamp': [1700000000000 + i * 300000 for i in range(n)],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")

    # Test 1: Indicator oluşturma
    print("\n2. Indicator oluşturma testi...")
    smc = SMC()
    print(f"   [OK] {smc.name} oluşturuldu")
    print(f"   [OK] Kategori: {smc.category.value}")
    print(f"   [OK] Gerekli periyot: {smc.get_required_periods()}")

    # Test 2: Batch hesaplama
    print("\n3. Batch hesaplama testi...")
    result_df = smc.calculate_batch(data)
    print(f"   [OK] Result shape: {result_df.shape}")
    print(f"   [OK] Columns: {list(result_df.columns)}")

    # BOS/CHoCH sayıları
    bullish_bos = (result_df['bos'] == 1).sum()
    bearish_bos = (result_df['bos'] == -1).sum()
    bullish_choch = (result_df['choch'] == 1).sum()
    bearish_choch = (result_df['choch'] == -1).sum()

    print(f"   [OK] Bullish BOS: {bullish_bos}")
    print(f"   [OK] Bearish BOS: {bearish_bos}")
    print(f"   [OK] Bullish CHoCH: {bullish_choch}")
    print(f"   [OK] Bearish CHoCH: {bearish_choch}")

    # Test 3: Real-time hesaplama
    print("\n4. Real-time hesaplama testi...")
    result = smc.calculate(data)
    print(f"   [OK] SMC Signal: {result.value['smc_signal']}")
    print(f"   [OK] BOS: {result.value['bos']}")
    print(f"   [OK] CHoCH: {result.value['choch']}")
    print(f"   [OK] FVG: {result.value['fvg']}")
    print(f"   [OK] Swing High: {result.value['swing_high']}")
    print(f"   [OK] Swing Low: {result.value['swing_low']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Strength: {result.strength}")

    # Test 4: Update (incremental)
    print("\n5. Incremental update testi...")
    smc2 = SMC()
    smc2.warmup_buffer(data.iloc[:-10])

    for i in range(-10, 0):
        row = data.iloc[i]
        candle = {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        result = smc2.update(candle)

    print(f"   [OK] Son update - SMC Signal: {result.value['smc_signal']}")
    print(f"   [OK] Son update - BOS: {result.value['bos']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = smc.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
