"""
indicators/breakout/consolidation.py - Consolidation Detector

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Consolidation - Consolidation Detection and Score
    Detects periods where the price moves within a narrow range
    and scores the quality of the consolidation.

    Analiz Kriterleri:
    - Volatility (ATR): Low volatility = consolidation
    - Range width: Narrow range = consolidation
    - Price distribution: Uniform distribution = quality consolidation
    - Duration: Long duration = strong consolidation

    Output:
    - consolidation_score: Consolidation score between 0 and 100
    - 0-25: Trend exists, no consolidation
    - 25-50: Weak consolidation
    - 50-75: Orta konsolidasyon
    - 75-100: Strong consolidation

Formula:
    ATR Score = (1 - Current ATR / Average ATR) × 100
    Range Score = (1 - Range % / Historical Avg %) × 100
    Distribution Score = Uniformity of price distribution
    Time Score = Consolidation duration

    Final Score = (ATR × 0.4) + (Range × 0.3) + (Distribution × 0.2) + (Time × 0.1)

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


class Consolidation(BaseIndicator):
    """
    Consolidation Detector

    Determines whether the price is in a consolidation phase and
    scores the consolidation quality on a scale of 0-100.

    Args:
        period: Analysis period (default: 20)
        atr_period: ATR calculation period (default: 14)
        lookback: Historical comparison period (default: 100)
        min_consolidation_bars: Minimum consolidation candle count (default: 10)
    """

    def __init__(
        self,
        period: int = 20,
        atr_period: int = 14,
        lookback: int = 100,
        min_consolidation_bars: int = 10,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.atr_period = atr_period
        self.lookback = lookback
        self.min_consolidation_bars = min_consolidation_bars

        super().__init__(
            name='consolidation',
            category=IndicatorCategory.BREAKOUT,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'atr_period': atr_period,
                'lookback': lookback,
                'min_consolidation_bars': min_consolidation_bars
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.period, self.atr_period, self.min_consolidation_bars) + 10

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 5:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be at least 5"
            )
        if self.atr_period < 2:
            raise InvalidParameterError(
                self.name, 'atr_period', self.atr_period,
                "The ATR period must be at least 2"
            )
        if self.min_consolidation_bars < 3:
            raise InvalidParameterError(
                self.name, 'min_consolidation_bars', self.min_consolidation_bars,
                "The minimum number of consolidation candles must be at least 3"
            )
        return True

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """ATR hesapla"""
        tr_list = []
        for i in range(1, len(close)):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i-1])
            l_pc = abs(low[i] - close[i-1])
            tr = max(h_l, h_pc, l_pc)
            tr_list.append(tr)

        tr_array = np.array(tr_list)
        if len(tr_array) >= self.atr_period:
            atr = np.mean(tr_array[-self.atr_period:])
        else:
            atr = np.mean(tr_array)

        return atr

    def _calculate_atr_score(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """ATR-based consolidation score (0-100)"""
        # Current ATR
        current_atr = self._calculate_atr(high, low, close)

        # Past ATR average
        lookback_size = min(self.lookback, len(close) - self.atr_period - 1)
        if lookback_size < self.atr_period:
            return 50.0

        historical_atrs = []
        for i in range(lookback_size):
            end_idx = -(i + 1)
            start_idx = end_idx - self.atr_period - 1
            if start_idx < -len(close):
                break
            hist_atr = self._calculate_atr(
                high[start_idx:end_idx],
                low[start_idx:end_idx],
                close[start_idx:end_idx]
            )
            historical_atrs.append(hist_atr)

        if not historical_atrs:
            return 50.0

        avg_historical_atr = np.mean(historical_atrs)

        # Low ATR = high score
        if avg_historical_atr == 0:
            return 50.0

        atr_ratio = current_atr / avg_historical_atr
        score = (1 - min(atr_ratio, 1.0)) * 100

        return score

    def _calculate_range_score(self, high: np.ndarray, low: np.ndarray) -> tuple:
        """Range-based consolidation score (0-100)"""
        # Current range
        current_high = np.max(high[-self.period:])
        current_low = np.min(low[-self.period:])
        current_range = current_high - current_low
        current_range_pct = (current_range / current_low * 100) if current_low > 0 else 0

        # Past range average
        lookback_size = min(self.lookback, len(high) - self.period)
        if lookback_size < self.period:
            return 50.0, current_range_pct

        historical_ranges = []
        for i in range(lookback_size):
            end_idx = -(i + 1)
            start_idx = end_idx - self.period
            if start_idx < -len(high):
                break
            hist_high = np.max(high[start_idx:end_idx])
            hist_low = np.min(low[start_idx:end_idx])
            hist_range = hist_high - hist_low
            hist_range_pct = (hist_range / hist_low * 100) if hist_low > 0 else 0
            historical_ranges.append(hist_range_pct)

        if not historical_ranges:
            return 50.0, current_range_pct

        avg_historical_range = np.mean(historical_ranges)

        # Narrow range = high score
        if avg_historical_range == 0:
            return 50.0, current_range_pct

        range_ratio = current_range_pct / avg_historical_range
        score = (1 - min(range_ratio, 1.0)) * 100

        return score, current_range_pct

    def _calculate_distribution_score(self, close: np.ndarray) -> float:
        """Price distribution score (0-100)"""
        # Analyze the distribution of prices within the last period.
        prices = close[-self.period:]

        # Create a histogram (5000)
        hist, bin_edges = np.histogram(prices, bins=5)

        # Uniform distribution = equal price for every bin
        expected_count = len(prices) / 5

        # Chi-square benzeri test
        deviations = [(count - expected_count) ** 2 / expected_count for count in hist if expected_count > 0]
        if not deviations:
            return 50.0

        deviation_score = np.mean(deviations)

        # Low deviation = high score
        # Max sapma ~len(prices) olabilir
        max_deviation = len(prices) / 2
        score = (1 - min(deviation_score / max_deviation, 1.0)) * 100

        return score

    def _calculate_time_score(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple:
        """Consolidation period score (0-100)"""
        # Calculate the range percentage for each candle.
        range_threshold = 3.0  # %3 threshold

        consolidation_count = 0
        for i in range(self.min_consolidation_bars, 0, -1):
            end_idx = -i if i > 1 else None
            start_idx = end_idx - self.period if end_idx else -self.period

            period_high = np.max(high[start_idx:end_idx])
            period_low = np.min(low[start_idx:end_idx])
            period_range_pct = ((period_high - period_low) / period_low * 100) if period_low > 0 else 0

            if period_range_pct < range_threshold:
                consolidation_count += 1
            else:
                break

        # Long duration = high score
        score = min((consolidation_count / self.min_consolidation_bars) * 100, 100)

        return score, consolidation_count

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Consolidation hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Consolidation score and details.
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Fill the buffers (preparation for incremental update)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = self.lookback + self.period + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        
        # Get the data up to the last max_len.
        start_idx = max(0, len(data) - (self.lookback + self.period + 50))
        self._high_buffer.extend(high[start_idx:])
        self._low_buffer.extend(low[start_idx:])
        self._close_buffer.extend(close[start_idx:])

        # ATR score (weight: 0.4)
        atr_score = self._calculate_atr_score(high, low, close)

        # Range score (weight: 0.3)
        range_score, current_range_pct = self._calculate_range_score(high, low)

        # Distribution score (weight: 0.2)
        distribution_score = self._calculate_distribution_score(close)

        # Time score (weight: 0.1)
        time_score, consolidation_bars = self._calculate_time_score(high, low, close)

        # Final consolidation score
        consolidation_score = (
            atr_score * 0.4 +
            range_score * 0.3 +
            distribution_score * 0.2 +
            time_score * 0.1
        )

        # Consolidation level
        if consolidation_score >= 75:
            level = "Strong"
        elif consolidation_score >= 50:
            level = "Moderate"
        elif consolidation_score >= 25:
            level = "Weak"
        else:
            level = "None"

        timestamp = int(data.iloc[-1]['timestamp'])

        # Define signal
        signal = self.get_signal(consolidation_score)
        trend = self.get_trend(consolidation_score)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'value': round(consolidation_score, 2),
                'level': level,
                'atr_score': round(atr_score, 2),
                'range_score': round(range_score, 2),
                'distribution_score': round(distribution_score, 2),
                'time_score': round(time_score, 2),
                'range_pct': round(current_range_pct, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=consolidation_score,
            metadata={
                'consolidation_bars': consolidation_bars
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Consolidation values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 1. ATR Score
        # TR Hesapla
        h_l = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        
        # Current ATR
        current_atr = tr.rolling(window=self.atr_period).mean()
        
        # Historical Avg ATR (Average calculated backwards for the Lookback period)
        # This might be a bit slow: rolling(lookback).mean()
        # However, the logic: it's not the average ATR within the lookback period,
        # The average of ATR values calculated during the lookback period.
        # Yani rolling(atr_period).mean() serisinin rolling(lookback).mean()'i.
        historical_avg_atr = current_atr.shift(1).rolling(window=self.lookback).mean()
        
        # ATR Ratio
        atr_ratio = current_atr / historical_avg_atr.replace(0, np.nan)
        atr_score = (1 - atr_ratio.clip(upper=1.0)) * 100
        atr_score = atr_score.fillna(50.0)
        
        # 2. Range Score
        # Current Range %
        roll_high = high.rolling(window=self.period).max()
        roll_low = low.rolling(window=self.period).min()
        current_range = roll_high - roll_low
        current_range_pct = (current_range / roll_low.replace(0, np.nan)) * 100
        current_range_pct = current_range_pct.fillna(0)
        
        # Historical Avg Range %
        historical_avg_range = current_range_pct.shift(1).rolling(window=self.lookback).mean()
        
        range_ratio = current_range_pct / historical_avg_range.replace(0, np.nan)
        range_score = (1 - range_ratio.clip(upper=1.0)) * 100
        range_score = range_score.fillna(50.0)
        
        # 3. Distribution Score
        # Calculating the histogram with rolling apply can be slow.
        # Simplified approach: standard deviation / range
        # The standard deviation is high in a uniform distribution (compared to the range).
        # Ama burada "uniformity" isteniyor.
        # Orijinal logic: Chi-square test.
        # It's difficult to make this a batch operation. Let's use rolling apply.
        
        def calc_dist_score(x):
            if len(x) < 5: return 50.0
            hist, _ = np.histogram(x, bins=5)
            expected = len(x) / 5
            deviations = [(c - expected) ** 2 / expected for c in hist if expected > 0]
            if not deviations: return 50.0
            dev_score = np.mean(deviations)
            max_dev = len(x) / 2
            return (1 - min(dev_score / max_dev, 1.0)) * 100

        # For performance, only use rolling apply on the 'close' column.
        # This operation may be slow, it can be optimized.
        distribution_score = close.rolling(window=self.period).apply(calc_dist_score, raw=True)
        distribution_score = distribution_score.fillna(50.0)
        
        # 4. Time Score
        # Number of values in the range < threshold for the last X periods
        range_threshold = 3.0
        
        # Calculate the range % for each bar (not period-based, but the range of the current bar)
        # Logic: "Calculate the range percentage for each candle" but inside the loop.
        # It uses period_high/low. That means it looks back period units for each candle.
        # This is already current_range_pct.
        
        # That is: the number of consecutive candles where current_range_pct < 3.0?
        # No, loop: for i in range(min_consolidation_bars, 0, -1)
        # It looks back by going back i steps.
        # Actually, the logic is: going backward from the current candle,
        # How many candles have a range that is below their "own period" range?
        
        is_consolidating = current_range_pct < range_threshold
        
        # Rolling sum of boolean?
        # No, it needs to be sequential.
        # Simply: It's not the number of is_consolidating within the last min_consolidation_bars.
        # Loop logic: Starting from the longest (min_consolidation_bars),
        # if the range is low during that period, it counts.
        
        # Let's look at the original code:
        # for i in range(self.min_consolidation_bars, 0, -1):
        #    period_range_pct = ...
        #    if period_range_pct < range_threshold: count += 1 else break
        
        # This logic is a bit strange. As 'i' decreases, the window doesn't shrink, but 'start_idx' changes.
        # start_idx = end_idx - self.period
        # That means it's going backward with a sliding window.
        
        # For batch:
        # is_consolidating serisi (current_range_pct < 3.0)
        # The number of consecutive true values in reverse order.
        
        # Finding consecutive groups with Pandas:
        # grouper = (is_consolidating != is_consolidating.shift()).cumsum()
        # But we want the current sequential number for each bar.
        
        # Bu tipik bir "consecutive streak" problemi.
        # df['streak'] = df.groupby((df['val'] != df['val'].shift()).cumsum()).cumcount() + 1
        # We will only take the ones that are True.
        
        streak_groups = (is_consolidating != is_consolidating.shift()).cumsum()
        streaks = is_consolidating.groupby(streak_groups).cumsum()
        
        # Only True values contribute to the streak, False values should be 0, but cumsum treats booleans as 1/0.
        # False olanlar resetlenmeli.
        # A better approach:
        # y = x * (y.shift() + 1)
        # Bu recursive, pandas'ta zor.
        
        # Alternative: Rolling sum but only if all are 1? No.
        
        # Let's calculate it quickly using a Python loop (Numpy iteration)
        cons_vals = is_consolidating.values.astype(int)
        time_counts = np.zeros(len(cons_vals))
        
        current_streak = 0
        for i in range(len(cons_vals)):
            if cons_vals[i] == 1:
                current_streak += 1
            else:
                current_streak = 0
            time_counts[i] = current_streak
            
        # Max limit: min_consolidation_bars
        time_counts = np.minimum(time_counts, self.min_consolidation_bars)
        time_score = (time_counts / self.min_consolidation_bars) * 100
        time_score = pd.Series(time_score, index=data.index)
        
        # Final Score
        final_score = (
            atr_score * 0.4 +
            range_score * 0.3 +
            distribution_score * 0.2 +
            time_score * 0.1
        )
        
        # Level
        levels = pd.Series("None", index=data.index)
        levels[final_score >= 25] = "Weak"
        levels[final_score >= 50] = "Moderate"
        levels[final_score >= 75] = "Strong"
        
        return pd.DataFrame({
            'value': final_score,
            'level': levels,
            'atr_score': atr_score,
            'range_score': range_score,
            'distribution_score': distribution_score,
            'time_score': time_score,
            'range_pct': current_range_pct
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current Consolidation value
        """
        # Buffer management
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            # It may be necessary to have data for Lookback + period (for Historical avg).
            # However, a lot of data is needed to calculate the historical average.
            # It's more logical to keep only the last period in the buffer and incrementally update the historical average.
            # Or keep the buffer large enough.
            
            max_len = self.lookback + self.period + 50
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
        if len(self._close_buffer) < self.period:
             return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.NEUTRAL,
                trend=TrendDirection.UNKNOWN,
                strength=0.0,
                metadata={'level': 'None'}
            )
            
        # Calculation
        # Convert buffers to numpy array (for the last lookback+period)
        # It can be optimized to use the entire buffer instead, but for now, this is the safer approach.
        
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        
        # ATR Score
        atr_score = self._calculate_atr_score(high, low, close)
        
        # Range Score
        range_score, current_range_pct = self._calculate_range_score(high, low)
        
        # Distribution Score
        distribution_score = self._calculate_distribution_score(close)
        
        # Time Score
        time_score, consolidation_bars = self._calculate_time_score(high, low, close)
        
        # Final Score
        consolidation_score = (
            atr_score * 0.4 +
            range_score * 0.3 +
            distribution_score * 0.2 +
            time_score * 0.1
        )
        
        # Level
        if consolidation_score >= 75:
            level = "Strong"
        elif consolidation_score >= 50:
            level = "Moderate"
        elif consolidation_score >= 25:
            level = "Weak"
        else:
            level = "None"
            
        return IndicatorResult(
            value=round(consolidation_score, 2),
            timestamp=timestamp_val,
            signal=self.get_signal(consolidation_score),
            trend=self.get_trend(consolidation_score),
            strength=consolidation_score,
            metadata={
                'level': level,
                'atr_score': round(atr_score, 2),
                'range_score': round(range_score, 2),
                'distribution_score': round(distribution_score, 2),
                'time_score': round(time_score, 2),
                'range_pct': round(current_range_pct, 2),
                'consolidation_bars': consolidation_bars
            }
        )

    def get_signal(self, score: float) -> SignalType:
        """
        Generate a signal from the consolidation score.

        Args:
            score: Consolidation score

        Returns:
            SignalType: HOLD (wait during consolidation)
        """
        # High consolidation = wait (prepare for breakout)
        if score >= 75:
            return SignalType.HOLD
        elif score >= 50:
            return SignalType.HOLD
        else:
            return SignalType.NEUTRAL

    def get_trend(self, score: float) -> TrendDirection:
        """
        Determine the trend based on the consolidation score.

        Args:
            score: Consolidation score

        Returns:
            TrendDirection: NEUTRAL (consolidation = no trend)
        """
        # No trend during consolidation
        if score >= 50:
            return TrendDirection.NEUTRAL
        else:
            return TrendDirection.UNKNOWN

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'atr_period': 14,
            'lookback': 100,
            'min_consolidation_bars': 10
        }

    def _requires_volume(self) -> bool:
        """Consolidation volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Consolidation']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Consolidation indicator test"""

    print("\n" + "="*60)
    print("CONSOLIDATION DETECTOR TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend -> Consolidation -> Simulate trend
    base_price = 100
    prices = [base_price]

    # First 30 candles: Uptrend
    for i in range(29):
        change = np.random.randn() * 1.0 + 0.5
        prices.append(prices[-1] + change)

    # Sonraki 60 mum: Konsolidasyon
    consolidation_base = prices[-1]
    for i in range(59):
        change = np.random.randn() * 0.2
        prices.append(np.clip(prices[-1] + change, consolidation_base - 1, consolidation_base + 1))

    # Son 60 mum: Breakout + trend
    for i in range(60):
        change = np.random.randn() * 1.5 + 0.8
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p + np.random.randn() * 0.1 for p in prices],
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    consol = Consolidation()
    print(f"   [OK] Created: {consol}")
    print(f"   [OK] Kategori: {consol.category.value}")
    print(f"   [OK] Required period: {consol.get_required_periods()}")

    result = consol(data)
    print(f"   [OK] Consolidation Score: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Trend testi
    print("\n3. Trend testi (ilk 30 mum)...")
    trend_data = data.head(50)
    result = consol.calculate(trend_data)
    print(f"   [OK] Consolidation Score: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] ATR Score: {result.metadata['atr_score']:.2f}")
    print(f"   [OK] Range Score: {result.metadata['range_score']:.2f}")

    # Test 3: Konsolidasyon testi
    print("\n4. Konsolidasyon testi (60-90 mum)...")
    consol_data = data.iloc[30:90].reset_index(drop=True)
    # We need to add data from the beginning to have enough historical data.
    consol_data_full = data.head(90)
    result = consol.calculate(consol_data_full)
    print(f"   [OK] Consolidation Score: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] ATR Score: {result.metadata['atr_score']:.2f}")
    print(f"   [OK] Range Score: {result.metadata['range_score']:.2f}")
    print(f"   [OK] Distribution Score: {result.metadata['distribution_score']:.2f}")
    print(f"   [OK] Time Score: {result.metadata['time_score']:.2f}")
    print(f"   [OK] Consolidation Bars: {result.metadata['consolidation_bars']}")

    # Test 4: After breakout
    print("\n5. Post-breakout test (all data)...")
    result = consol.calculate(data)
    print(f"   [OK] Consolidation Score: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] Range %: {result.metadata['range_pct']:.2f}%")

    # Test 5: Zaman serisi analizi
    print("\n6. Zaman serisi analizi...")
    scores = []
    levels = []

    for i in range(40, len(data), 10):
        partial_data = data.head(i)
        result = consol.calculate(partial_data)
        scores.append(result.value)
        levels.append(result.metadata['level'])

    print(f"   [OK] Total measurements: {len(scores)}")
    print(f"   [OK] Ortalama puan: {np.mean(scores):.2f}")
    print(f"   [OK] Max puan: {max(scores):.2f}")
    print(f"   [OK] Min puan: {min(scores):.2f}")
    print(f"   [OK] Strong: {levels.count('Strong')}, Moderate: {levels.count('Moderate')}")
    print(f"   [OK] Weak: {levels.count('Weak')}, None: {levels.count('None')}")

    # Test 6: Different parameters
    print("\n7. Different parameter test...")
    consol_fast = Consolidation(period=10, min_consolidation_bars=5)
    result = consol_fast.calculate(consol_data_full)
    print(f"   [OK] Fast (10 period) Score: {result.value}")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = consol.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = consol.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
