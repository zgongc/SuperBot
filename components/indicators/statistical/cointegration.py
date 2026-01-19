"""
indicators/statistical/cointegration.py - Cointegration

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Cointegration - Tests the long-term equilibrium of two time series.
    Outputs:
        - spread: The difference between the two assets.
        - zscore: The Z-Score of the spread.
        - is_cointegrated: Whether cointegration exists (boolean).

    It is of critical importance in pairs trading strategies.

Formula:
    1. Hedge Ratio (β) hesapla: Linear Regression
       Asset1 = β × Asset2 + ε

    2. Calculate the spread:
       Spread = Asset1 - (β x Asset2)

    3. Calculate the Z-Score of the Spread:
       Z-Score = (Spread - Mean(Spread)) / Std(Spread)

    4. Stationarity check using the Augmented Dickey-Fuller (ADF) test.

Usage:
    - Pairs trading
    - Statistical arbitrage
    - Mean reversion stratejileri

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scipy>=1.10.0
    - statsmodels>=0.14.0 (for the ADF test)
"""

import numpy as np
import pandas as pd
from scipy import stats
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)

# optional import for statsmodels
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class Cointegration(BaseIndicator):
    """
    Cointegration (Cointegration)

    Calculates the long-term equilibrium and spread between two assets.
    Used for pairs trading strategies.

    Args:
        period: Calculation period (default: 50)
        reference_data: Reference data to compare with (default: None)
        entry_threshold: Spread Z-Score entry threshold (default: 2.0)
        exit_threshold: Spread Z-Score exit threshold (default: 0.5)
        adf_significance: ADF test significance level (default: 0.05)
    """

    def __init__(
        self,
        period: int = 50,
        reference_data: pd.DataFrame = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        adf_significance: float = 0.05,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.reference_data = reference_data
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.adf_significance = adf_significance

        super().__init__(
            name='cointegration',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'period': period,
                'reference_data': reference_data,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'adf_significance': adf_significance
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 10:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period must be at least 10 (for cointegration)"
            )
        if self.entry_threshold <= self.exit_threshold:
            raise InvalidParameterError(
                self.name, 'thresholds',
                f"entry={self.entry_threshold}, exit={self.exit_threshold}",
                "Entry threshold must be greater than the exit threshold"
            )
        if not (0 < self.adf_significance < 1):
            raise InvalidParameterError(
                self.name, 'adf_significance', self.adf_significance,
                "ADF significance level must be between 0 and 1"
            )
        return True

    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Set the reference data (the other asset of the pair).

        Args:
            reference_data: Referans OHLCV DataFrame
        """
        self.reference_data = reference_data

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: spread, zscore, is_cointegrated
        """
        if self.reference_data is None:
            return pd.DataFrame(index=data.index, columns=['spread', 'zscore', 'is_cointegrated'])
            
        # Veri hizalama
        # We must ensure that the indices match
        common_index = data.index.intersection(self.reference_data.index)
        if len(common_index) < self.period:
            return pd.DataFrame(index=data.index, columns=['spread', 'zscore', 'is_cointegrated'])
            
        s1 = data.loc[common_index, 'close']
        s2 = self.reference_data.loc[common_index, 'close']
        
        # Rolling Statistics
        roll = s1.rolling(window=self.period)
        mean_s1 = roll.mean()
        var_s1 = roll.var()
        
        roll2 = s2.rolling(window=self.period)
        mean_s2 = roll2.mean()
        var_s2 = roll2.var()
        
        cov_s1s2 = s1.rolling(window=self.period).cov(s2)
        
        # Beta (Hedge Ratio)
        beta = cov_s1s2 / var_s2
        
        # Spread Mean (using current beta for the whole window)
        # mean(e) = mean(s1 - beta*s2) = mean(s1) - beta * mean(s2)
        spread_mean = mean_s1 - beta * mean_s2
        
        # Spread Variance (using current beta for the whole window)
        # var(e) = var(s1 - beta*s2) = var(s1) + beta^2 * var(s2) - 2*beta*cov(s1,s2)
        spread_var = var_s1 + (beta ** 2) * var_s2 - 2 * beta * cov_s1s2
        spread_std = np.sqrt(spread_var)
        
        # Current Spread (Residual at t)
        spread = s1 - beta * s2
        
        # Z-Score
        # Prevent division by zero error
        spread_std = spread_std.replace(0, np.nan)
        zscore = (spread - spread_mean) / spread_std
        
        # Cointegration Check (Vectorized approximation)
        # Since the ADF test is very slow, we are using the volatility ratio in batch mode.
        # spread_volatility = spread_std / abs(spread_mean)
        # is_cointegrated = spread_volatility < 0.5
        
        # Prevent division by zero error
        spread_mean_abs = spread_mean.abs().replace(0, np.nan)
        spread_volatility = spread_std / spread_mean_abs
        is_cointegrated = spread_volatility < 0.5
        
        # Result DataFrame
        result = pd.DataFrame({
            'spread': spread,
            'zscore': zscore,
            'is_cointegrated': is_cointegrated
        }, index=common_index)
        
        # Reindex to the original index (missing values will remain NaN)
        return result.reindex(data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Current cointegration values
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        if self.reference_data is None:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Timestamp check
        timestamp = candle.get('timestamp')
        if timestamp is None:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Referans veriden ilgili mumu bul
        # Note: This is a simple lookup. Synchronization is important in real life.
        # self.reference_data's index must be a timestamp or datetime.
        # If the range is an index, we need to search in the timestamp column.
        
        ref_price = None
        
        # Try to find the line that matches the timestamp
        if isinstance(self.reference_data.index, pd.DatetimeIndex):
            # If the timestamp is in milliseconds, convert it to datetime if necessary.
            try:
                ts_dt = pd.to_datetime(timestamp, unit='ms')
                if ts_dt in self.reference_data.index:
                    ref_price = self.reference_data.loc[ts_dt, 'close']
            except:
                pass
        else:
            # 'timestamp' kolonu varsa
            if 'timestamp' in self.reference_data.columns:
                matches = self.reference_data[self.reference_data['timestamp'] == timestamp]
                if not matches.empty:
                    ref_price = matches.iloc[0]['close']
        
        if ref_price is None:
            # If there is no matching reference data, the calculation cannot be performed.
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Buffer management (BaseIndicator does not have a standard buffer, we need to manage it ourselves or call calculate)
        # Buffer management for cointegration is complex because there are two series.
        # The easiest way: Get data up to the last period and call calculate (Not incremental, but real-time)
        # However, if we want to do it incrementally for performance:
        
        # For now, the safe way: RealtimeCalculator already holds a buffer (data['close']).
        # However, it only keeps the main entity.
        # We also need the latest period data for our reference asset.
        
        # Therefore, it's most logical to simply call the `calculate` method with the latest data here.
        # However, the `calculate` method seems to be taking all the data.
        # BaseIndicator.update() by default calls calculate() already.
        # Ama biz optimize etmek istiyoruz.
        
        # For optimized update:
        # 1. The last period's close1 and close2 values must be stored in the class state.
        # 2. The newly arrived close1 and the found close2 should be added.
        # 3. Calculation should be performed.
        
        # State initialize (ilk seferde)
        if not hasattr(self, '_close1_buffer'):
            from collections import deque
            self._close1_buffer = deque(maxlen=self.period)
            self._close2_buffer = deque(maxlen=self.period)
            
            # If there is historical data, fill it (warmup)
            # This part is a bit tricky, because update() is called individually.
            # Warmup should be done externally or during the initial update.
            pass

        close1 = candle['close']
        close2 = ref_price
        
        self._close1_buffer.append(close1)
        self._close2_buffer.append(close2)
        
        if len(self._close1_buffer) < self.period:
             return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
             
        # Calculation
        c1 = np.array(self._close1_buffer)
        c2 = np.array(self._close2_buffer)
        
        # Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(c2, c1)
        hedge_ratio = slope
        
        # Spread
        spread_series = c1 - (hedge_ratio * c2)
        current_spread = spread_series[-1]
        
        # Z-Score
        spread_mean = np.mean(spread_series)
        spread_std = np.std(spread_series, ddof=1)
        
        if spread_std > 0:
            spread_zscore = (current_spread - spread_mean) / spread_std
        else:
            spread_zscore = 0.0
            
        # Cointegration Check (ADF or Volatility)
        # We can run ADF in real-time (since it's a one-time operation)
        is_cointegrated = False
        adf_pvalue = 1.0
        
        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(spread_series, autolag='AIC')
                adf_pvalue = adf_result[1]
                is_cointegrated = adf_pvalue < self.adf_significance
            except:
                spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
                is_cointegrated = spread_volatility < 0.5
        else:
            spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
            is_cointegrated = spread_volatility < 0.5
            
        # Result
        strength = min(abs(spread_zscore) * 50, 100)
        
        return IndicatorResult(
            value={
                'spread': round(current_spread, 4),
                'zscore': round(spread_zscore, 4),
                'is_cointegrated': is_cointegrated
            },
            timestamp=int(timestamp) if timestamp else 0,
            signal=self.get_signal(spread_zscore, is_cointegrated),
            trend=self.get_trend(spread_zscore),
            strength=strength,
            metadata={
                'period': self.period,
                'hedge_ratio': round(hedge_ratio, 4),
                'spread_mean': round(spread_mean, 4),
                'spread_std': round(spread_std, 4),
                'adf_pvalue': round(adf_pvalue, 6),
                'correlation': round(r_value, 4),
                'asset1_price': round(close1, 2),
                'asset2_price': round(close2, 2),
                'has_statsmodels': HAS_STATSMODELS
            }
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Cointegration hesapla

        Args:
            data: OHLCV DataFrame (first asset)

        Returns:
            IndicatorResult: Results of the cointegration analysis.
        """
        if self.reference_data is None:
            # If reference data is not available, return dummy values.
            timestamp = int(data.iloc[-1]['timestamp'])
            return IndicatorResult(
                value={
                    'spread': 0.0,
                    'zscore': 0.0,
                    'is_cointegrated': False
                },
                timestamp=timestamp,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={
                    'error': 'No reference data provided',
                    'hedge_ratio': 0.0,
                    'adf_pvalue': 1.0
                }
            )

        # Data preparation
        close1 = data['close'].values[-self.period:]
        close2 = self.reference_data['close'].values[-self.period:]

        # Equalize data lengths
        min_len = min(len(close1), len(close2))
        if min_len < 10:
            timestamp = int(data.iloc[-1]['timestamp'])
            return IndicatorResult(
                value={
                    'spread': 0.0,
                    'zscore': 0.0,
                    'is_cointegrated': False
                },
                timestamp=timestamp,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={
                    'error': 'Insufficient data',
                    'hedge_ratio': 0.0,
                    'adf_pvalue': 1.0
                }
            )

        close1 = close1[-min_len:]
        close2 = close2[-min_len:]
        
        # Fill the buffers (preparation for incremental update)
        if not hasattr(self, '_close1_buffer'):
            from collections import deque
            self._close1_buffer = deque(maxlen=self.period)
            self._close2_buffer = deque(maxlen=self.period)
            
        # Add data to the buffer up to the last period.
        # Note: calculate() usually takes the entire history, but we want to save the latest status as the state.
        # If calculate() is called for backtesting, this state represents the last moment.
        buffer_data1 = close1[-self.period:] if len(close1) >= self.period else close1
        buffer_data2 = close2[-self.period:] if len(close2) >= self.period else close2
        
        self._close1_buffer.clear()
        self._close2_buffer.clear()
        self._close1_buffer.extend(buffer_data1)
        self._close2_buffer.extend(buffer_data2)

        # 1. Hedge Ratio hesapla (Linear Regression)
        # close1 = beta * close2 + alpha
        slope, intercept, r_value, p_value, std_err = stats.linregress(close2, close1)
        hedge_ratio = slope

        # 2. Calculate spread
        spread_series = close1 - (hedge_ratio * close2)
        current_spread = spread_series[-1]

        # Calculate the Z-Score of the 3rd spread.
        spread_mean = np.mean(spread_series)
        spread_std = np.std(spread_series, ddof=1)

        if spread_std > 0:
            spread_zscore = (current_spread - spread_mean) / spread_std
        else:
            spread_zscore = 0.0

        # 4. Stationarity check with the ADF test (cointegration test)
        is_cointegrated = False
        adf_pvalue = 1.0

        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(spread_series, autolag='AIC')
                adf_statistic = adf_result[0]
                adf_pvalue = adf_result[1]

                # If p-value < significance level, then the spread is stationary (cointegrated)
                is_cointegrated = adf_pvalue < self.adf_significance
            except Exception:
                # If the ADF test fails, perform a simple volatility check.
                spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
                is_cointegrated = spread_volatility < 0.5
        else:
            # If statsmodels is not available, perform a simple volatility check.
            spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
            is_cointegrated = spread_volatility < 0.5

        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal strength: Absolute value of the Z-score
        strength = min(abs(spread_zscore) * 50, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'spread': round(current_spread, 4),
                'zscore': round(spread_zscore, 4),
                'is_cointegrated': is_cointegrated
            },
            timestamp=timestamp,
            signal=self.get_signal(spread_zscore, is_cointegrated),
            trend=self.get_trend(spread_zscore),
            strength=strength,
            metadata={
                'period': self.period,
                'hedge_ratio': round(hedge_ratio, 4),
                'spread_mean': round(spread_mean, 4),
                'spread_std': round(spread_std, 4),
                'adf_pvalue': round(adf_pvalue, 6),
                'correlation': round(r_value, 4),
                'asset1_price': round(close1[-1], 2),
                'asset2_price': round(close2[-1], 2),
                'has_statsmodels': HAS_STATSMODELS
            }
        )

    def get_signal(self, zscore: float, is_cointegrated: bool) -> SignalType:
        """
        Generate a signal from the Spread Z-Score.

        Args:
            zscore: The Z-score of the spread.
            is_cointegrated: Does cointegration exist?

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # Do not send a signal if there is no co-integration
        if not is_cointegrated:
            return SignalType.HOLD

        # Spread is very low (asset1 is cheap, asset2 is expensive)
        # Asset1 al, Asset2 sat
        if zscore <= -self.entry_threshold:
            return SignalType.BUY

        # Spread is very high (asset1 is expensive, asset2 is cheap)
        # Asset1 sat, Asset2 al
        elif zscore >= self.entry_threshold:
            return SignalType.SELL

        # Returning to normal, close the position.
        elif abs(zscore) <= self.exit_threshold:
            return SignalType.HOLD

        return SignalType.HOLD

    def get_trend(self, zscore: float) -> TrendDirection:
        """
        Determine the trend from the Spread Z-Score.

        Args:
            zscore: The Z-score of the spread.

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if zscore > 1:
            return TrendDirection.UP  # Spread is widening
        elif zscore < -1:
            return TrendDirection.DOWN  # Spread is narrowing
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 50,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'adf_significance': 0.05
        }

    def _requires_volume(self) -> bool:
        """Cointegration volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Cointegration']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Cointegration indicator test"""

    print("\n" + "="*60)
    print("COINTEGRATION TEST")
    print("="*60 + "\n")

    # Test 1: Merged assets
    print("1. Paired asset integration test...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Entity 1 - common trend + individual noise
    common_trend = np.cumsum(np.random.randn(100) * 0.1)
    individual_noise1 = np.random.randn(100) * 0.3

    prices1 = 100 + common_trend + individual_noise1

    data1 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices1,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices1],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices1],
        'close': prices1,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices1]
    })

    # Entity 2 - same common trend + different individual noise (co-integrated)
    individual_noise2 = np.random.randn(100) * 0.3
    prices2 = 110 + common_trend * 1.2 + individual_noise2

    data2 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices2,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices2],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices2],
        'close': prices2,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices2]
    })

    print(f"   [OK] {len(data1)} candles created")
    print(f"   [OK] Asset 1 price: {prices1[-1]:.2f}")
    print(f"   [OK] Asset 2 price: {prices2[-1]:.2f}")

    coint = Cointegration(period=50, reference_data=data2)
    print(f"   [OK] Created: {coint}")
    print(f"   [OK] Kategori: {coint.category.value}")
    print(f"   [OK] statsmodels var: {HAS_STATSMODELS}")

    result = coint(data1)
    print(f"   [OK] Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Cointegration: {result.value['is_cointegrated']}")
    print(f"   [OK] Hedge Ratio: {result.metadata['hedge_ratio']}")
    print(f"   [OK] Correlation: {result.metadata['correlation']}")
    print(f"   [OK] ADF P-value: {result.metadata['adf_pvalue']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Non-integrated assets
    print("\n2. Uncorrelated asset pair test...")
    np.random.seed(99)

    # Entity 3 - completely independent random walk
    prices3 = [100]
    for _ in range(99):
        prices3.append(prices3[-1] + np.random.randn() * 2)

    data3 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices3,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices3],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices3],
        'close': prices3,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices3]
    })

    # Entity 4 - another independent random walk
    prices4 = [110]
    for _ in range(99):
        prices4.append(prices4[-1] + np.random.randn() * 2)

    data4 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices4,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices4],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices4],
        'close': prices4,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices4]
    })

    coint.set_reference_data(data4)
    result = coint.calculate(data3)
    print(f"   [OK] Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Cointegration: {result.value['is_cointegrated']}")
    print(f"   [OK] Correlation: {result.metadata['correlation']}")
    print(f"   [OK] ADF P-value: {result.metadata['adf_pvalue']}")

    # Test 3: Spread expansion - trading signal
    print("\n3. Trading signal test (spread expansion)...")
    # Artificially expand the spread.
    prices1_wide = prices1.copy()
    prices1_wide[-1] += 5  # Increase the last price

    data1_wide = data1.copy()
    data1_wide.loc[data1_wide.index[-1], 'close'] = prices1_wide[-1]

    coint.set_reference_data(data2)
    result = coint.calculate(data1_wide)
    print(f"   [OK] Expanded Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 4: Spread reduction
    print("\n4. Spread shrinkage test...")
    prices1_narrow = prices1.copy()
    prices1_narrow[-1] -= 3  # Decrease the last price

    data1_narrow = data1.copy()
    data1_narrow.loc[data1_narrow.index[-1], 'close'] = prices1_narrow[-1]

    result = coint.calculate(data1_narrow)
    print(f"   [OK] Reduced Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 5: Different periods
    print("\n5. Different period test...")
    for period in [30, 50, 70]:
        if len(data1) >= period:
            coint_test = Cointegration(period=period, reference_data=data2)
            result = coint_test.calculate(data1)
            print(f"   [OK] Period({period}): Z-Score={result.value['zscore']:7.4f} | "
                  f"Coint={result.value['is_cointegrated']} | "
                  f"Hedge Ratio={result.metadata['hedge_ratio']:.4f}")

    # Test 6: Spread time series
    print("\n6. Spread time series (last 10 candles)...")
    coint_ts = Cointegration(period=50, reference_data=data2)
    for i in range(-10, 0):
        test_data1 = data1.iloc[:len(data1)+i]
        test_data2 = data2.iloc[:len(data2)+i]
        if len(test_data1) >= coint_ts.period:
            coint_ts.set_reference_data(test_data2)
            result = coint_ts.calculate(test_data1)
            print(f"   [OK] Mum {i:3d}: Spread={result.value['spread']:8.4f} | "
                  f"Z-Score={result.value['zscore']:7.4f} | "
                  f"Signal={result.signal.value}")

    # Test 7: Custom thresholds
    print("\n7. Special threshold test...")
    coint_custom = Cointegration(period=50, reference_data=data2,
                                  entry_threshold=3.0, exit_threshold=1.0)
    result = coint_custom.calculate(data1)
    print(f"   [OK] Entry threshold: {coint_custom.entry_threshold}")
    print(f"   [OK] Exit threshold: {coint_custom.exit_threshold}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 8: Statistics
    print("\n8. Statistical test...")
    stats_data = coint.statistics
    print(f"   [OK] Calculation count: {stats_data['calculation_count']}")
    print(f"   [OK] Error count: {stats_data['error_count']}")

    # Test 9: Metadata
    print("\n9. Metadata testi...")
    metadata = coint.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    # Test 10: Referans veri olmadan
    print("\n10. Reference data missing test...")
    coint_no_ref = Cointegration(period=50)
    result = coint_no_ref.calculate(data1)
    print(f"   [OK] Spread: {result.value['spread']}")
    print(f"   [OK] Error message: {result.metadata.get('error', 'N/A')}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
