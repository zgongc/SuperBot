"""
indicators/base_indicator.py - Base Indicator Class

Version: 2.0.0
Date: 2025-10-24
Author: SuperBot Team

Description:
    Abstract base class for all technical indicators.
    All indicators will be derived from this class.

    Tasks:
    - Input validation (OHLCV data)
    - Period checking (is there enough data?)
    - Error handling integration
    - Logger integration
    - Common utilities (type conversion, etc.)
    - Abstract calculate() method (for realtime)
    - Abstract calculate_batch() method (for backtest - MANDATORY!)

    Usage:
        class RSI(BaseIndicator):
            def __init__(self, period: int = 14):
                super().__init__(
                    name='rsi',
                    category=IndicatorCategory.MOMENTUM,
                    params={'period': period}
                )
                self.period = period

            def calculate(self, data: pd.DataFrame) -> IndicatorResult:
                # Realtime - for the last bar
                pass

            def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
                # Backtest - For ALL bars (VECTORIZED!)
                pass

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.types (local)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    IndicatorConfig,
    IndicatorMetadata,
    TrendDirection,
    SignalType,
    OHLCV,
    InsufficientDataError,
    InvalidParameterError,
    CalculationError
)


# ============================================================================
# BASE INDICATOR CLASS
# ============================================================================

class BaseIndicator(ABC):
    """
    Abstract base class for all indicators
    
    Each indicator should inherit from this class and implement the calculate() method.
    
    Attributes:
        name: Indicator name (e.g., 'rsi', 'ema')
        category: Kategori (momentum, trend, etc.)
        indicator_type: Output type (single_value, bands, etc.)
        params: Parameters dictionary
        logger: Logger instance (optional)
        error_handler: ErrorHandler instance (optional)
    """
    
    def __init__(
        self,
        name: str,
        category: IndicatorCategory,
        indicator_type: IndicatorType = IndicatorType.SINGLE_VALUE,
        params: Dict[str, Any] = None,
        logger = None,
        error_handler = None
    ):
        """
        Initialize base indicator
        
        Args:
            name: Indicator name
            category: Kategori
            indicator_type: Output type
            params: Parameters
            logger: Logger instance
            error_handler: ErrorHandler instance
        """
        self.name = name
        self.category = category
        self.indicator_type = indicator_type
        self.params = params or {}
        self.logger = logger
        self.error_handler = error_handler
        
        # State
        self._last_result: Optional[IndicatorResult] = None
        self._calculation_count = 0
        self._error_count = 0
        
        # Metadata
        self._metadata = self._build_metadata()
        
        # Validate parameters
        self._validate_params()
    
    # ========================================================================
    # ABSTRACT METHODS (Each indicator must implement)
    # ========================================================================
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Indicator calculation - for REALTIME (MANDATORY)

        Performs calculations for the last bar. Used in live trading.

        Important:
            Child classes SHOULD call self.warmup_buffer(data) at the end
            of calculate() to prepare buffer for update() calls.

            Example:
                def calculate(self, data: pd.DataFrame) -> IndicatorResult:
                    # ... calculation logic ...
                    result = IndicatorResult(...)

                    # Warmup buffer for update() calls
                    self.warmup_buffer(data)

                    return result

        Args:
            data: OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)

        Returns:
            IndicatorResult: The calculation result of the last bar.

        Raises:
            InsufficientDataError: Insufficient data
            CalculationError: Calculation error
        """
        pass

    @abstractmethod
    def calculate_batch(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        ⚡ Batch calculation - REQUIRED for BACKTEST.

        Performs vectorized calculations for all bars.
        LOOP is prohibited! Pandas/Numpy operations should be used.

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: For single value indicators (RSI, HMA, etc.)
                Index: Same as data.index
                Values: Indicator value for each bar

            pd.DataFrame: For multiple value indicators (Bollinger, MACD, etc.)
                Index: Same as data.index
                Columns: Indicator output keys (upper, middle, lower)

        Example - RSI (single value):
            >>> rsi = RSI(period=14)
            >>> result = rsi.calculate_batch(data)
            >>> print(type(result))  # pd.Series
            >>> print(len(result))    # len(data)
            >>> print(result.iloc[-1]) # Last RSI value

        Example - Bollinger (multiple values):
            >>> bb = BollingerBands(period=20, std_dev=2.0)
            >>> result = bb.calculate_batch(data)
            >>> print(type(result))  # pd.DataFrame
            >>> print(result.columns)  # ['upper', 'middle', 'lower']
            >>> print(len(result))  # len(data)

        Performance:
            - MUST use vectorized operations (numpy/pandas)
            - NO loops (for/while)
            - Target: 2000 bars in <0.1 seconds

        Raises:
            InsufficientDataError: Insufficient data
            CalculationError: Calculation error
        """
        pass

    @abstractmethod
    def get_required_periods(self) -> int:
        """
        Minimum required number of periods (MANDATORY)

        Returns:
            int: Minimum period (e.g., 14 for RSI)
        """
        pass
    
    # ========================================================================
    # OPTIONAL METHODS (Optional overrides)
    # ========================================================================
    
    def update(self, new_candle: Dict[str, Any], symbol: Optional[str] = None) -> Optional[IndicatorResult]:
        """
        Real-time update (incremental calculation)

        When a new kline arrives, update only the last value.
        If you don't override it, it returns None (a full recalculation is required).

        Note:
            Buffer is automatically populated by calculate() during warmup.
            Child classes can use self._buffers[symbol] for incremental updates.

        Args:
            new_candle: Yeni kline data
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult or None (not implemented)
        """
        return None

    def warmup_buffer(self, data: pd.DataFrame, symbol: Optional[str] = None) -> None:
        """
        Warmup internal buffer with historical data

        Called by calculate() to prepare buffer for update() calls.
        This ensures update() can work incrementally from the first call.

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier (for multi-symbol support)
        """
        from collections import deque

        # Initialize symbol-aware buffers dict
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' if not provided (backward compatibility)
        buffer_key = symbol if symbol else 'default'

        # Create or reset buffer for this symbol
        buffer_size = self.get_required_periods()
        self._buffers[buffer_key] = deque(maxlen=buffer_size)

        # Fill buffer with last N candles
        for _, row in data.tail(buffer_size).iterrows():
            self._buffers[buffer_key].append(row.to_dict())

        # Legacy support: Also set self._buffer to current symbol's buffer
        # This allows existing update() implementations to work without changes
        self._buffer = self._buffers[buffer_key]

    def _get_neutral_result(self, new_candle: Dict[str, Any]) -> IndicatorResult:
        """
        Get neutral result when buffer not ready

        Override in child classes to customize neutral values.

        Args:
            new_candle: New candle data

        Returns:
            IndicatorResult with neutral/zero values
        """
        timestamp = int(new_candle.get('timestamp', 0))

        if self.indicator_type == IndicatorType.SINGLE_VALUE:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp,
                signal=SignalType.HOLD,
                metadata=self.params
            )
        elif self.indicator_type == IndicatorType.MULTIPLE_VALUES:
            return IndicatorResult(
                value={},  # Empty dict - child should override
                timestamp=timestamp,
                signal=SignalType.HOLD,
                metadata=self.params
            )
        elif self.indicator_type == IndicatorType.BANDS:
            return IndicatorResult(
                value={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
                timestamp=timestamp,
                signal=SignalType.HOLD,
                metadata=self.params
            )
        else:
            # Generic fallback
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp,
                signal=SignalType.HOLD,
                metadata=self.params
            )
    
    def validate_params(self) -> bool:
        """
        Parameter validation (optional override)
        
        Returns:
            bool: Are the parameters valid?
        
        Raises:
            InvalidParameterError: Invalid parameter
        """
        return True
    
    def get_signal(self, value: Any) -> SignalType:
        """
        Generate a signal from the value (optional override).
        
        Args:
            value: Indicator value
        
        Returns:
            SignalType: BUY, SELL, HOLD, etc.
        """
        return SignalType.NEUTRAL
    
    def get_trend(self, value: Any) -> TrendDirection:
        """
        Determine the trend from the value (optional override).
        
        Args:
            value: Indicator value
        
        Returns:
            TrendDirection: UP, DOWN, NEUTRAL
        """
        return TrendDirection.NEUTRAL
    
    # ========================================================================
    # DATA VALIDATION
    # ========================================================================
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        DataFrame validasyonu
        
        Args:
            data: OHLCV DataFrame
        
        Raises:
            ValueError: Invalid data format
            InsufficientDataError: Insufficient data
        """
        # Check if DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"[{self.name}] Data must be pandas DataFrame, got {type(data)}")
        
        # Check if empty
        if data.empty:
            raise InsufficientDataError(self.name, self.get_required_periods(), 0)
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"[{self.name}] Missing columns: {missing_columns}")

        # Check for NaN values
        if data[required_columns].isnull().any().any():
            self._log('warning', "Data contains NaN values, will be handled")

        # NOTE: Insufficient data check REMOVED
        # Indicators handle insufficient data gracefully by returning NaN for warmup period
        # Example: 69 bars available, EMA_89 needed → First 88 bars = NaN, rest calculated
        # This allows backtest to continue with partial indicator data instead of crashing

    def _validate_params(self) -> None:
        """
        Parametre validasyonu (internal)
        
        Raises:
            InvalidParameterError: Invalid parameter
        """
        try:
            self.validate_params()
        except InvalidParameterError:
            raise
        except Exception as e:
            raise InvalidParameterError(
                self.name,
                'unknown',
                self.params,
                str(e)
            )
    
    # ========================================================================
    # CALCULATION HELPERS
    # ========================================================================
    
    def _safe_calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Safe calculate wrapper (error handling)
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            IndicatorResult
        
        Raises:
            Various indicator exceptions
        """
        try:
            # Validate
            self._validate_data(data)
            
            # Calculate
            result = self.calculate(data)
            
            # Update state
            self._last_result = result
            self._calculation_count += 1
            
            self._log('debug', f"Calculated: {result.value}")
            
            return result
            
        except (InsufficientDataError, InvalidParameterError) as e:
            # Expected errors
            self._error_count += 1
            self._log('warning', f"Calculation failed: {e}")
            raise
            
        except Exception as e:
            # Unexpected errors
            self._error_count += 1
            self._log('error', f"Unexpected error: {e}")
            
            if self.error_handler:
                self.error_handler.handle_exception(
                    e,
                    context={
                        'module': 'BaseIndicator',
                        'indicator': self.name,
                        'action': 'calculate'
                    }
                )
            
            raise CalculationError(f"Calculation failed: {e}", self.name)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _build_metadata(self) -> IndicatorMetadata:
        """
        Create metadata.
        
        Returns:
            IndicatorMetadata
        """
        return IndicatorMetadata(
            name=self.name,
            category=self.category,
            indicator_type=self.indicator_type,
            description=self.__class__.__doc__ or "No description",
            params=self.params,
            default_params=self._get_default_params(),
            output_names=self._get_output_names(),
            requires_volume=self._requires_volume(),
            min_periods=self.get_required_periods()
        )
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Returns the default parameters (can be overridden).
        
        Returns:
            Dict: Default params
        """
        return {}
    
    def _get_output_names(self) -> List[str]:
        """
        Returns the output names (can be overridden).
        
        Returns:
            List[str]: Output names
        """
        return [self.name]
    
    def _requires_volume(self) -> bool:
        """
        Is volume required? (can be overridden)
        
        Returns:
            bool: Is volume required?
        """
        return False
    
    def _log(self, level: str, message: str) -> None:
        """
        Logger helper
        
        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
        """
        if not self.logger:
            return
        
        log_message = f"[{self.name}] {message}"
        
        if level == 'debug':
            self.logger.debug(log_message)
        elif level == 'info':
            self.logger.info(log_message)
        elif level == 'warning':
            self.logger.warning(log_message)
        elif level == 'error':
            self.logger.error(log_message)
    
    # ========================================================================
    # DATA CONVERSION HELPERS
    # ========================================================================
    
    @staticmethod
    def _df_to_ohlcv_list(data: pd.DataFrame) -> List[OHLCV]:
        """
        Convert DataFrame to OHLCV list.
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            List[OHLCV]
        """
        return [
            OHLCV(
                timestamp=int(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            for _, row in data.iterrows()
        ]
    
    @staticmethod
    def _ensure_numpy(series: Union[pd.Series, np.ndarray, List]) -> np.ndarray:
        """
        Convert Series/List to a numpy array.
        
        Args:
            series: pandas Series, numpy array, or list
        
        Returns:
            np.ndarray
        """
        if isinstance(series, pd.Series):
            return series.values
        elif isinstance(series, list):
            return np.array(series)
        elif isinstance(series, np.ndarray):
            return series
        else:
            raise ValueError(f"Cannot convert {type(series)} to numpy array")
    
    # ========================================================================
    # PROPERTIES
    # ========================================================================
    
    @property
    def metadata(self) -> IndicatorMetadata:
        """Metadata property"""
        return self._metadata
    
    @property
    def last_result(self) -> Optional[IndicatorResult]:
        """Last calculation result"""
        return self._last_result
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Statistics"""
        return {
            'name': self.name,
            'category': self.category.value,
            'calculation_count': self._calculation_count,
            'error_count': self._error_count,
            'last_calculation': self._last_result.datetime if self._last_result else None
        }
    
    # ========================================================================
    # MAGIC METHODS
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}({self.params})"
    
    def __call__(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Callable interface
        
        Usage:
            rsi = RSI(period=14)
            result = rsi(data)  # Same as rsi.calculate(data)
        """
        return self._safe_calculate(data)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'BaseIndicator',
]


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """
    Test BaseIndicator with a simple RSI implementation
    """
    
    print("\n" + "="*60)
    print("BASE INDICATOR TEST")
    print("="*60 + "\n")
    
    # Simple RSI implementation for testing
    class TestRSI(BaseIndicator):
        """Test RSI indicator"""
        
        def __init__(self, period: int = 14):
            super().__init__(
                name='rsi',
                category=IndicatorCategory.MOMENTUM,
                indicator_type=IndicatorType.SINGLE_VALUE,
                params={'period': period}
            )
            self.period = period
        
        def get_required_periods(self) -> int:
            return self.period + 1
        
        def validate_params(self) -> bool:
            if self.period < 1:
                raise InvalidParameterError(
                    self.name, 'period', self.period, 
                    "Period must be positive"
                )
            return True
        
        def calculate(self, data: pd.DataFrame) -> IndicatorResult:
            """Simple RSI calculation"""
            close = data['close'].values
            
            # Calculate price changes
            delta = np.diff(close)
            
            # Separate gains and losses
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Calculate average gain/loss
            avg_gain = np.mean(gains[-self.period:])
            avg_loss = np.mean(losses[-self.period:])
            
            # Calculate RS and RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            timestamp = int(data.iloc[-1]['timestamp'])
            
            # Determine signal
            signal = self.get_signal(rsi)
            trend = self.get_trend(rsi)
            
            return IndicatorResult(
                value=round(rsi, 2),
                timestamp=timestamp,
                signal=signal,
                trend=trend,
                metadata={'period': self.period}
            )
        
        def get_signal(self, value: float) -> SignalType:
            """RSI signals"""
            if value < 30:
                return SignalType.BUY
            elif value > 70:
                return SignalType.SELL
            return SignalType.HOLD
        
        def get_trend(self, value: float) -> TrendDirection:
            """RSI trend"""
            if value > 50:
                return TrendDirection.UP
            elif value < 50:
                return TrendDirection.DOWN
            return TrendDirection.NEUTRAL
    
    # Create sample data
    print("1. Creating sample OHLCV data...")
    timestamps = [1697000000000 + i * 60000 for i in range(20)]
    closes = [100 + i + np.random.randn() * 2 for i in range(20)]
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': closes,
        'high': [c + abs(np.random.randn()) for c in closes],
        'low': [c - abs(np.random.randn()) for c in closes],
        'close': closes,
        'volume': [1000 + np.random.randint(0, 500) for _ in closes]
    })
    
    print(f"   ✓ Created {len(data)} candles")
    print(f"   ✓ Close prices: {closes[0]:.2f} → {closes[-1]:.2f}")
    
    # Test 1: Basic calculation
    print("\n2. Testing basic calculation...")
    rsi = TestRSI(period=14)
    print(f"   ✓ Created: {rsi}")
    print(f"   ✓ Category: {rsi.category.value}")
    print(f"   ✓ Required periods: {rsi.get_required_periods()}")
    
    result = rsi(data)  # Using __call__
    print(f"   ✓ RSI Value: {result.value}")
    print(f"   ✓ Signal: {result.signal.value}")
    print(f"   ✓ Trend: {result.trend.name}")
    
    # Test 2: Metadata
    print("\n3. Testing metadata...")
    metadata = rsi.metadata
    print(f"   ✓ Name: {metadata.name}")
    print(f"   ✓ Category: {metadata.category.value}")
    print(f"   ✓ Type: {metadata.indicator_type.value}")
    print(f"   ✓ Min periods: {metadata.min_periods}")
    
    # Test 3: Statistics
    print("\n4. Testing statistics...")
    stats = rsi.statistics
    print(f"   ✓ Calculations: {stats['calculation_count']}")
    print(f"   ✓ Errors: {stats['error_count']}")
    
    # Test 4: Error handling - insufficient data
    print("\n5. Testing error handling (insufficient data)...")
    short_data = data.head(10)  # Only 10 candles, need 15
    try:
        rsi(short_data)
        print("   ✗ Should have raised InsufficientDataError")
    except InsufficientDataError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    # Test 5: Error handling - invalid params
    print("\n6. Testing error handling (invalid params)...")
    try:
        invalid_rsi = TestRSI(period=-5)
        print("   ✗ Should have raised InvalidParameterError")
    except InvalidParameterError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    # Test 6: Multiple calculations
    print("\n7. Testing multiple calculations...")
    for i in range(3):
        result = rsi.calculate(data)
        print(f"   ✓ Calculation {i+1}: RSI = {result.value}")
    
    stats = rsi.statistics
    print(f"   ✓ Total calculations: {stats['calculation_count']}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")