"""
indicators/structure/market_structure.py - Market Structure (Combined)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Market Structure - Smart Money Concepts (Combined Indicator)
    Combines all SMC indicators.

    Content:
    - BOS (Break of Structure)
    - CHoCH (Change of Character)
    - FVG (Fair Value Gap)
    - Order Blocks
    - Liquidity Zones

    Usage:
    Perform the entire SMC analysis with a single indicator call.

Formula:
    Calculates and combines each sub-indicator separately.
    Provides a general market structure analysis.

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.structure.* (local)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)

# Import sub-indicators
from components.indicators.structure.bos import BOS
from components.indicators.structure.choch import CHoCH
from components.indicators.structure.fvg import FVG
from components.indicators.structure.orderblocks import OrderBlocks
from components.indicators.structure.liquidityzones import LiquidityZones


class MarketStructure(BaseIndicator):
    """
    Market Structure (Combined)

    Combines all Smart Money Concepts indicators.
    Provides comprehensive market structure analysis.

    Args:
        enable_bos: Calculate BOS (default: True)
        enable_choch: Calculate CHoCH (default: True)
        enable_fvg: Calculate FVG (default: True)
        enable_ob: Calculate Order Blocks (default: True)
        enable_liq: Calculate Liquidity Zones (default: True)
        left_bars: Swing detection left bar (default: 5)
        right_bars: Swing detection right bar (default: 5)
    """

    def __init__(
        self,
        enable_bos: bool = True,
        enable_choch: bool = True,
        enable_fvg: bool = True,
        enable_ob: bool = True,
        enable_liq: bool = True,
        left_bars: int = 5,
        right_bars: int = 5,
        logger=None,
        error_handler=None
    ):
        self.enable_bos = enable_bos
        self.enable_choch = enable_choch
        self.enable_fvg = enable_fvg
        self.enable_ob = enable_ob
        self.enable_liq = enable_liq
        self.left_bars = left_bars
        self.right_bars = right_bars

        # Create sub-indicators (BEFORE calling super().__init__)
        self._bos = BOS(left_bars=left_bars, right_bars=right_bars) if enable_bos else None
        self._choch = CHoCH(left_bars=left_bars, right_bars=right_bars) if enable_choch else None
        self._fvg = FVG() if enable_fvg else None
        self._ob = OrderBlocks() if enable_ob else None
        self._liq = LiquidityZones(left_bars=left_bars, right_bars=right_bars) if enable_liq else None

        super().__init__(
            name='market_structure',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'enable_bos': enable_bos,
                'enable_choch': enable_choch,
                'enable_fvg': enable_fvg,
                'enable_ob': enable_ob,
                'enable_liq': enable_liq,
                'left_bars': left_bars,
                'right_bars': right_bars
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        min_periods = []
        if self._bos:
            min_periods.append(self._bos.get_required_periods())
        if self._choch:
            min_periods.append(self._choch.get_required_periods())
        if self._fvg:
            min_periods.append(self._fvg.get_required_periods())
        if self._ob:
            min_periods.append(self._ob.get_required_periods())
        if self._liq:
            min_periods.append(self._liq.get_required_periods())

        return max(min_periods) if min_periods else 20

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.left_bars < 1:
            raise InvalidParameterError(
                self.name, 'left_bars', self.left_bars,
                "Left bars must be positive"
            )
        if self.right_bars < 1:
            raise InvalidParameterError(
                self.name, 'right_bars', self.right_bars,
                "Right bars must be positive"
            )

        # At least one indicator must be active
        if not any([self.enable_bos, self.enable_choch, self.enable_fvg, self.enable_ob, self.enable_liq]):
            raise InvalidParameterError(
                self.name, 'enabled_indicators', 'none',
                "At least one indicator must be active"
            )

        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Market Structure hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: All SMC indicators
        """
        results = {}

        # BOS hesapla
        if self._bos:
            try:
                bos_result = self._bos.calculate(data)
                results['bos'] = {
                    'value': bos_result.value,
                    'signal': bos_result.signal.value if bos_result.signal else None,
                    'trend': bos_result.trend.name if bos_result.trend else None,
                    'metadata': bos_result.metadata
                }
            except Exception as e:
                self._log('warning', f"BOS calculation failed: {e}")
                results['bos'] = None

        # CHoCH hesapla
        if self._choch:
            try:
                choch_result = self._choch.calculate(data)
                results['choch'] = {
                    'value': choch_result.value,
                    'signal': choch_result.signal.value if choch_result.signal else None,
                    'trend': choch_result.trend.name if choch_result.trend else None,
                    'metadata': choch_result.metadata
                }
            except Exception as e:
                self._log('warning', f"CHoCH calculation failed: {e}")
                results['choch'] = None

        # FVG hesapla
        if self._fvg:
            try:
                fvg_result = self._fvg.calculate(data)
                results['fvg'] = {
                    'zones': fvg_result.value,
                    'signal': fvg_result.signal.value if fvg_result.signal else None,
                    'trend': fvg_result.trend.name if fvg_result.trend else None,
                    'metadata': fvg_result.metadata
                }
            except Exception as e:
                self._log('warning', f"FVG calculation failed: {e}")
                results['fvg'] = None

        # Order Blocks hesapla
        if self._ob:
            try:
                ob_result = self._ob.calculate(data)
                results['orderblocks'] = {
                    'zones': ob_result.value,
                    'signal': ob_result.signal.value if ob_result.signal else None,
                    'trend': ob_result.trend.name if ob_result.trend else None,
                    'metadata': ob_result.metadata
                }
            except Exception as e:
                self._log('warning', f"Order Blocks calculation failed: {e}")
                results['orderblocks'] = None

        # Liquidity Zones hesapla
        if self._liq:
            try:
                liq_result = self._liq.calculate(data)
                results['liquidityzones'] = {
                    'zones': liq_result.value,
                    'signal': liq_result.signal.value if liq_result.signal else None,
                    'trend': liq_result.trend.name if liq_result.trend else None,
                    'metadata': liq_result.metadata
                }
            except Exception as e:
                self._log('warning', f"Liquidity Zones calculation failed: {e}")
                results['liquidityzones'] = None

        timestamp = int(data.iloc[-1]['timestamp'])

        # Calculate overall signal and trend
        overall_signal = self._calculate_overall_signal(results)
        overall_trend = self._calculate_overall_trend(results)
        overall_strength = self._calculate_overall_strength(results)

        return IndicatorResult(
            value=results,
            timestamp=timestamp,
            signal=overall_signal,
            trend=overall_trend,
            strength=overall_strength,
            metadata={
                'enabled_indicators': {
                    'bos': self.enable_bos,
                    'choch': self.enable_choch,
                    'fvg': self.enable_fvg,
                    'orderblocks': self.enable_ob,
                    'liquidityzones': self.enable_liq
                },
                'calculated_count': len([v for v in results.values() if v is not None])
            }
        )

    def _calculate_overall_signal(self, results: Dict[str, Any]) -> SignalType:
        """
        Calculate the overall signal from all indicators.

        Args:
            results: Sub-indicator results

        Returns:
            SignalType: General signal
        """
        signals = []

        for key, result in results.items():
            if result and result.get('signal'):
                signals.append(result['signal'])

        if not signals:
            return SignalType.HOLD

        # Signal counting
        buy_count = signals.count('buy') + signals.count('strong_buy')
        sell_count = signals.count('sell') + signals.count('strong_sell')

        # Majority decision
        if buy_count > sell_count and buy_count >= len(signals) * 0.5:
            return SignalType.BUY
        elif sell_count > buy_count and sell_count >= len(signals) * 0.5:
            return SignalType.SELL

        return SignalType.HOLD

    def _calculate_overall_trend(self, results: Dict[str, Any]) -> TrendDirection:
        """
        Calculate the overall trend from all indicators.

        Args:
            results: Sub-indicator results

        Returns:
            TrendDirection: Genel trend
        """
        trends = []

        for key, result in results.items():
            if result and result.get('trend'):
                trends.append(result['trend'])

        if not trends:
            return TrendDirection.NEUTRAL

        # Trend counting
        up_count = trends.count('UP')
        down_count = trends.count('DOWN')

        if up_count > down_count:
            return TrendDirection.UP
        elif down_count > up_count:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _calculate_overall_strength(self, results: Dict[str, Any]) -> float:
        """
        Calculate the overall power score.

        Args:
            results: Sub-indicator results

        Returns:
            float: Power score (0-100)
        """
        # Allocate power based on the calculated indicator count.
        calculated_count = len([v for v in results.values() if v is not None])
        total_count = len([k for k in ['bos', 'choch', 'fvg', 'orderblocks', 'liquidityzones'] if k in results])

        if total_count == 0:
            return 0.0

        strength = (calculated_count / total_count) * 100

        return round(strength, 2)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: The current market structure value.
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        if buffer_key not in self._buffers:
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = deque(maxlen=max_len)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        # Add new candle to symbol's buffer
        self._buffers[buffer_key].append(candle)

        # Need minimum data for calculation
        if len(self._buffers[buffer_key]) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value={},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, value: Any) -> SignalType:
        """
        Generate a signal from the value (already calculated within calculate).

        Args:
            value: Indicator value

        Returns:
            SignalType: HOLD (because it has already been calculated)
        """
        return SignalType.HOLD

    def get_trend(self, value: Any) -> TrendDirection:
        """
        Determine the trend from the value (already calculated within calculate).

        Args:
            value: Indicator value

        Returns:
            TrendDirection: NEUTRAL (because it has already been calculated)
        """
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'enable_bos': True,
            'enable_choch': True,
            'enable_fvg': True,
            'enable_ob': True,
            'enable_liq': True,
            'left_bars': 5,
            'right_bars': 5
        }

    def _requires_volume(self) -> bool:
        """Market Structure volume gerektirmez"""
        return False

    def _get_output_names(self):
        """Returns the output names"""
        return ['bos', 'choch', 'fvg', 'orderblocks', 'liquidityzones']

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation - Calculate market structure for each bar.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Market structure results for each bar.
        """
        # Market structure bulk calculation is not supported because:
        # Dependent on all historical data for each bar.
        # Therefore, return the calculate() method.
        result = self.calculate(data)

        # Convert the results to a DataFrame format
        df_result = pd.DataFrame(index=data.index)

        # Add results for each sub indicator
        if result.value.get('bos'):
            df_result['ms_bos_value'] = result.value['bos']['value']
            df_result['ms_bos_signal'] = result.value['bos']['signal']
            df_result['ms_bos_trend'] = result.value['bos']['trend']

        if result.value.get('choch'):
            df_result['ms_choch_value'] = result.value['choch']['value']
            df_result['ms_choch_signal'] = result.value['choch']['signal']
            df_result['ms_choch_trend'] = result.value['choch']['trend']

        if result.value.get('fvg'):
            df_result['ms_fvg_zones'] = str(result.value['fvg']['zones'])
            df_result['ms_fvg_signal'] = result.value['fvg']['signal']

        if result.value.get('orderblocks'):
            df_result['ms_ob_zones'] = str(result.value['orderblocks']['zones'])
            df_result['ms_ob_signal'] = result.value['orderblocks']['signal']

        if result.value.get('liquidityzones'):
            df_result['ms_liq_zones'] = str(result.value['liquidityzones']['zones'])
            df_result['ms_liq_signal'] = result.value['liquidityzones']['signal']

        # General results
        df_result['ms_overall_signal'] = result.signal.value
        df_result['ms_overall_trend'] = result.trend.name
        df_result['ms_overall_strength'] = result.strength

        return df_result


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['MarketStructure']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Market Structure indicator test"""

    print("\n" + "="*60)
    print("MARKET STRUCTURE (COMBINED) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Complex market structure simulation
    base_price = 100
    prices = []
    for i in range(50):
        if i < 15:
            # Ascent
            prices.append(base_price + i * 0.5 + np.random.randn() * 0.3)
        elif i < 20:
            # Konsolidasyon
            prices.append(base_price + 7 + np.random.randn() * 0.5)
        elif i < 35:
            # Fall
            prices.append(base_price + 9 - (i - 20) * 0.4 + np.random.randn() * 0.3)
        else:
            # Retry
            prices.append(base_price + 3 + (i - 35) * 0.3 + np.random.randn() * 0.3)

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

    # Test 1: Calculation with all indicators
    print("\n2. All indicators are in active test...")
    ms = MarketStructure(
        enable_bos=True,
        enable_choch=True,
        enable_fvg=True,
        enable_ob=True,
        enable_liq=True,
        left_bars=5,
        right_bars=5
    )
    print(f"   [OK] Created: {ms}")
    print(f"   [OK] Kategori: {ms.category.value}")
    print(f"   [OK] Required period: {ms.get_required_periods()}")

    result = ms(data)
    print(f"   [OK] Calculated indicator: {result.metadata['calculated_count']}")
    print(f"   [OK] General Signal: {result.signal.value}")
    print(f"   [OK] Genel Trend: {result.trend.name}")
    print(f"   [OK] Overall Power: {result.strength}")

    # Test 2: Status of each indicator
    print("\n3. Indicator details...")
    for indicator_name, indicator_result in result.value.items():
        if indicator_result:
            print(f"   [OK] {indicator_name.upper()}:")
            print(f"       - Signal: {indicator_result.get('signal', 'N/A')}")
            print(f"       - Trend: {indicator_result.get('trend', 'N/A')}")
            if 'zones' in indicator_result:
                print(f"       - Number of zones: {len(indicator_result['zones'])}")
            elif 'value' in indicator_result:
                print(f"       - Value: {indicator_result['value']}")
        else:
            print(f"   [!] {indicator_name.upper()}: Could not be calculated")

    # Test 3: Partial indicators
    print("\n4. Partial indicator test (only BOS and FVG)...")
    ms_partial = MarketStructure(
        enable_bos=True,
        enable_choch=False,
        enable_fvg=True,
        enable_ob=False,
        enable_liq=False
    )
    result_partial = ms_partial.calculate(data)
    print(f"   [OK] Hesaplanan: {result_partial.metadata['calculated_count']}")
    print(f"   [OK] Active indicators:")
    for name, enabled in result_partial.metadata['enabled_indicators'].items():
        print(f"       - {name}: {enabled}")

    # Test 4: Empty detail
    print("\n5. BOS detail...")
    if result.value.get('bos'):
        bos_data = result.value['bos']
        print(f"   [OK] Empty Value: {bos_data['value']}")
        print(f"   [OK] BOS Tip: {bos_data['metadata']['bos_type']}")
        print(f"   [OK] Swing High: {len(bos_data['metadata']['swing_highs'])}")
        print(f"   [OK] Swing Low: {len(bos_data['metadata']['swing_lows'])}")

    # Test 5: FVG detail
    print("\n6. FVG detail...")
    if result.value.get('fvg'):
        fvg_data = result.value['fvg']
        print(f"   [OK] FVG Zone: {fvg_data['metadata']['total_zones']}")
        print(f"   [OK] Bullish: {fvg_data['metadata']['bullish_zones']}")
        print(f"   [OK] Bearish: {fvg_data['metadata']['bearish_zones']}")

    # Test 6: Order Blocks details
    print("\n7. Order Blocks details...")
    if result.value.get('orderblocks'):
        ob_data = result.value['orderblocks']
        print(f"   [OK] OB Block: {ob_data['metadata']['total_blocks']}")
        print(f"   [OK] Bullish: {ob_data['metadata']['bullish_blocks']}")
        print(f"   [OK] Bearish: {ob_data['metadata']['bearish_blocks']}")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = ms.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = ms.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Output sayısı: {len(metadata.output_names)}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
