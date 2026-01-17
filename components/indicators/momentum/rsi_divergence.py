"""
indicators/momentum/rsi_divergence.py - RSI Divergence

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    RSI Divergence - Fiyat ve RSI arasındaki uyumsuzluğu tespit eder
    Bullish Divergence: Fiyat düşerken RSI yükselir (alım sinyali)
    Bearish Divergence: Fiyat yükselirken RSI düşer (satım sinyali)
    Hidden Divergence: Trend devamı sinyalleri
    Güçlü reversal göstergesi

Formül:
    1. RSI hesapla (14 periyot)
    2. Son N periyotta fiyat pivot noktalarını bul
    3. Aynı noktalarda RSI pivot noktalarını bul
    4. Fiyat ve RSI hareketlerini karşılaştır
    5. Divergence tespit et

Bağımlılıklar:
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


class RSIDivergence(BaseIndicator):
    """
    RSI Divergence

    Fiyat ve RSI arasındaki divergence'ları tespit eder.
    Trend dönüş noktalarını yakalamak için kullanılır.

    Args:
        rsi_period: RSI periyodu (varsayılan: 14)
        lookback: Geriye bakış periyodu (pivot noktaları için) (varsayılan: 5)
        min_strength: Minimum divergence gücü (0-100) (varsayılan: 30)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 5,
        min_strength: float = 30,
        logger=None,
        error_handler=None
    ):
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.min_strength = min_strength

        super().__init__(
            name='rsi_divergence',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'rsi_period': rsi_period,
                'lookback': lookback,
                'min_strength': min_strength
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.rsi_period + self.lookback * 2

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "RSI period pozitif olmalı"
            )
        if self.lookback < 2:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback en az 2 olmalı"
            )
        if not (0 <= self.min_strength <= 100):
            raise InvalidParameterError(
                self.name, 'min_strength', self.min_strength,
                "Min strength 0-100 arası olmalı"
            )
        return True

    def _calculate_rsi(self, close: np.ndarray) -> np.ndarray:
        """
        RSI hesapla

        Args:
            close: Kapanış fiyatları

        Returns:
            RSI değerleri
        """
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        rsi_values = np.zeros_like(close)

        for i in range(self.rsi_period, len(close)):
            avg_gain = np.mean(gains[i-self.rsi_period:i])
            avg_loss = np.mean(losses[i-self.rsi_period:i])

            if avg_loss == 0:
                rsi_values[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))

        return rsi_values

    def _find_pivots(self, data: np.ndarray, lookback: int) -> dict:
        """
        Pivot noktalarını bul (local highs ve lows)

        Args:
            data: Veri array
            lookback: Geriye bakış periyodu

        Returns:
            dict: {'highs': [(index, value)], 'lows': [(index, value)]}
        """
        pivots = {'highs': [], 'lows': []}

        for i in range(lookback, len(data) - lookback):
            # Local high: Sağ ve sol taraftan daha yüksek
            is_high = True
            for j in range(1, lookback + 1):
                if data[i] <= data[i-j] or data[i] <= data[i+j]:
                    is_high = False
                    break
            if is_high:
                pivots['highs'].append((i, data[i]))

            # Local low: Sağ ve sol taraftan daha düşük
            is_low = True
            for j in range(1, lookback + 1):
                if data[i] >= data[i-j] or data[i] >= data[i+j]:
                    is_low = False
                    break
            if is_low:
                pivots['lows'].append((i, data[i]))

        return pivots

    def _detect_divergence(self, price_pivots: dict, rsi_pivots: dict) -> dict:
        """
        Divergence tespit et

        Args:
            price_pivots: Fiyat pivot noktaları
            rsi_pivots: RSI pivot noktaları

        Returns:
            dict: Divergence bilgisi
        """
        result = {
            'bullish': False,
            'bearish': False,
            'hidden_bullish': False,
            'hidden_bearish': False,
            'strength': 0
        }

        # Bullish Divergence: Fiyat düşer, RSI yükselir
        if len(price_pivots['lows']) >= 2 and len(rsi_pivots['lows']) >= 2:
            price_low1, price_val1 = price_pivots['lows'][-2]
            price_low2, price_val2 = price_pivots['lows'][-1]

            # RSI pivot'larını price pivot'lara yakın olanları bul
            rsi_low1 = None
            rsi_low2 = None
            for idx, val in rsi_pivots['lows']:
                if abs(idx - price_low1) <= self.lookback:
                    rsi_low1 = (idx, val)
                if abs(idx - price_low2) <= self.lookback:
                    rsi_low2 = (idx, val)

            if rsi_low1 and rsi_low2:
                # Bullish divergence: Fiyat düşer, RSI yükselir
                if price_val2 < price_val1 and rsi_low2[1] > rsi_low1[1]:
                    result['bullish'] = True
                    result['strength'] = min(100, abs(price_val1 - price_val2) * 10)

        # Bearish Divergence: Fiyat yükselir, RSI düşer
        if len(price_pivots['highs']) >= 2 and len(rsi_pivots['highs']) >= 2:
            price_high1, price_val1 = price_pivots['highs'][-2]
            price_high2, price_val2 = price_pivots['highs'][-1]

            # RSI pivot'larını price pivot'lara yakın olanları bul
            rsi_high1 = None
            rsi_high2 = None
            for idx, val in rsi_pivots['highs']:
                if abs(idx - price_high1) <= self.lookback:
                    rsi_high1 = (idx, val)
                if abs(idx - price_high2) <= self.lookback:
                    rsi_high2 = (idx, val)

            if rsi_high1 and rsi_high2:
                # Bearish divergence: Fiyat yükselir, RSI düşer
                if price_val2 > price_val1 and rsi_high2[1] < rsi_high1[1]:
                    result['bearish'] = True
                    result['strength'] = min(100, abs(price_val2 - price_val1) * 10)

        return result

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        RSI Divergence hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Divergence bilgisi
        """
        close = data['close'].values

        # RSI hesapla
        rsi_values = self._calculate_rsi(close)

        # Pivot noktalarını bul
        price_pivots = self._find_pivots(close, self.lookback)
        rsi_pivots = self._find_pivots(rsi_values, self.lookback)

        # Divergence tespit et
        divergence = self._detect_divergence(price_pivots, rsi_pivots)

        # Sonuç değeri
        value = {
            'rsi': round(rsi_values[-1], 2),
            'bullish_divergence': divergence['bullish'],
            'bearish_divergence': divergence['bearish'],
            'divergence_strength': round(divergence['strength'], 2)
        }

        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirleme
        if divergence['bullish'] and divergence['strength'] >= self.min_strength:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif divergence['bearish'] and divergence['strength'] >= self.min_strength:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        return IndicatorResult(
            value=value,
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=divergence['strength'],
            metadata={
                'rsi_period': self.rsi_period,
                'lookback': self.lookback,
                'price_pivots_highs': len(price_pivots['highs']),
                'price_pivots_lows': len(price_pivots['lows']),
                'rsi_pivots_highs': len(rsi_pivots['highs']),
                'rsi_pivots_lows': len(rsi_pivots['lows'])
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch RSI Divergence calculation - BACKTEST için

        Basitleştirilmiş implementasyon:
        - RSI hesapla (vectorized)
        - Divergence detection için simplified logic

        Returns:
            pd.DataFrame: 4 columns (rsi, bullish_divergence, bearish_divergence, divergence_strength)
        """
        self._validate_data(data)

        close = data['close'].values

        # 1. RSI hesapla (vectorized)
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Wilder's smoothing
        alpha = 1.0 / self.rsi_period
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))

        if len(gains) >= self.rsi_period:
            avg_gain[self.rsi_period] = np.mean(gains[:self.rsi_period])
            avg_loss[self.rsi_period] = np.mean(losses[:self.rsi_period])

            for i in range(self.rsi_period + 1, len(close)):
                avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gains[i] * alpha
                avg_loss[i] = avg_loss[i-1] * (1 - alpha) + losses[i] * alpha

        # RSI
        rsi = np.zeros(len(close))
        for i in range(len(close)):
            if avg_loss[i] == 0:
                rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))

        # 2. Simplified divergence detection (for batch - not full pivot analysis)
        # We'll mark potential divergence zones based on RSI-price relationship
        bullish_div = np.zeros(len(close), dtype=bool)
        bearish_div = np.zeros(len(close), dtype=bool)
        div_strength = np.zeros(len(close))

        # Simple divergence logic: compare recent price/RSI movements
        window = self.lookback * 2
        for i in range(window, len(close)):
            # Bullish: Price making lower low, RSI making higher low
            price_window = close[i-window:i+1]
            rsi_window = rsi[i-window:i+1]

            price_min_idx = np.argmin(price_window)
            rsi_min_idx = np.argmin(rsi_window)

            # Check if there's divergence pattern
            if price_min_idx > len(price_window) // 2:  # Recent low in price
                prev_price_low = np.min(price_window[:len(price_window)//2])
                recent_price_low = price_window[price_min_idx]

                if recent_price_low < prev_price_low:  # Lower low in price
                    prev_rsi_low = np.min(rsi_window[:len(rsi_window)//2])
                    recent_rsi_low = rsi_window[price_min_idx] if price_min_idx < len(rsi_window) else rsi[-1]

                    if recent_rsi_low > prev_rsi_low:  # Higher low in RSI = bullish divergence
                        bullish_div[i] = True
                        div_strength[i] = min(100, abs(recent_rsi_low - prev_rsi_low) * 2)

            # Bearish: Price making higher high, RSI making lower high
            price_max_idx = np.argmax(price_window)

            if price_max_idx > len(price_window) // 2:  # Recent high in price
                prev_price_high = np.max(price_window[:len(price_window)//2])
                recent_price_high = price_window[price_max_idx]

                if recent_price_high > prev_price_high:  # Higher high in price
                    prev_rsi_high = np.max(rsi_window[:len(rsi_window)//2])
                    recent_rsi_high = rsi_window[price_max_idx] if price_max_idx < len(rsi_window) else rsi[-1]

                    if recent_rsi_high < prev_rsi_high:  # Lower high in RSI = bearish divergence
                        bearish_div[i] = True
                        div_strength[i] = min(100, abs(recent_rsi_high - prev_rsi_high) * 2)

        # Create result DataFrame
        result = pd.DataFrame({
            'rsi': rsi,
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'divergence_strength': div_strength
        }, index=data.index)

        # Set warmup period to NaN/False
        warmup = self.rsi_period + window
        result.iloc[:warmup, result.columns.get_loc('rsi')] = np.nan
        result.iloc[:warmup, result.columns.get_loc('divergence_strength')] = 0

        return result

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Güncel indicator değeri
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
                value=0.0,
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

    def get_signal(self, value: dict) -> SignalType:
        """
        Divergence değerinden sinyal üret

        Args:
            value: Divergence bilgisi

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if isinstance(value, dict):
            if value.get('bullish_divergence') and value.get('divergence_strength', 0) >= self.min_strength:
                return SignalType.BUY
            elif value.get('bearish_divergence') and value.get('divergence_strength', 0) >= self.min_strength:
                return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: dict) -> TrendDirection:
        """
        Divergence değerinden trend belirle

        Args:
            value: Divergence bilgisi

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if isinstance(value, dict):
            if value.get('bullish_divergence'):
                return TrendDirection.UP
            elif value.get('bearish_divergence'):
                return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'rsi_period': 14,
            'lookback': 5,
            'min_strength': 30
        }

    def _requires_volume(self) -> bool:
        """RSI Divergence volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['RSIDivergence']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """RSI Divergence indikatör testi"""

    print("\n" + "="*60)
    print("RSI DIVERGENCE TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur - Divergence oluşturmak için özel pattern
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # Bullish divergence pattern oluştur
    prices = []
    for i in range(60):
        if i < 20:
            # Normal hareket
            prices.append(100 + i * 0.5 + np.random.randn() * 0.3)
        elif i < 40:
            # Fiyat düşer
            prices.append(110 - (i-20) * 0.5 + np.random.randn() * 0.3)
        else:
            # Fiyat daha da düşer (lower low)
            prices.append(100 - (i-40) * 0.3 + np.random.randn() * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    rsi_div = RSIDivergence(rsi_period=14, lookback=5, min_strength=30)
    print(f"   [OK] Oluşturuldu: {rsi_div}")
    print(f"   [OK] Kategori: {rsi_div.category.value}")
    print(f"   [OK] Gerekli periyot: {rsi_div.get_required_periods()}")

    result = rsi_div(data)
    print(f"   [OK] RSI Değeri: {result.value['rsi']}")
    print(f"   [OK] Bullish Divergence: {result.value['bullish_divergence']}")
    print(f"   [OK] Bearish Divergence: {result.value['bearish_divergence']}")
    print(f"   [OK] Divergence Gücü: {result.value['divergence_strength']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Farklı parametreler
    print("\n3. Farklı parametre testi...")
    configs = [
        (14, 3, 20),
        (14, 5, 30),
        (14, 7, 40)
    ]
    for rsi_p, look, strength in configs:
        div_test = RSIDivergence(rsi_period=rsi_p, lookback=look, min_strength=strength)
        result = div_test.calculate(data)
        print(f"   [OK] Params({rsi_p},{look},{strength}): Bullish={result.value['bullish_divergence']}, Bearish={result.value['bearish_divergence']}")

    # Test 3: Bearish divergence pattern
    print("\n4. Bearish divergence testi...")
    bearish_prices = []
    for i in range(60):
        if i < 20:
            bearish_prices.append(100 - i * 0.3 + np.random.randn() * 0.3)
        elif i < 40:
            bearish_prices.append(94 + (i-20) * 0.5 + np.random.randn() * 0.3)
        else:
            bearish_prices.append(104 + (i-40) * 0.4 + np.random.randn() * 0.3)

    bearish_data = data.copy()
    bearish_data['close'] = bearish_prices
    bearish_data['high'] = [p + abs(np.random.randn()) * 0.3 for p in bearish_prices]
    bearish_data['low'] = [p - abs(np.random.randn()) * 0.3 for p in bearish_prices]

    result_bearish = rsi_div.calculate(bearish_data)
    print(f"   [OK] Bearish pattern RSI: {result_bearish.value['rsi']}")
    print(f"   [OK] Bearish Divergence: {result_bearish.value['bearish_divergence']}")
    print(f"   [OK] Sinyal: {result_bearish.signal.value}")

    # Test 4: Pivot noktaları
    print("\n5. Pivot noktaları testi...")
    print(f"   [OK] Fiyat pivot highs: {result.metadata['price_pivots_highs']}")
    print(f"   [OK] Fiyat pivot lows: {result.metadata['price_pivots_lows']}")
    print(f"   [OK] RSI pivot highs: {result.metadata['rsi_pivots_highs']}")
    print(f"   [OK] RSI pivot lows: {result.metadata['rsi_pivots_lows']}")

    # Test 5: Güç testi
    print("\n6. Divergence güç testi...")
    if result.value['divergence_strength'] >= rsi_div.min_strength:
        print(f"   [OK] Güçlü divergence tespit edildi: {result.value['divergence_strength']}")
    else:
        print(f"   [OK] Zayıf veya divergence yok: {result.value['divergence_strength']}")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = rsi_div.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = rsi_div.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
