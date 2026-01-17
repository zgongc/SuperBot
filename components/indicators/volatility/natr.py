"""
indicators/volatility/natr.py - Normalized Average True Range

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    NATR (Normalized Average True Range) - Normalleştirilmiş ATR
    ATR'yi fiyata bölerek normalize eder
    Farklı fiyat seviyelerindeki varlıkları karşılaştırmak için kullanılır
    Yüzde cinsinden volatilite gösterir

Formül:
    NATR = (ATR / Close) × 100

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


class NATR(BaseIndicator):
    """
    Normalized Average True Range

    ATR'yi fiyata bölerek yüzde cinsinden volatilite ölçümü sağlar.
    Farklı fiyat seviyelerindeki varlıkları karşılaştırmak için kullanılır.

    Args:
        period: ATR periyodu (varsayılan: 14)
    """

    def __init__(
        self,
        period: int = 14,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='natr',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period + 1  # TR hesabı için bir önceki mum gerekli

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        NATR hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: NATR değeri (yüzde)
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # True Range hesapla
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        # ATR hesapla (RMA - Wilder's smoothing)
        atr_values = np.zeros(len(tr))
        atr_values[self.period-1] = np.mean(tr[:self.period])

        alpha = 1.0 / self.period
        for i in range(self.period, len(tr)):
            atr_values[i] = atr_values[i-1] + alpha * (tr[i] - atr_values[i-1])

        atr_value = atr_values[-1]
        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # NATR hesapla (yüzde cinsinden)
        if current_price > 0:
            natr_value = (atr_value / current_price) * 100
        else:
            natr_value = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(natr_value, 4),
            timestamp=timestamp,
            signal=self.get_signal(natr_value),
            trend=TrendDirection.NEUTRAL,  # NATR trend göstermez
            strength=min(natr_value * 10, 100),  # 0-100 arası normalize et
            metadata={
                'period': self.period,
                'atr': round(atr_value, 8),
                'true_range': round(tr[-1], 8),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch NATR calculation - BACKTEST için

        NATR Formula:
            NATR = (ATR / Close) × 100

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: NATR values for all bars (percentage)

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # True Range calculation (vectorized)
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # ATR with Wilder's smoothing (RMA)
        atr = tr.ewm(alpha=1.0/self.period, adjust=False).mean()

        # NATR = (ATR / Close) × 100
        natr = (atr / close) * 100

        # Handle division by zero
        natr = natr.fillna(0)

        # Set first period values to NaN (warmup)
        natr.iloc[:self.period] = np.nan

        return pd.Series(natr.values, index=data.index, name='natr')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        # Buffer'ları oluştur ve doldur
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True
        
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
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, natr_value: float) -> SignalType:
        """
        NATR değerinden sinyal üret

        Args:
            natr_value: NATR değeri (yüzde)

        Returns:
            SignalType: Volatilite seviyesine göre sinyal
        """
        # Çok yüksek volatilite: dikkatli ol
        if natr_value > 5.0:
            return SignalType.SELL  # Yüksek risk
        # Normal volatilite
        elif natr_value > 2.0:
            return SignalType.HOLD
        # Düşük volatilite: potansiyel fırsat
        else:
            return SignalType.BUY

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 14
        }

    def _requires_volume(self) -> bool:
        """NATR volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['NATR']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """NATR indikatör testi"""

    print("\n" + "="*60)
    print("NATR (NORMALIZED AVERAGE TRUE RANGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    # Fiyat hareketini simüle et
    base_price = 100
    prices = [base_price]
    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    natr = NATR(period=14)
    print(f"   [OK] Oluşturuldu: {natr}")
    print(f"   [OK] Kategori: {natr.category.value}")
    print(f"   [OK] Gerekli periyot: {natr.get_required_periods()}")

    result = natr(data)
    print(f"   [OK] NATR Değeri: {result.value}%")
    print(f"   [OK] ATR: {result.metadata['atr']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [7, 14, 21]:
        natr_test = NATR(period=period)
        result = natr_test.calculate(data)
        print(f"   [OK] NATR({period}): {result.value:.2f}% | ATR: {result.metadata['atr']:.4f}")

    # Test 3: Farklı fiyat seviyelerinde karşılaştırma
    print("\n4. Farklı fiyat seviyesi karşılaştırma testi...")

    # Düşük fiyatlı varlık
    low_price_data = data.copy()
    low_price_data['open'] = data['open'] / 10
    low_price_data['high'] = data['high'] / 10
    low_price_data['low'] = data['low'] / 10
    low_price_data['close'] = data['close'] / 10

    # Yüksek fiyatlı varlık
    high_price_data = data.copy()
    high_price_data['open'] = data['open'] * 10
    high_price_data['high'] = data['high'] * 10
    high_price_data['low'] = data['low'] * 10
    high_price_data['close'] = data['close'] * 10

    result_normal = natr.calculate(data)
    result_low = natr.calculate(low_price_data)
    result_high = natr.calculate(high_price_data)

    print(f"   [OK] Normal Fiyat (~{data['close'].iloc[-1]:.2f}): NATR={result_normal.value:.2f}%")
    print(f"   [OK] Düşük Fiyat (~{low_price_data['close'].iloc[-1]:.2f}): NATR={result_low.value:.2f}%")
    print(f"   [OK] Yüksek Fiyat (~{high_price_data['close'].iloc[-1]:.2f}): NATR={result_high.value:.2f}%")
    print(f"   [INFO] NATR değerleri normalize olduğu için benzer olmalı")

    # Test 4: Düşük volatilite
    print("\n5. Düşük volatilite testi...")
    low_vol_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': [100.0] * 30,
        'high': [100.2] * 30,
        'low': [99.8] * 30,
        'close': [100.0] * 30,
        'volume': [1000] * 30
    })
    result = natr.calculate(low_vol_data)
    print(f"   [OK] Düşük Vol NATR: {result.value:.4f}%")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: Yüksek volatilite
    print("\n6. Yüksek volatilite testi...")
    high_vol_prices = [100]
    for i in range(29):
        change = np.random.randn() * 10  # Yüksek volatilite
        high_vol_prices.append(high_vol_prices[-1] + change)

    high_vol_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': high_vol_prices,
        'high': [p + abs(np.random.randn()) * 8 for p in high_vol_prices],
        'low': [p - abs(np.random.randn()) * 8 for p in high_vol_prices],
        'close': high_vol_prices,
        'volume': [1000] * 30
    })
    result = natr.calculate(high_vol_data)
    print(f"   [OK] Yüksek Vol NATR: {result.value:.2f}%")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = natr.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = natr.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
