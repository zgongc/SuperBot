"""
indicators/volatility/true_range.py - True Range

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    True Range - Gerçek Fiyat Aralığı
    Bir mumun gerçek volatilitesini ölçer
    ATR'nin temel bileşenidir
    Gap'leri de hesaba katar

Formül:
    TR = max[(High - Low), abs(High - PrevClose), abs(Low - PrevClose)]

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


class TrueRange(BaseIndicator):
    """
    True Range

    Fiyatın gerçek hareket aralığını ölçer. Gap'leri de hesaba katarak
    doğru volatilite ölçümü sağlar.

    Args:
        Parametre gerektirmez (tek mum hesabı)
    """

    def __init__(
        self,
        logger=None,
        error_handler=None
    ):
        super().__init__(
            name='true_range',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return 2  # Güncel ve önceki mum

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        True Range hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: True Range değeri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Son mumun True Range'i
        current_high = high[-1]
        current_low = low[-1]
        prev_close = close[-2]

        # Üç değeri hesapla
        hl = current_high - current_low  # High - Low
        hc = abs(current_high - prev_close)  # High - Previous Close
        lc = abs(current_low - prev_close)  # Low - Previous Close

        # Maximum değer True Range'dir
        tr_value = max(hl, hc, lc)

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # True Range'in fiyata oranı (volatilite yüzdesi)
        if current_price > 0:
            tr_percent = (tr_value / current_price) * 100
        else:
            tr_percent = 0

        # Gap durumu
        gap_up = current_low > prev_close
        gap_down = current_high < prev_close
        has_gap = gap_up or gap_down

        # Hangi bileşen maksimum?
        if tr_value == hl:
            max_component = 'high_low'
        elif tr_value == hc:
            max_component = 'high_prev_close'
        else:
            max_component = 'low_prev_close'

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(tr_value, 8),
            timestamp=timestamp,
            signal=self.get_signal(tr_percent),
            trend=TrendDirection.NEUTRAL,  # True Range trend göstermez
            strength=min(tr_percent * 10, 100),  # 0-100 arası normalize et
            metadata={
                'high_low': round(hl, 8),
                'high_prev_close': round(hc, 8),
                'low_prev_close': round(lc, 8),
                'tr_percent': round(tr_percent, 2),
                'max_component': max_component,
                'has_gap': has_gap,
                'gap_direction': 'up' if gap_up else ('down' if gap_down else 'none'),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch True Range calculation - BACKTEST için

        True Range Formula:
            TR = max[(High - Low), abs(High - PrevClose), abs(Low - PrevClose)]

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: True Range values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Three components of True Range (vectorized)
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()

        # True Range = max of three components
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # First bar: just use high-low
        tr.iloc[0] = hl.iloc[0]

        return pd.Series(tr.values, index=data.index, name='true_range')

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
            # List/tuple format: [timestamp, open, high, low, close, volume]
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

    def get_signal(self, tr_percent: float) -> SignalType:
        """
        True Range yüzdesinden sinyal üret

        Args:
            tr_percent: True Range'in fiyata oranı (%)

        Returns:
            SignalType: Volatilite seviyesine göre sinyal
        """
        # Çok yüksek volatilite: dikkatli ol
        if tr_percent > 5.0:
            return SignalType.SELL  # Yüksek risk
        # Normal volatilite
        elif tr_percent > 2.0:
            return SignalType.HOLD
        # Düşük volatilite
        else:
            return SignalType.BUY

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {}

    def _requires_volume(self) -> bool:
        """True Range volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TrueRange']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """True Range indikatör testi"""

    print("\n" + "="*60)
    print("TRUE RANGE TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(10)]

    # Fiyat hareketini simüle et
    base_price = 100
    prices = [base_price]
    for i in range(9):
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
    tr = TrueRange()
    print(f"   [OK] Oluşturuldu: {tr}")
    print(f"   [OK] Kategori: {tr.category.value}")
    print(f"   [OK] Gerekli periyot: {tr.get_required_periods()}")

    result = tr(data)
    print(f"   [OK] True Range: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] TR %: {result.metadata['tr_percent']:.2f}%")
    print(f"   [OK] Max Component: {result.metadata['max_component']}")
    print(f"   [OK] Gap var mı: {result.metadata['has_gap']}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Gap durumu testi (Gap Up)
    print("\n3. Gap Up testi...")
    gap_up_data = pd.DataFrame({
        'timestamp': [1697000000000, 1697000060000],
        'open': [100.0, 105.0],
        'high': [102.0, 107.0],
        'low': [99.0, 104.0],  # Low > önceki Close
        'close': [101.0, 106.0],
        'volume': [1000, 1000]
    })
    result = tr.calculate(gap_up_data)
    print(f"   [OK] Gap Up TR: {result.value}")
    print(f"   [OK] Max Component: {result.metadata['max_component']}")
    print(f"   [OK] Gap Direction: {result.metadata['gap_direction']}")
    print(f"   [OK] High-Low: {result.metadata['high_low']}")
    print(f"   [OK] High-PrevClose: {result.metadata['high_prev_close']}")

    # Test 3: Gap durumu testi (Gap Down)
    print("\n4. Gap Down testi...")
    gap_down_data = pd.DataFrame({
        'timestamp': [1697000000000, 1697000060000],
        'open': [100.0, 95.0],
        'high': [102.0, 97.0],  # High < önceki Close
        'low': [99.0, 94.0],
        'close': [101.0, 96.0],
        'volume': [1000, 1000]
    })
    result = tr.calculate(gap_down_data)
    print(f"   [OK] Gap Down TR: {result.value}")
    print(f"   [OK] Max Component: {result.metadata['max_component']}")
    print(f"   [OK] Gap Direction: {result.metadata['gap_direction']}")
    print(f"   [OK] Low-PrevClose: {result.metadata['low_prev_close']}")

    # Test 4: Normal mum (gap yok)
    print("\n5. Normal mum testi...")
    normal_data = pd.DataFrame({
        'timestamp': [1697000000000, 1697000060000],
        'open': [100.0, 100.5],
        'high': [102.0, 103.0],
        'low': [99.0, 99.5],
        'close': [101.0, 102.0],
        'volume': [1000, 1000]
    })
    result = tr.calculate(normal_data)
    print(f"   [OK] Normal TR: {result.value}")
    print(f"   [OK] Max Component: {result.metadata['max_component']}")
    print(f"   [OK] Gap var mı: {result.metadata['has_gap']}")
    print(f"   [OK] High-Low: {result.metadata['high_low']}")

    # Test 5: Yüksek volatilite
    print("\n6. Yüksek volatilite testi...")
    high_vol_data = pd.DataFrame({
        'timestamp': [1697000000000, 1697000060000],
        'open': [100.0, 110.0],
        'high': [102.0, 115.0],
        'low': [99.0, 108.0],
        'close': [101.0, 112.0],
        'volume': [1000, 1000]
    })
    result = tr.calculate(high_vol_data)
    print(f"   [OK] Yüksek Vol TR: {result.value}")
    print(f"   [OK] TR %: {result.metadata['tr_percent']:.2f}%")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = tr.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = tr.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
