"""
indicators/momentum/cci.py - Commodity Channel Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    CCI (Commodity Channel Index) - Momentum osilatörü
    Aralık: Genellikle -100 ile +100 arası (dışına çıkabilir)
    Aşırı Alım: > +100
    Aşırı Satım: < -100

Formül:
    CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
    Typical Price = (High + Low + Close) / 3

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


class CCI(BaseIndicator):
    """
    Commodity Channel Index

    Fiyatın istatistiksel ortalamasından sapmasını ölçer.
    Aşırı alım/satım koşullarını ve trend gücünü belirlemek için kullanılır.

    Args:
        period: CCI periyodu (varsayılan: 20)
        overbought: Aşırı alım seviyesi (varsayılan: 100)
        oversold: Aşırı satım seviyesi (varsayılan: -100)
    """

    def __init__(
        self,
        period: int = 20,
        overbought: float = 100,
        oversold: float = -100,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='cci',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'overbought': overbought,
                'oversold': oversold
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot pozitif olmalı"
            )
        if self.oversold >= self.overbought:
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold, overbought'tan küçük olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        CCI hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: CCI değeri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Typical Price hesapla
        typical_price = (high + low + close) / 3

        # Typical Price'ın SMA'sını hesapla
        sma_tp = np.mean(typical_price[-self.period:])

        # Mean Deviation hesapla
        mean_deviation = np.mean(np.abs(typical_price[-self.period:] - sma_tp))

        # CCI hesapla
        if mean_deviation == 0:
            cci_value = 0.0
        else:
            cci_value = (typical_price[-1] - sma_tp) / (0.015 * mean_deviation)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(cci_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(cci_value),
            trend=self.get_trend(cci_value),
            strength=min(abs(cci_value), 100),  # 0-100 arası normalize et
            metadata={
                'period': self.period,
                'typical_price': round(typical_price[-1], 2),
                'sma': round(sma_tp, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch CCI calculation - BACKTEST için

        CCI Formula:
            CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
            Typical Price = (High + Low + Close) / 3

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: CCI values for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        # Typical Price = (High + Low + Close) / 3
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # SMA of Typical Price
        sma_tp = typical_price.rolling(window=self.period).mean()

        # Mean Deviation (vectorized)
        mad = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )

        # CCI = (TP - SMA) / (0.015 * MAD)
        cci = (typical_price - sma_tp) / (0.015 * mad)

        # Handle division by zero
        cci = cci.replace([np.inf, -np.inf], 0)
        cci = cci.fillna(0)

        # Set first period values to NaN (warmup)
        cci.iloc[:self.period-1] = np.nan

        return pd.Series(cci.values, index=data.index, name='cci')

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

    def get_signal(self, value: float) -> SignalType:
        """
        CCI değerinden sinyal üret

        Args:
            value: CCI değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if value < self.oversold:
            return SignalType.BUY
        elif value > self.overbought:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        CCI değerinden trend belirle

        Args:
            value: CCI değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > 0:
            return TrendDirection.UP
        elif value < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20,
            'overbought': 100,
            'oversold': -100
        }

    def _requires_volume(self) -> bool:
        """CCI volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['CCI']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """CCI indikatör testi - cache_manager.py tarzı"""

    print("=" * 70)
    print("📊 CCI (Commodity Channel Index) Test")
    print("=" * 70)

    # Test data oluştur
    print("\n1️⃣  Test Verisi Oluştur")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    base_price = 100
    prices = [base_price]
    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"  ✅ {len(data)} mum oluşturuldu")
    print(f"  ✅ Fiyat: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"  ✅ Değişim: {((max(prices) - min(prices)) / min(prices) * 100):.2f}%")

    # Test 1: Temel Hesaplama
    print("\n2️⃣  Temel Hesaplama")
    cci = CCI(period=20)
    print(f"  ✅ İndikatör: {cci}")
    print(f"  ✅ Kategori: {cci.category.value}")
    print(f"  ✅ Tip: {cci.indicator_type.value}")
    print(f"  ✅ Min periyot: {cci.get_required_periods()}")

    result = cci(data)

    # Sinyal emoji
    signal_emoji = {
        'buy': '🟢',
        'sell': '🔴',
        'hold': '🟡'
    }.get(result.signal.value, '⚪')

    # Trend emoji
    trend_emoji = {
        'UP': '⬆️',
        'DOWN': '⬇️',
        'NEUTRAL': '➡️'
    }.get(result.trend.name, '❓')

    print(f"\n  📈 CCI Sonuçları:")
    print(f"  ✅ Değer: {result.value}")
    print(f"  ✅ Sinyal: {signal_emoji} {result.signal.value.upper()}")
    print(f"  ✅ Trend: {trend_emoji} {result.trend.name}")
    print(f"  ✅ Güç: {result.strength:.2f}/100")
    print(f"  ✅ Metadata: period={result.metadata['period']}, tp={result.metadata['typical_price']}")

    # Test 2: Farklı Periyotlar
    print("\n3️⃣  Farklı Periyot Karşılaştırması")
    for period in [10, 20, 30]:
        cci_test = CCI(period=period)
        res = cci_test.calculate(data)
        sig_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}.get(res.signal.value, '⚪')
        print(f"  {sig_emoji} CCI({period:2d}): {res.value:7.2f} | {res.signal.value:4s}")

    # Test 3: Özel Seviyeler
    print("\n4️⃣  Özel Seviyeler")
    cci_custom = CCI(period=20, overbought=150, oversold=-150)
    result = cci_custom.calculate(data)
    print(f"  ✅ OB/OS: ±{cci_custom.overbought}")
    print(f"  ✅ CCI: {result.value}")
    print(f"  ✅ Sinyal: {result.signal.value}")

    # Test 4: İstatistikler
    print("\n5️⃣  İstatistikler")
    stats = cci.statistics
    print(f"  📊 Hesaplama: {stats['calculation_count']}")
    print(f"  ❌ Hata: {stats['error_count']}")
    print(f"  🕐 Son: {stats['last_calculation']}")

    # Test 5: Metadata
    print("\n6️⃣  Metadata")
    metadata = cci.metadata
    print(f"  ✅ İsim: {metadata.name}")
    print(f"  ✅ Kategori: {metadata.category.value}")
    print(f"  ✅ Açıklama: {metadata.description[:50]}...")
    print(f"  ✅ Volume gerekli: {metadata.requires_volume}")
    print(f"  ✅ Default params: {metadata.default_params}")

    # Test 6: Sinyal Analizi
    print("\n7️⃣  Sinyal Analizi")
    if result.signal == SignalType.BUY:
        print("  🟢 ALIŞ SİNYALİ - CCI aşırı satım bölgesinde")
    elif result.signal == SignalType.SELL:
        print("  🔴 SATIŞ SİNYALİ - CCI aşırı alım bölgesinde")
    else:
        print("  🟡 BEKLETİN - CCI normal aralıkta")

    print(f"\n  📋 Detay:")
    print(f"  ✅ Mevcut: {result.value}")
    print(f"  ✅ Aşırı alım: {cci.overbought}")
    print(f"  ✅ Aşırı satım: {cci.oversold}")
    print(f"  ✅ Tavsiye: {'AL' if result.value < cci.oversold else 'SAT' if result.value > cci.overbought else 'BEKLE'}")

    print("\n" + "=" * 70)
    print("✅ Tüm testler başarıyla tamamlandı!")
    print("=" * 70 + "\n")
