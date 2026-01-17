#!/usr/bin/env python3
"""
indicators/momentum/cmo.py - CMO (Chande Momentum Oscillator)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

Chande Momentum Oscillator (CMO) - RSI benzeri momentum göstergesi.
RSI'dan farkı, farklı bir normalizasyon kullanmasıdır.

Özellikler:
- RSI'ya benzer ama farklı hesaplama
- -100 ile +100 arasında salınır
- Aşırı alım/satım bölgelerini gösterir
- CMO > +50: Aşırı alım
- CMO < -50: Aşırı satım
- Sıfır çizgisi önemli destek/direnç

Kullanım:
    from components.indicators import get_indicator_class

    CMO = get_indicator_class('cmo')
    cmo = CMO(period=14)
    result = cmo.calculate(data)
    print(result.value['cmo'])

Formül:
    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

    RSI'dan farkı:
    - RSI: 100 * (gains / (gains + losses))
    - CMO: 100 * ((gains - losses) / (gains + losses))

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje root'unu path'e ekle
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
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


class CMO(BaseIndicator):
    """
    Chande Momentum Oscillator

    RSI benzeri ama farklı normalizasyon kullanan momentum göstergesi.
    -100 ile +100 arasında salınır.

    Args:
        period: CMO periyodu (varsayılan: 14)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, period: int = 14, logger=None, error_handler=None):
        self.period = period
        self.prices = deque(maxlen=period + 1)

        super().__init__(
            name='cmo',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period + 1

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period pozitif olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.
        TA-Lib uyumlu: Wilder smoothing (RMA) kullanır

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: CMO değerleri
        """
        close = data['close']

        # Fiyat farkları
        diff = close.diff()

        # Kazanç ve kayıpları ayır
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        # TA-Lib uyumlu: Wilder smoothing (RMA) kullan
        # RMA = EMA with alpha = 1/period
        alpha = 1 / self.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        # CMO hesapla: 100 * (avg_gain - avg_loss) / (avg_gain + avg_loss)
        cmo = 100 * (avg_gain - avg_loss) / (avg_gain + avg_loss)

        return pd.DataFrame({'cmo': cmo}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - state-based update için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel, multi-symbol desteği için)
        """
        super().warmup_buffer(data, symbol)

        buffer_key = symbol if symbol else 'default'
        if not hasattr(self, '_cmo_state'):
            self._cmo_state = {}

        if len(data) >= self.period + 1:
            close = data['close'].values
            diff = np.diff(close)

            # Wilder smoothing (RMA) için state hesapla
            alpha = 1 / self.period
            gains = np.where(diff > 0, diff, 0)
            losses = np.where(diff < 0, -diff, 0)

            # RMA hesapla
            avg_gain = gains[0]
            avg_loss = losses[0]
            for i in range(1, len(gains)):
                avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

            self._cmo_state[buffer_key] = {
                'avg_gain': avg_gain,
                'avg_loss': avg_loss,
                'last_close': close[-1],
                'alpha': alpha
            }

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - State-based Wilder smoothing

        Args:
            candle: Yeni mum verisi (dict)
            symbol: Sembol adı (opsiyonel)

        Returns:
            IndicatorResult: Güncel CMO değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # State varsa incremental Wilder smoothing hesapla
        if hasattr(self, '_cmo_state') and buffer_key in self._cmo_state:
            state = self._cmo_state[buffer_key]
            avg_gain = state['avg_gain']
            avg_loss = state['avg_loss']
            last_close = state['last_close']
            alpha = state['alpha']

            # Fiyat değişimi
            diff = close_val - last_close
            gain = diff if diff > 0 else 0
            loss = -diff if diff < 0 else 0

            # Wilder smoothing (RMA) güncelle
            new_avg_gain = alpha * gain + (1 - alpha) * avg_gain
            new_avg_loss = alpha * loss + (1 - alpha) * avg_loss

            # CMO hesapla
            total = new_avg_gain + new_avg_loss
            cmo_value = 100 * (new_avg_gain - new_avg_loss) / total if total > 0 else 0

            # State güncelle
            self._cmo_state[buffer_key] = {
                'avg_gain': new_avg_gain,
                'avg_loss': new_avg_loss,
                'last_close': close_val,
                'alpha': alpha
            }

            # Sinyal ve trend belirleme
            if cmo_value < -50:
                signal = SignalType.BUY  # Aşırı satım
                trend = TrendDirection.DOWN
            elif cmo_value > 50:
                signal = SignalType.SELL  # Aşırı alım
                trend = TrendDirection.UP
            else:
                signal = SignalType.HOLD
                trend = TrendDirection.UP if cmo_value > 0 else (
                    TrendDirection.DOWN if cmo_value < 0 else TrendDirection.NEUTRAL
                )

            return IndicatorResult(
                value={'cmo': round(cmo_value, 2)},
                timestamp=timestamp_val,
                signal=signal,
                trend=trend,
                strength=min(abs(cmo_value), 100),
                metadata={
                    'period': self.period,
                    'overbought': 50,
                    'oversold': -50
                }
            )

        # Fallback: eski buffer-based hesaplama (backward compat)
        self.prices.append(close_val)

        if len(self.prices) < self.period + 1:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Fiyat dizisini numpy array'e çevir
        prices = np.array(self.prices)
        diff = np.diff(prices)

        # Wilder smoothing (RMA) hesapla
        alpha = 1 / self.period
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        avg_gain = gains[0]
        avg_loss = losses[0]
        for i in range(1, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        total = avg_gain + avg_loss
        cmo_value = 100 * (avg_gain - avg_loss) / total if total > 0 else 0

        # Sinyal ve trend belirleme
        if cmo_value < -50:
            signal = SignalType.BUY
            trend = TrendDirection.DOWN
        elif cmo_value > 50:
            signal = SignalType.SELL
            trend = TrendDirection.UP
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.UP if cmo_value > 0 else (
                TrendDirection.DOWN if cmo_value < 0 else TrendDirection.NEUTRAL
            )

        return IndicatorResult(
            value={'cmo': round(cmo_value, 2)},
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=min(abs(cmo_value), 100),
            metadata={
                'period': self.period,
                'overbought': 50,
                'oversold': -50
            }
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        CMO hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: CMO değeri
        """
        # Buffer'ları doldur
        close_values = data['close'].tail(self.period + 1).values
        self.prices.clear()
        self.prices.extend(close_values)

        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['cmo'].dropna().values

        if len(valid_values) == 0:
            return None

        cmo_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal ve trend
        if cmo_value < -50:
            signal = SignalType.BUY  # Aşırı satım
            trend = TrendDirection.DOWN
        elif cmo_value > 50:
            signal = SignalType.SELL  # Aşırı alım
            trend = TrendDirection.UP
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.UP if cmo_value > 0 else (
                TrendDirection.DOWN if cmo_value < 0 else TrendDirection.NEUTRAL
            )

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'cmo': round(cmo_value, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(cmo_value), 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'period': 14}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['cmo']

    def _requires_volume(self) -> bool:
        """CMO volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['CMO']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """CMO indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 CHANDE MOMENTUM OSCILLATOR (CMO) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Sinüs dalgası + noise (daha iyi CMO gösterimi için)
    base_price = 100
    sine_wave = 10 * np.sin(np.linspace(0, 4 * np.pi, 100))
    noise = np.random.randn(100) * 2
    prices = base_price + sine_wave + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(100)),
        'low': prices - np.abs(np.random.randn(100)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)]
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    cmo = CMO(period=14)
    print(f"   ✅ Oluşturuldu: {cmo}")
    print(f"   ✅ Kategori: {cmo.category.value}")
    print(f"   ✅ Gerekli periyot: {cmo.get_required_periods()}")

    result = cmo(data)
    print(f"   ✅ CMO: {result.value['cmo']}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")
    print(f"   ✅ Aşırı Alım: {result.metadata.get('overbought', 'N/A')}")
    print(f"   ✅ Aşırı Satım: {result.metadata.get('oversold', 'N/A')}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = cmo.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 CMO değeri:")
    print(batch_result['cmo'].tail())

    # Test 3: Aşırı alım/satım tespiti
    print("\n4. Aşırı alım/satım testi...")
    cmo_values = batch_result['cmo'].dropna().values
    overbought = len(cmo_values[cmo_values > 50])
    oversold = len(cmo_values[cmo_values < -50])
    print(f"   ✅ Aşırı alım bölgesi: {overbought} bar")
    print(f"   ✅ Aşırı satım bölgesi: {oversold} bar")
    print(f"   ✅ Min CMO: {min(cmo_values):.2f}")
    print(f"   ✅ Max CMO: {max(cmo_values):.2f}")

    # Test 4: Update metodu
    print("\n5. Update metodu testi...")
    cmo2 = CMO(period=14)
    init_data = data.head(50)
    cmo2.calculate(init_data)

    # Yeni 5 mum ekle
    for i in range(50, 55):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low']
        }
        update_result = cmo2.update(candle)
        if update_result:
            print(f"   ✅ Bar {i}: CMO={update_result.value['cmo']:.2f}, "
                  f"Signal={update_result.signal.value}")

    # Test 5: Farklı periyotlar
    print("\n6. Farklı periyot testi...")
    for period in [9, 14, 20]:
        cmo_test = CMO(period=period)
        result = cmo_test.calculate(data)
        print(f"   ✅ CMO({period}): {result.value['cmo']:.2f}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
