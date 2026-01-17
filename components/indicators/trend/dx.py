#!/usr/bin/env python3
"""
indicators/trend/dx.py - DX (Directional Movement Index)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

DX (Directional Movement Index) - Yön Hareketli İndeks.
ADX'in temel bileşeni, trend gücünü ölçer.

Özellikler:
- +DI ve -DI arasındaki farkı gösterir
- Trend gücünü ölçer (yönünü değil)
- 0-100 arasında değer üretir
- Yüksek DX = Güçlü trend
- Düşük DX = Zayıf trend
- ADX hesaplamasında kullanılır

Kullanım:
    from components.indicators import get_indicator_class

    DX = get_indicator_class('dx')
    dx = DX(period=14)
    result = dx.calculate(data)
    print(result.value['dx'], result.value['plus_di'], result.value['minus_di'])

Formül:
    TR = max(High - Low, abs(High - Close[1]), abs(Low - Close[1]))
    +DM = High - High[1] (if > 0 and > -DM, else 0)
    -DM = Low[1] - Low (if > 0 and > +DM, else 0)

    +DI = 100 * SMA(+DM, period) / SMA(TR, period)
    -DI = 100 * SMA(-DM, period) / SMA(TR, period)

    DX = 100 * abs(+DI - -DI) / (+DI + -DI)

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
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class DX(BaseIndicator):
    """
    DX - Directional Movement Index

    ADX'in temel bileşeni. +DI ve -DI arasındaki farkı gösterir.
    Trend gücünü ölçer.

    Args:
        period: DX periyodu (varsayılan: 14)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, period: int = 14, logger=None, error_handler=None):
        self.period = period

        super().__init__(
            name='dx',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period * 2

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
            pd.DataFrame: DX, +DI ve -DI değerleri
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range hesaplama
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement hesaplama
        up_move = high - high.shift()
        down_move = low.shift() - low

        # +DM ve -DM
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

        # TA-Lib uyumlu: Wilder smoothing (RMA) kullan
        # RMA = EMA with alpha = 1/period
        alpha = 1 / self.period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # +DI ve -DI hesaplama
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # DX hesaplama
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)

        return pd.DataFrame({
            'dx': dx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - state-based update için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel, multi-symbol desteği için)
        """
        super().warmup_buffer(data, symbol)

        buffer_key = symbol if symbol else 'default'
        if not hasattr(self, '_dx_state'):
            self._dx_state = {}

        if len(data) >= self.period + 1:
            # calculate_batch ile aynı hesaplamayı kullan
            batch = self.calculate_batch(data)

            high = data['high'].values
            low = data['low'].values
            close = data['close'].values

            # Son state için batch'ten değerleri al
            # EWM state'ini hesaplamak için TR, +DM, -DM serilerini hesapla
            high_s = data['high']
            low_s = data['low']
            close_s = data['close']

            # True Range
            high_low = high_s - low_s
            high_close = abs(high_s - close_s.shift())
            low_close = abs(low_s - close_s.shift())
            tr_series = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Directional Movement
            up_move = high_s - high_s.shift()
            down_move = low_s.shift() - low_s

            plus_dm_series = ((up_move > down_move) & (up_move > 0)) * up_move
            minus_dm_series = ((down_move > up_move) & (down_move > 0)) * down_move

            # Wilder smoothing son değerleri (EWM ile hesapla)
            alpha = 1 / self.period
            atr = tr_series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
            plus_dm_smooth = plus_dm_series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
            minus_dm_smooth = minus_dm_series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

            self._dx_state[buffer_key] = {
                'atr': atr,
                'plus_dm_smooth': plus_dm_smooth,
                'minus_dm_smooth': minus_dm_smooth,
                'last_high': high[-1],
                'last_low': low[-1],
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
            IndicatorResult: Güncel DX değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # State varsa incremental Wilder smoothing hesapla
        if hasattr(self, '_dx_state') and buffer_key in self._dx_state:
            state = self._dx_state[buffer_key]
            atr = state['atr']
            plus_dm_smooth = state['plus_dm_smooth']
            minus_dm_smooth = state['minus_dm_smooth']
            last_high = state['last_high']
            last_low = state['last_low']
            last_close = state['last_close']
            alpha = state['alpha']

            # True Range
            hl = high_val - low_val
            hc = abs(high_val - last_close)
            lc = abs(low_val - last_close)
            tr = max(hl, hc, lc)

            # Directional Movement
            up_move = high_val - last_high
            down_move = last_low - low_val

            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0

            # Wilder smoothing (RMA) güncelle
            new_atr = alpha * tr + (1 - alpha) * atr
            new_plus_dm = alpha * plus_dm + (1 - alpha) * plus_dm_smooth
            new_minus_dm = alpha * minus_dm + (1 - alpha) * minus_dm_smooth

            # +DI ve -DI hesapla
            plus_di = 100 * (new_plus_dm / new_atr) if new_atr > 0 else 0
            minus_di = 100 * (new_minus_dm / new_atr) if new_atr > 0 else 0

            # DX hesapla
            di_sum = plus_di + minus_di
            dx_val = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0

            # State güncelle
            self._dx_state[buffer_key] = {
                'atr': new_atr,
                'plus_dm_smooth': new_plus_dm,
                'minus_dm_smooth': new_minus_dm,
                'last_high': high_val,
                'last_low': low_val,
                'last_close': close_val,
                'alpha': alpha
            }

            # Sinyal ve trend belirleme
            if plus_di > minus_di:
                signal = SignalType.BUY
                trend = TrendDirection.UP
            elif plus_di < minus_di:
                signal = SignalType.SELL
                trend = TrendDirection.DOWN
            else:
                signal = SignalType.HOLD
                trend = TrendDirection.NEUTRAL

            return IndicatorResult(
                value={
                    'dx': round(dx_val, 2),
                    'plus_di': round(plus_di, 2),
                    'minus_di': round(minus_di, 2)
                },
                timestamp=timestamp_val,
                signal=signal,
                trend=trend,
                strength=min(dx_val, 100),
                metadata={'period': self.period}
            )

        # State yoksa yetersiz veri
        return IndicatorResult(
            value=0.0,
            timestamp=timestamp_val,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=0.0,
            metadata={'insufficient_data': True}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        DX hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: DX değerleri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)

        # Son değerleri al
        dx_val = batch_result['dx'].iloc[-1]
        plus_di = batch_result['plus_di'].iloc[-1]
        minus_di = batch_result['minus_di'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirleme: +DI > -DI = BUY
        if plus_di > minus_di:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif plus_di < minus_di:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Güç: DX değeri (0-100 arası)
        strength = min(dx_val, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'dx': round(dx_val, 2),
                'plus_di': round(plus_di, 2),
                'minus_di': round(minus_di, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'period': 14}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['dx', 'plus_di', 'minus_di']

    def _requires_volume(self) -> bool:
        """DX volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['DX']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """DX indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 DX (DIRECTIONAL MOVEMENT INDEX) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 25, 150)
    noise = np.random.randn(150) * 2.5
    close_prices = base_price + trend + noise

    # OHLC oluştur
    opens = close_prices + np.random.randn(150) * 0.5
    highs = np.maximum(opens, close_prices) + np.abs(np.random.randn(150) * 1.5)
    lows = np.minimum(opens, close_prices) - np.abs(np.random.randn(150) * 1.5)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(150)]
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(close_prices):.2f} -> {max(close_prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    dx = DX(period=14)
    print(f"   ✅ Oluşturuldu: {dx}")
    print(f"   ✅ Kategori: {dx.category.value}")
    print(f"   ✅ Gerekli periyot: {dx.get_required_periods()}")

    result = dx(data)
    print(f"   ✅ DX: {result.value['dx']}")
    print(f"   ✅ +DI: {result.value['plus_di']}")
    print(f"   ✅ -DI: {result.value['minus_di']}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = dx.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 değer:")
    print(batch_result.tail())

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [7, 14, 21]:
        dx_test = DX(period=period)
        result = dx_test.calculate(data)
        print(f"   ✅ DX({period}): DX={result.value['dx']:.2f}, "
              f"+DI={result.value['plus_di']:.2f}, -DI={result.value['minus_di']:.2f}")

    # Test 4: DI crossover analizi
    print("\n5. +DI/-DI crossover analizi...")
    batch_result = dx.calculate_batch(data)
    plus_di_values = batch_result['plus_di'].dropna()
    minus_di_values = batch_result['minus_di'].dropna()

    crossovers = 0
    for i in range(1, min(len(plus_di_values), len(minus_di_values))):
        if (plus_di_values.iloc[i-1] < minus_di_values.iloc[i-1] and plus_di_values.iloc[i] > minus_di_values.iloc[i]) or \
           (plus_di_values.iloc[i-1] > minus_di_values.iloc[i-1] and plus_di_values.iloc[i] < minus_di_values.iloc[i]):
            crossovers += 1

    print(f"   ✅ Toplam DI crossover: {crossovers}")
    print(f"   ✅ +DI > -DI barlar: {sum(plus_di_values > minus_di_values)}")
    print(f"   ✅ +DI < -DI barlar: {sum(plus_di_values < minus_di_values)}")

    # Test 5: DX seviye analizi
    print("\n6. DX seviye analizi...")
    dx_values = batch_result['dx'].dropna()

    print(f"   ✅ Ortalama DX: {dx_values.mean():.2f}")
    print(f"   ✅ Max DX: {dx_values.max():.2f}")
    print(f"   ✅ Min DX: {dx_values.min():.2f}")
    print(f"   ✅ Güçlü trend (DX>25): {sum(dx_values > 25)}")
    print(f"   ✅ Zayıf trend (DX<20): {sum(dx_values < 20)}")

    # Test 6: Validasyon testi
    print("\n7. Validasyon testi...")
    try:
        invalid_dx = DX(period=0)
        print("   ❌ Hata: Geçersiz period kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
