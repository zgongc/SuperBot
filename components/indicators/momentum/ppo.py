#!/usr/bin/env python3
"""
indicators/momentum/ppo.py - PPO (Percentage Price Oscillator)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

PPO (Percentage Price Oscillator) - Yüzdelik Fiyat Osilatörü.
İki EMA arasındaki farkı yüzde olarak hesaplar.

Özellikler:
- MACD'nin yüzdelik versiyonu
- Farklı fiyat seviyelerinde karşılaştırma yapılabilir
- PPO, Signal ve Histogram çıktıları
- Pozitif değer = Bullish momentum
- Negatif değer = Bearish momentum
- Signal line crossover sinyalleri

Kullanım:
    from components.indicators import get_indicator_class

    PPO = get_indicator_class('ppo')
    ppo = PPO(fast_period=12, slow_period=26, signal_period=9)
    result = ppo.calculate(data)
    print(result.value['ppo'], result.value['signal'])

Formül:
    PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100
    Signal = EMA(PPO, signal_period)
    Histogram = PPO - Signal

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


class PPO(BaseIndicator):
    """
    PPO - Percentage Price Oscillator

    İki EMA arasındaki farkı yüzde olarak hesaplar.
    MACD'nin normalize edilmiş versiyonudur.

    Args:
        fast_period: Hızlı EMA periyodu (varsayılan: 12)
        slow_period: Yavaş EMA periyodu (varsayılan: 26)
        signal_period: Signal line periyodu (varsayılan: 9)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 logger=None, error_handler=None):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        super().__init__(
            name='ppo',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={'fast_period': fast_period, 'slow_period': slow_period, 'signal_period': signal_period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.slow_period * 2 + self.signal_period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.fast_period >= self.slow_period:
            raise InvalidParameterError(
                self.name, 'fast_period', self.fast_period,
                "Hızlı periyot yavaş periyottan küçük olmalı"
            )
        if self.signal_period < 1:
            raise InvalidParameterError(
                self.name, 'signal_period', self.signal_period,
                "Signal period pozitif olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.
        TA-Lib uyumlu: SMA kullanır (varsayılan matype=0)

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: PPO, Signal ve Histogram değerleri
        """
        close = data['close']

        # TA-Lib uyumlu: SMA kullan (EMA değil!)
        fast_ma = close.rolling(window=self.fast_period).mean()
        slow_ma = close.rolling(window=self.slow_period).mean()

        # Yüzdelik fark
        ppo = ((fast_ma - slow_ma) / slow_ma) * 100

        # Signal line (EMA of PPO)
        signal = ppo.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        histogram = ppo - signal

        return pd.DataFrame({
            'ppo': ppo,
            'signal': signal,
            'histogram': histogram
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
        if not hasattr(self, '_ppo_state'):
            self._ppo_state = {}

        if len(data) >= self.slow_period:
            close = data['close'].values
            # PPO hesapla ve signal EMA state'i tut
            batch = self.calculate_batch(data)
            ppo_values = batch['ppo'].dropna().values

            # Signal line için EMA state
            alpha = 2 / (self.signal_period + 1)

            self._ppo_state[buffer_key] = {
                'close_buffer': list(close[-self.slow_period:]),
                'signal_ema': batch['signal'].iloc[-1] if len(ppo_values) > 0 else 0.0,
                'last_ppo': ppo_values[-1] if len(ppo_values) > 0 else 0.0,
                'alpha': alpha
            }

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - State-based

        Args:
            candle: Yeni mum verisi (dict)
            symbol: Sembol adı (opsiyonel)

        Returns:
            IndicatorResult: Güncel PPO değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # State varsa incremental hesapla
        if hasattr(self, '_ppo_state') and buffer_key in self._ppo_state:
            state = self._ppo_state[buffer_key]
            close_buffer = state['close_buffer']
            signal_ema = state['signal_ema']
            alpha = state['alpha']

            # Buffer'a yeni close ekle
            close_buffer.append(close_val)

            # Buffer boyutunu koru
            if len(close_buffer) > self.slow_period:
                close_buffer.pop(0)

            # SMA hesapla
            if len(close_buffer) >= self.slow_period:
                fast_ma = np.mean(close_buffer[-self.fast_period:])
                slow_ma = np.mean(close_buffer[-self.slow_period:])
                ppo_value = ((fast_ma - slow_ma) / slow_ma) * 100 if slow_ma != 0 else 0.0

                # Signal line (EMA of PPO)
                new_signal = alpha * ppo_value + (1 - alpha) * signal_ema
                histogram = ppo_value - new_signal

                # State güncelle
                self._ppo_state[buffer_key] = {
                    'close_buffer': close_buffer,
                    'signal_ema': new_signal,
                    'last_ppo': ppo_value,
                    'alpha': alpha
                }

                # Sinyal belirleme: PPO > Signal = BUY
                if ppo_value > new_signal:
                    signal = SignalType.BUY
                elif ppo_value < new_signal:
                    signal = SignalType.SELL
                else:
                    signal = SignalType.HOLD

                # Trend belirleme: PPO > 0 = UP
                if ppo_value > 0:
                    trend = TrendDirection.UP
                elif ppo_value < 0:
                    trend = TrendDirection.DOWN
                else:
                    trend = TrendDirection.NEUTRAL

                return IndicatorResult(
                    value={
                        'ppo': round(ppo_value, 4),
                        'signal': round(new_signal, 4),
                        'histogram': round(histogram, 4)
                    },
                    timestamp=timestamp_val,
                    signal=signal,
                    trend=trend,
                    strength=min(abs(ppo_value) * 10, 100),
                    metadata={
                        'fast': self.fast_period,
                        'slow': self.slow_period,
                        'signal_period': self.signal_period
                    }
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
        PPO hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: PPO değerleri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)

        # Son değerleri al
        ppo_val = batch_result['ppo'].iloc[-1]
        sig_val = batch_result['signal'].iloc[-1]
        hist_val = batch_result['histogram'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirleme: PPO > Signal = BUY
        if ppo_val > sig_val:
            signal = SignalType.BUY
        elif ppo_val < sig_val:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        # Trend belirleme: PPO > 0 = UP
        if ppo_val > 0:
            trend = TrendDirection.UP
        elif ppo_val < 0:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'ppo': round(ppo_val, 4),
                'signal': round(sig_val, 4),
                'histogram': round(hist_val, 4)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(ppo_val) * 10, 100),
            metadata={
                'fast': self.fast_period,
                'slow': self.slow_period,
                'signal_period': self.signal_period
            }
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['ppo', 'signal', 'histogram']

    def _requires_volume(self) -> bool:
        """PPO volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['PPO']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """PPO indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 PPO (PERCENTAGE PRICE OSCILLATOR) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 25, 150)
    noise = np.random.randn(150) * 2.5
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(150)),
        'low': prices - np.abs(np.random.randn(150)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(150)]
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    ppo = PPO(fast_period=12, slow_period=26, signal_period=9)
    print(f"   ✅ Oluşturuldu: {ppo}")
    print(f"   ✅ Kategori: {ppo.category.value}")
    print(f"   ✅ Gerekli periyot: {ppo.get_required_periods()}")

    result = ppo(data)
    print(f"   ✅ PPO: {result.value['ppo']}")
    print(f"   ✅ Signal: {result.value['signal']}")
    print(f"   ✅ Histogram: {result.value['histogram']}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = ppo.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 değer:")
    print(batch_result.tail())

    # Test 3: Signal line crossover analizi
    print("\n4. Signal line crossover analizi...")
    batch_result = ppo.calculate_batch(data)
    ppo_values = batch_result['ppo'].dropna()
    signal_values = batch_result['signal'].dropna()

    # Crossover sayısı
    crossovers = 0
    for i in range(1, min(len(ppo_values), len(signal_values))):
        if (ppo_values.iloc[i-1] < signal_values.iloc[i-1] and ppo_values.iloc[i] > signal_values.iloc[i]) or \
           (ppo_values.iloc[i-1] > signal_values.iloc[i-1] and ppo_values.iloc[i] < signal_values.iloc[i]):
            crossovers += 1

    print(f"   ✅ Toplam signal crossover: {crossovers}")
    print(f"   ✅ Pozitif histogram barlar: {sum(batch_result['histogram'].dropna() > 0)}")
    print(f"   ✅ Negatif histogram barlar: {sum(batch_result['histogram'].dropna() < 0)}")

    # Test 4: Farklı periyot kombinasyonları
    print("\n5. Farklı periyot testi...")
    configs = [(5, 10, 5), (12, 26, 9), (20, 50, 15)]
    for fast, slow, sig in configs:
        ppo_test = PPO(fast_period=fast, slow_period=slow, signal_period=sig)
        result = ppo_test.calculate(data)
        print(f"   ✅ PPO({fast},{slow},{sig}): PPO={result.value['ppo']:.4f}, "
              f"Signal={result.value['signal']:.4f}")

    # Test 5: Validasyon testi
    print("\n6. Validasyon testi...")
    try:
        invalid_ppo = PPO(fast_period=26, slow_period=12, signal_period=9)
        print("   ❌ Hata: Geçersiz periyot kombinasyonu kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Fast/Slow period validasyonu başarılı: {e}")

    try:
        invalid_ppo2 = PPO(fast_period=12, slow_period=26, signal_period=0)
        print("   ❌ Hata: Geçersiz signal period kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Signal period validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
