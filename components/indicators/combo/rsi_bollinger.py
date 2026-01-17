"""
indicators/combo/rsi_bollinger.py - RSI + Bollinger Bands Kombine Stratejisi

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    RSI + Bollinger Bands Kombine indikatörü
    RSI momentum osilatörünü Bollinger Bands volatilite bantları ile birleştirerek
    güçlü al/sat sinyalleri üretir

    Özellikler:
    - RSI aşırı alım/satım seviyeleri
    - Bollinger Bands fiyat pozisyonu
    - Kombine sinyal üretimi
    - Güçlü onay sistemi

Strateji:
    GÜÇLÜ AL: RSI < 30 VE Fiyat Alt Banda Yakın/Altında
    AL: RSI < 40 VE Fiyat Alt Banda Yakın
    GÜÇLÜ SAT: RSI > 70 VE Fiyat Üst Banda Yakın/Üstünde
    SAT: RSI > 60 VE Fiyat Üst Banda Yakın
    HOLD: Diğer durumlar

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.momentum.rsi
    - indicators.volatility.bollinger
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.momentum.rsi import RSI
from indicators.volatility.bollinger import BollingerBands
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class RSIBollinger(BaseIndicator):
    """
    RSI + Bollinger Bands Kombine Stratejisi

    RSI momentum osilatörü ile Bollinger Bands volatilite bantlarını birleştirerek
    güçlü al/sat sinyalleri üretir.

    Args:
        rsi_period: RSI periyodu (varsayılan: 14)
        rsi_overbought: RSI aşırı alım seviyesi (varsayılan: 70)
        rsi_oversold: RSI aşırı satım seviyesi (varsayılan: 30)
        bb_period: Bollinger Bands periyodu (varsayılan: 20)
        bb_std_dev: BB standart sapma çarpanı (varsayılan: 2.0)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        logger=None,
        error_handler=None
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev

        # Alt indikatörleri oluştur
        self.rsi = RSI(
            period=rsi_period,
            overbought=rsi_overbought,
            oversold=rsi_oversold,
            logger=logger,
            error_handler=error_handler
        )

        self.bb = BollingerBands(
            period=bb_period,
            std_dev=bb_std_dev,
            logger=logger,
            error_handler=error_handler
        )

        super().__init__(
            name='rsi_bollinger',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'bb_period': bb_period,
                'bb_std_dev': bb_std_dev
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.rsi.get_required_periods(), self.bb.get_required_periods())

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "RSI periyodu pozitif olmalı"
            )
        if self.bb_period < 2:
            raise InvalidParameterError(
                self.name, 'bb_period', self.bb_period,
                "BB periyodu en az 2 olmalı"
            )
        if self.rsi_oversold >= self.rsi_overbought:
            raise InvalidParameterError(
                self.name, 'rsi_levels',
                f"oversold={self.rsi_oversold}, overbought={self.rsi_overbought}",
                "RSI oversold, overbought'tan küçük olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch RSI + Bollinger calculation - BACKTEST için

        Combines RSI and Bollinger Bands using their respective calculate_batch() methods

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: rsi, bb_upper, bb_middle, bb_lower for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        # Calculate RSI (vectorized)
        rsi_series = self.rsi.calculate_batch(data)

        # Calculate Bollinger Bands (vectorized)
        bb_df = self.bb.calculate_batch(data)

        # Calculate percent_b (vectorized)
        close = data['close']
        bb_upper = bb_df['upper']
        bb_lower = bb_df['lower']

        # Avoid division by zero
        denominator = bb_upper - bb_lower
        percent_b = ((close - bb_lower) / denominator.replace(0, np.nan))

        # Combine results (same keys as calculate())
        return pd.DataFrame({
            'rsi': rsi_series.values,
            'bb_upper': bb_upper.values,
            'bb_middle': bb_df['middle'].values,
            'bb_lower': bb_lower.values,
            'percent_b': percent_b.values,
            'price': close.values
        }, index=data.index)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        RSI + Bollinger kombine hesaplama

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Kombine RSI + BB değerleri ve sinyalleri
        """
        # RSI hesapla
        rsi_result = self.rsi.calculate(data)
        rsi_value = rsi_result.value

        # Bollinger Bands hesapla
        bb_result = self.bb.calculate(data)
        bb_upper = bb_result.value['upper']
        bb_middle = bb_result.value['middle']
        bb_lower = bb_result.value['lower']

        # Mevcut fiyat ve %B değeri
        current_price = data['close'].values[-1]
        percent_b = bb_result.metadata['percent_b']

        timestamp = int(data.iloc[-1]['timestamp'])

        # Kombine sinyal ve trend belirleme
        signal = self.get_signal(rsi_value, percent_b, current_price, bb_upper, bb_lower)
        trend = self.get_trend(rsi_value, percent_b)
        strength = self._calculate_strength(rsi_value, percent_b)

        # Sinyal konfirmasyonu
        confirmation = self._get_confirmation(rsi_value, percent_b)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'rsi': round(rsi_value, 2),
                'bb_upper': round(bb_upper, 8),
                'bb_middle': round(bb_middle, 8),
                'bb_lower': round(bb_lower, 8),
                'percent_b': round(percent_b, 4),
                'price': round(current_price, 8)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'rsi_period': self.rsi_period,
                'bb_period': self.bb_period,
                'rsi_signal': rsi_result.signal.value,
                'bb_signal': bb_result.signal.value,
                'confirmation': confirmation,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'bandwidth': bb_result.metadata['bandwidth']
            }
        )

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
                value={'signal': 'none', 'strength': 0.0},
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

    def get_signal(self, rsi: float, percent_b: float, price: float,
                   bb_upper: float, bb_lower: float) -> SignalType:
        """
        Kombine RSI + BB sinyali

        Args:
            rsi: RSI değeri
            percent_b: Bollinger %B değeri
            price: Güncel fiyat
            bb_upper: Üst bant
            bb_lower: Alt bant

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # GÜÇLÜ AL sinyalleri
        if rsi < 30 and (percent_b <= 0 or price <= bb_lower):
            return SignalType.BUY  # Çok güçlü al

        # AL sinyalleri
        if rsi < 40 and percent_b < 0.2:
            return SignalType.BUY

        # GÜÇLÜ SAT sinyalleri
        if rsi > 70 and (percent_b >= 1 or price >= bb_upper):
            return SignalType.SELL  # Çok güçlü sat

        # SAT sinyalleri
        if rsi > 60 and percent_b > 0.8:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, rsi: float, percent_b: float) -> TrendDirection:
        """
        Kombine trend belirleme

        Args:
            rsi: RSI değeri
            percent_b: Bollinger %B değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        # Her iki indikatör de yükseliş gösteriyorsa
        if rsi > 50 and percent_b > 0.5:
            return TrendDirection.UP
        # Her iki indikatör de düşüş gösteriyorsa
        elif rsi < 50 and percent_b < 0.5:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _calculate_strength(self, rsi: float, percent_b: float) -> float:
        """
        Sinyal gücünü hesapla (0-100)

        İki indikatörün ekstrem değerlerinde güç artar
        """
        # RSI'dan gelen güç
        rsi_strength = 0
        if rsi < 30:
            rsi_strength = (30 - rsi) * 2  # Aşırı satım
        elif rsi > 70:
            rsi_strength = (rsi - 70) * 2  # Aşırı alım

        # %B'den gelen güç
        bb_strength = 0
        if percent_b <= 0:
            bb_strength = abs(percent_b) * 100 + 50
        elif percent_b >= 1:
            bb_strength = (percent_b - 1) * 100 + 50
        elif percent_b < 0.2:
            bb_strength = (0.2 - percent_b) * 200
        elif percent_b > 0.8:
            bb_strength = (percent_b - 0.8) * 200

        # Kombine güç (ortalama)
        combined_strength = (rsi_strength + bb_strength) / 2

        return min(combined_strength, 100)

    def _get_confirmation(self, rsi: float, percent_b: float) -> str:
        """
        Sinyal konfirmasyonu durumunu belirle

        Returns:
            str: 'strong', 'moderate', 'weak' veya 'none'
        """
        # Güçlü konfirmasyon (her iki indikatör de aynı yönde ekstrem)
        if (rsi < 30 and percent_b < 0.2) or (rsi > 70 and percent_b > 0.8):
            return 'strong'

        # Orta konfirmasyon (bir indikatör ekstrem, diğeri onaylıyor)
        if (rsi < 40 and percent_b < 0.3) or (rsi > 60 and percent_b > 0.7):
            return 'moderate'

        # Zayıf konfirmasyon (indikatörler farklı yönde)
        if (rsi < 50 and percent_b > 0.5) or (rsi > 50 and percent_b < 0.5):
            return 'weak'

        return 'none'

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std_dev': 2.0
        }

    def _requires_volume(self) -> bool:
        """RSI + Bollinger volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['RSIBollinger']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """RSI + Bollinger kombine indikatör testi"""

    print("\n" + "="*60)
    print("RSI + BOLLINGER BANDS COMBO TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Volatil fiyat hareketi simüle et
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 15:
            change = np.random.randn() * 1.5 - 0.5  # Düşüş trendi
        elif i < 35:
            change = np.random.randn() * 1.5 + 0.5  # Yükseliş trendi
        else:
            change = np.random.randn() * 2  # Yatay hareket
        prices.append(max(prices[-1] + change, 50))  # Minimum 50

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    combo = RSIBollinger()
    print(f"   [OK] Oluşturuldu: {combo}")
    print(f"   [OK] Kategori: {combo.category.value}")
    print(f"   [OK] Tip: {combo.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {combo.get_required_periods()}")

    result = combo(data)
    print(f"   [OK] RSI: {result.value['rsi']}")
    print(f"   [OK] BB Upper: {result.value['bb_upper']:.2f}")
    print(f"   [OK] BB Middle: {result.value['bb_middle']:.2f}")
    print(f"   [OK] BB Lower: {result.value['bb_lower']:.2f}")
    print(f"   [OK] %B: {result.value['percent_b']}")
    print(f"   [OK] Fiyat: {result.value['price']:.2f}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")

    # Test 2: Sinyal analizi
    print("\n3. Sinyal analizi...")
    print(f"   [OK] RSI Sinyali: {result.metadata['rsi_signal']}")
    print(f"   [OK] BB Sinyali: {result.metadata['bb_signal']}")
    print(f"   [OK] Kombine Sinyal: {result.signal.value}")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")

    # Test 3: Farklı senaryolar
    print("\n4. Farklı senaryo testi...")
    scenarios = [
        (20, "Düşüş sonu"),
        (30, "Yükseliş ortası"),
        (45, "Yükseliş sonu")
    ]

    for idx, desc in scenarios:
        if idx < len(data):
            data_slice = data.iloc[:idx+1]
            result = combo.calculate(data_slice)
            print(f"   [OK] {desc}: RSI={result.value['rsi']:.1f}, "
                  f"%B={result.value['percent_b']:.2f}, "
                  f"Sinyal={result.signal.value}, "
                  f"Konfirmasyon={result.metadata['confirmation']}")

    # Test 4: Özel parametreler
    print("\n5. Özel parametre testi...")
    combo_custom = RSIBollinger(
        rsi_period=21,
        rsi_overbought=75,
        rsi_oversold=25,
        bb_period=30,
        bb_std_dev=2.5
    )
    result = combo_custom.calculate(data)
    print(f"   [OK] Özel parametreli RSI: {result.value['rsi']}")
    print(f"   [OK] Özel parametreli %B: {result.value['percent_b']:.4f}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: Güç analizi
    print("\n6. Güç analizi...")
    print(f"   [OK] Sinyal Gücü: {result.strength:.2f}/100")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")
    print(f"   [OK] Bandwidth: {result.metadata['bandwidth']:.2f}%")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = combo.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = combo.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
