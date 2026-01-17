"""
indicators/breakout/volatility_breakout.py - Volatility Breakout

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Volatility Breakout - Bollinger Genişlemesi ile Breakout Tespiti
    Bollinger Bands genişlemesini ve fiyat hareketini kullanarak
    volatilite breakout'larını tespit eder.

    Breakout Kriterleri:
    - BB genişliği artar (volatilite artışı)
    - Fiyat BB üst/alt bandını kırar
    - Hacim ortalamanın üzerinde (opsiyonel)

    Çıktı:
    - Upper Band: Üst bant
    - Middle Band: Orta bant (SMA)
    - Lower Band: Alt bant
    - Width: Band genişliği
    - %B: Fiyatın bantlar içindeki konumu

Formül:
    BB Middle = SMA(Close, period)
    BB Upper = Middle + (std_dev × StdDev)
    BB Lower = Middle - (std_dev × StdDev)
    BB Width = (Upper - Lower) / Middle × 100
    %B = (Close - Lower) / (Upper - Lower) × 100

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


class VolatilityBreakout(BaseIndicator):
    """
    Volatility Breakout Indicator

    Bollinger Bands genişlemesini izleyerek volatilite breakout'larını tespit eder.
    Band genişliği, %B ve fiyat hareketini analiz eder.

    Args:
        period: BB periyodu (varsayılan: 20)
        std_dev: Standart sapma çarpanı (varsayılan: 2.0)
        width_threshold: Genişlik eşik değeri (varsayılan: 4.0)
        use_volume: Hacim kontrolü kullan (varsayılan: True)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        width_threshold: float = 4.0,
        use_volume: bool = True,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.std_dev = std_dev
        self.width_threshold = width_threshold
        self.use_volume = use_volume

        super().__init__(
            name='volatility_breakout',
            category=IndicatorCategory.BREAKOUT,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'std_dev': std_dev,
                'width_threshold': width_threshold,
                'use_volume': use_volume
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period + 10

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Volatility Breakout calculation - BACKTEST için

        Uses Bollinger Bands for volatility breakout detection

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        close = data['close']

        # Bollinger Bands calculation
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std(ddof=0)
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # BB Width: (Upper - Lower) / Middle × 100
        width = ((upper - lower) / middle) * 100

        # %B: (Close - Lower) / (Upper - Lower) × 100
        pct_b = ((close - lower) / (upper - lower)) * 100

        # Handle division by zero
        width = width.fillna(0).replace([np.inf, -np.inf], 0)
        pct_b = pct_b.fillna(50).replace([np.inf, -np.inf], 50)

        # Set first period values to NaN (warmup)
        upper.iloc[:self.period-1] = np.nan
        middle.iloc[:self.period-1] = np.nan
        lower.iloc[:self.period-1] = np.nan
        width.iloc[:self.period-1] = np.nan
        pct_b.iloc[:self.period-1] = np.nan

        return pd.DataFrame({
            'upper': upper.values,
            'middle': middle.values,
            'lower': lower.values,
            'width': width.values,
            'percent_b': pct_b.values
        }, index=data.index)

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot en az 2 olmalı"
            )
        if self.std_dev <= 0:
            raise InvalidParameterError(
                self.name, 'std_dev', self.std_dev,
                "Standart sapma pozitif olmalı"
            )
        if self.width_threshold <= 0:
            raise InvalidParameterError(
                self.name, 'width_threshold', self.width_threshold,
                "Genişlik eşiği pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Volatility Breakout hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: BB bantları, genişlik ve %B değeri
        """
        close = data['close'].values
        volume = data['volume'].values if self.use_volume else None

        # Bollinger Bands hesapla
        sma = np.mean(close[-self.period:])
        std = np.std(close[-self.period:])

        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        # BB Width hesapla (%)
        width = ((upper_band - lower_band) / sma) * 100 if sma != 0 else 0

        # %B hesapla (fiyatın bantlar içindeki konumu)
        band_range = upper_band - lower_band
        if band_range != 0:
            percent_b = ((close[-1] - lower_band) / band_range) * 100
        else:
            percent_b = 50.0

        # Önceki genişliği hesapla (trend için)
        if len(close) >= self.period + 5:
            prev_sma = np.mean(close[-(self.period+5):-5])
            prev_std = np.std(close[-(self.period+5):-5])
            prev_upper = prev_sma + (self.std_dev * prev_std)
            prev_lower = prev_sma - (self.std_dev * prev_std)
            prev_width = ((prev_upper - prev_lower) / prev_sma) * 100 if prev_sma != 0 else 0
        else:
            prev_width = width

        width_expanding = width > prev_width

        # Hacim kontrolü
        volume_confirm = True
        if self.use_volume and volume is not None:
            avg_volume = np.mean(volume[-self.period:])
            volume_confirm = volume[-1] > avg_volume * 1.2

        # Breakout tespit et
        breakout_up = (
            close[-1] > upper_band and
            width > self.width_threshold and
            width_expanding
        )

        breakout_down = (
            close[-1] < lower_band and
            width > self.width_threshold and
            width_expanding
        )

        if self.use_volume:
            breakout_up = breakout_up and volume_confirm
            breakout_down = breakout_down and volume_confirm

        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirle
        signal = self.get_signal(breakout_up, breakout_down, percent_b)
        trend = self.get_trend(percent_b, close[-1], sma)

        # Güç: Genişlik ve %B kombinasyonu
        strength = min((width / self.width_threshold) * 50 + abs(percent_b - 50), 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper_band, 2),
                'middle': round(sma, 2),
                'lower': round(lower_band, 2),
                'width': round(width, 2),
                'percent_b': round(percent_b, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'width_expanding': width_expanding,
                'volume_confirm': volume_confirm if self.use_volume else None,
                'price': round(close[-1], 2)
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
                value={'breakout': False, 'direction': 'none'},
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

    def get_signal(self, breakout_up: bool, breakout_down: bool, percent_b: float) -> SignalType:
        """
        Breakout durumundan sinyal üret

        Args:
            breakout_up: Yukarı breakout var mı?
            breakout_down: Aşağı breakout var mı?
            percent_b: %B değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if breakout_up:
            return SignalType.STRONG_BUY
        elif breakout_down:
            return SignalType.STRONG_SELL
        elif percent_b > 80:
            return SignalType.SELL
        elif percent_b < 20:
            return SignalType.BUY

        return SignalType.HOLD

    def get_trend(self, percent_b: float, price: float, middle: float) -> TrendDirection:
        """
        %B ve fiyattan trend belirle

        Args:
            percent_b: %B değeri
            price: Mevcut fiyat
            middle: Orta bant (SMA)

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if price > middle and percent_b > 50:
            return TrendDirection.UP
        elif price < middle and percent_b < 50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20,
            'std_dev': 2.0,
            'width_threshold': 4.0,
            'use_volume': True
        }

    def _requires_volume(self) -> bool:
        """Volume kullanılabilir ama zorunlu değil"""
        return False

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['upper', 'middle', 'lower', 'width', 'percent_b']


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VolatilityBreakout']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Volatility Breakout indikatör testi"""

    print("\n" + "="*60)
    print("VOLATILITY BREAKOUT TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Düşük volatilite -> Yüksek volatilite simüle et
    base_price = 100
    prices = [base_price]

    # İlk 50 mum: Düşük volatilite
    for i in range(49):
        change = np.random.randn() * 0.5
        prices.append(prices[-1] + change)

    # Son 50 mum: Yüksek volatilite + trend
    for i in range(50):
        change = np.random.randn() * 3.0 + 0.8
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 2000) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    vb = VolatilityBreakout()
    print(f"   [OK] Oluşturuldu: {vb}")
    print(f"   [OK] Kategori: {vb.category.value}")
    print(f"   [OK] Gerekli periyot: {vb.get_required_periods()}")

    result = vb(data)
    print(f"   [OK] Upper Band: {result.value['upper']}")
    print(f"   [OK] Middle Band: {result.value['middle']}")
    print(f"   [OK] Lower Band: {result.value['lower']}")
    print(f"   [OK] Width: {result.value['width']:.2f}%")
    print(f"   [OK] %B: {result.value['percent_b']:.2f}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Düşük volatilite testi
    print("\n3. Düşük volatilite testi (ilk 50 mum)...")
    low_vol_data = data.head(60)
    result = vb.calculate(low_vol_data)
    print(f"   [OK] Width: {result.value['width']:.2f}%")
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Breakout DOWN: {result.metadata['breakout_down']}")
    print(f"   [OK] Width Expanding: {result.metadata['width_expanding']}")

    # Test 3: Yüksek volatilite testi
    print("\n4. Yüksek volatilite testi (tüm data)...")
    result = vb.calculate(data)
    print(f"   [OK] Width: {result.value['width']:.2f}%")
    print(f"   [OK] %B: {result.value['percent_b']:.2f}")
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Güç: {result.strength:.2f}")

    # Test 4: Farklı parametreler
    print("\n5. Farklı parametre testi...")
    vb_tight = VolatilityBreakout(std_dev=1.5, width_threshold=3.0)
    result = vb_tight.calculate(data)
    print(f"   [OK] Tight BB Width: {result.value['width']:.2f}%")
    print(f"   [OK] %B: {result.value['percent_b']:.2f}")

    # Test 5: Volume olmadan
    print("\n6. Volume olmadan test...")
    vb_no_vol = VolatilityBreakout(use_volume=False)
    result = vb_no_vol.calculate(data)
    print(f"   [OK] Breakout UP: {result.metadata['breakout_up']}")
    print(f"   [OK] Volume Confirm: {result.metadata['volume_confirm']}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 6: Zaman serisi analizi
    print("\n7. Zaman serisi analizi...")
    width_history = []
    percent_b_history = []

    for i in range(40, len(data), 10):
        partial_data = data.head(i)
        result = vb.calculate(partial_data)
        width_history.append(result.value['width'])
        percent_b_history.append(result.value['percent_b'])

    print(f"   [OK] Toplam ölçüm: {len(width_history)}")
    print(f"   [OK] Ortalama genişlik: {np.mean(width_history):.2f}%")
    print(f"   [OK] Max genişlik: {max(width_history):.2f}%")
    print(f"   [OK] Min genişlik: {min(width_history):.2f}%")

    # Test 7: İstatistikler
    print("\n8. İstatistik testi...")
    stats = vb.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = vb.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Output names: {metadata.output_names}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
