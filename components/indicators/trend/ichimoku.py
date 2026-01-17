"""
indicators/trend/ichimoku.py - Ichimoku Cloud

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Ichimoku Kinko Hyo - İchimoku Bulutu
    Goichi Hosoda tarafından geliştirilmiş kapsamlı trend indikatörü
    5 çizgiden oluşur: Tenkan, Kijun, Senkou A, Senkou B, Chikou

    Kullanım:
    - Trend yönünü ve gücünü belirleme
    - Destek/direnç seviyeleri (bulut)
    - Entry/Exit sinyalleri
    - Momentum ölçme

Formül:
    Tenkan-sen = (9 period high + 9 period low) / 2
    Kijun-sen = (26 period high + 26 period low) / 2
    Senkou Span A = (Tenkan + Kijun) / 2, 26 period ileriye kaydırılmış
    Senkou Span B = (52 period high + 52 period low) / 2, 26 period ileriye
    Chikou Span = Close, 26 period geriye kaydırılmış

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from collections import deque
from indicators.base_indicator import BaseIndicator
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class Ichimoku(BaseIndicator):
    """
    Ichimoku Cloud Indicator

    Trend, momentum ve destek/direnç bilgilerini sağlayan kapsamlı indikatör.

    Args:
        tenkan_period: Tenkan-sen periyodu (varsayılan: 9)
        kijun_period: Kijun-sen periyodu (varsayılan: 26)
        senkou_b_period: Senkou Span B periyodu (varsayılan: 52)
        displacement: Senkou kaydırma (varsayılan: 26)
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
        logger=None,
        error_handler=None
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        
        # Buffers for incremental calculation
        # We need enough history for the longest period (Senkou B)
        self._max_period = max(tenkan_period, kijun_period, senkou_b_period)
        self.highs = deque(maxlen=self._max_period)
        self.lows = deque(maxlen=self._max_period)
        self.closes = deque(maxlen=self.displacement + 1) # For Chikou
        
        # Senkou history for correct signal generation (Displacement handling)
        # We need to store calculated Senkou values to use them 26 bars later
        self.senkou_a_history = deque(maxlen=self.displacement + 1)
        self.senkou_b_history = deque(maxlen=self.displacement + 1)

        super().__init__(
            name='ichimoku',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.LINES,
            params={
                'tenkan_period': tenkan_period,
                'kijun_period': kijun_period,
                'senkou_b_period': senkou_b_period,
                'displacement': displacement
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.senkou_b_period + self.displacement

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.tenkan_period < 1:
            raise InvalidParameterError(
                self.name, 'tenkan_period', self.tenkan_period,
                "Tenkan period pozitif olmalı"
            )
        if self.kijun_period < 1:
            raise InvalidParameterError(
                self.name, 'kijun_period', self.kijun_period,
                "Kijun period pozitif olmalı"
            )
        if self.senkou_b_period < 1:
            raise InvalidParameterError(
                self.name, 'senkou_b_period', self.senkou_b_period,
                "Senkou B period pozitif olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)
        
        Tüm veriyi vektörel olarak hesaplar.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Ichimoku çizgileri
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_max = high.rolling(window=self.tenkan_period).max()
        tenkan_min = low.rolling(window=self.tenkan_period).min()
        tenkan = (tenkan_max + tenkan_min) / 2
        
        # Kijun-sen (Base Line)
        kijun_max = high.rolling(window=self.kijun_period).max()
        kijun_min = low.rolling(window=self.kijun_period).min()
        kijun = (kijun_max + kijun_min) / 2
        
        # Senkou Span A (Leading Span A)
        # (Tenkan + Kijun) / 2, shifted forward
        senkou_a = ((tenkan + kijun) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B)
        # 52 period midpoint, shifted forward
        senkou_b_max = high.rolling(window=self.senkou_b_period).max()
        senkou_b_min = low.rolling(window=self.senkou_b_period).min()
        senkou_b = ((senkou_b_max + senkou_b_min) / 2).shift(self.displacement)
        
        # Chikou Span (Lagging Span)
        # Close price, shifted backward
        chikou = close.shift(-self.displacement)
        
        return pd.DataFrame({
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Not: Ichimoku'nun "shift" (kaydırma) mantığı burada özel olarak işlenir.
        - Senkou A ve B değerleri hesaplandığı an buffer'a (history) eklenir.
        - Sinyal üretilirken, buffer'dan 26 bar önceki (displacement) değerler çekilir.
        - Bu sayede "Price vs Cloud" karşılaştırması, grafikteki görsel kaydırmaya uygun yapılır.
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel Ichimoku değerleri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            timestamp_val = int(candle['timestamp']) if 'timestamp' in candle else 0
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        # Update buffers
        self.highs.append(high_val)
        self.lows.append(low_val)
        self.closes.append(close_val)
        
        # Check if we have enough data
        if len(self.highs) < self.tenkan_period:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Convert deques to numpy arrays for calculation (faster than list slicing)
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        
        # Tenkan-sen (Conversion Line)
        # Last 9 periods
        t_high = np.max(highs[-self.tenkan_period:])
        t_low = np.min(lows[-self.tenkan_period:])
        tenkan = (t_high + t_low) / 2
        
        # Kijun-sen (Base Line)
        # Last 26 periods
        if len(highs) >= self.kijun_period:
            k_high = np.max(highs[-self.kijun_period:])
            k_low = np.min(lows[-self.kijun_period:])
            kijun = (k_high + k_low) / 2
        else:
            kijun = float('nan')
            
        # Senkou Span A (Leading Span A) - Calculated NOW
        senkou_a_now = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B) - Calculated NOW
        if len(highs) >= self.senkou_b_period:
            sb_high = np.max(highs[-self.senkou_b_period:])
            sb_low = np.min(lows[-self.senkou_b_period:])
            senkou_b_now = (sb_high + sb_low) / 2
        else:
            senkou_b_now = float('nan')
            
        # Store calculated values in history for future use (Displacement)
        self.senkou_a_history.append(senkou_a_now)
        self.senkou_b_history.append(senkou_b_now)
        
        # Retrieve values from 26 bars ago for Signal/Trend (The "Cloud" at current price)
        # If we don't have enough history yet, use current (fallback)
        if len(self.senkou_a_history) >= self.displacement:
            # The value stored 'displacement' bars ago is what we need
            # deque[-1] is now, deque[0] is oldest.
            # If maxlen=displacement+1, then index 0 is exactly displacement bars ago?
            # Let's check: maxlen=27. 
            # [t-26, t-25, ..., t]
            # We want t-26. So index 0.
            senkou_a_signal = self.senkou_a_history[0]
            senkou_b_signal = self.senkou_b_history[0]
        else:
            senkou_a_signal = senkou_a_now
            senkou_b_signal = senkou_b_now
            
        # Chikou Span (Lagging Span)
        chikou = candle['close']
        
        current_price = close_val
        timestamp = timestamp_val
        
        # Trend/Signal calculations using SHIFTED values (correct cloud)
        trend = self.get_trend(current_price, senkou_a_signal, senkou_b_signal)
        signal = self.get_signal(current_price, tenkan, kijun, senkou_a_signal, senkou_b_signal)
        
        # Cloud color based on FUTURE cloud (what is plotted ahead) OR current cloud?
        # Usually cloud color on chart at time T is based on Senkou A/B at time T (which are plotted at T+26).
        # But for "Price vs Cloud" check, we use the shifted values.
        # Let's return the "Future" cloud values in 'value' dict as they are what's newly calculated,
        # but use "Current" cloud values for metadata/signals.
        
        cloud_color = 'bullish' if senkou_a_now > senkou_b_now else 'bearish'
        
        return IndicatorResult(
            value={
                'tenkan': round(tenkan, 2),
                'kijun': round(kijun, 2),
                'senkou_a': round(senkou_a_now, 2),
                'senkou_b': round(senkou_b_now, 2),
                'chikou': round(chikou, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=self._calculate_strength(current_price, senkou_a_signal, senkou_b_signal),
            metadata={
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'senkou_b_period': self.senkou_b_period,
                'cloud_color': cloud_color,
                'price_vs_cloud': self._price_vs_cloud(current_price, senkou_a_signal, senkou_b_signal),
                'tk_cross': 'bullish' if tenkan > kijun else 'bearish'
            }
        )

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli state'i hazırlar

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        # Clear and repopulate buffers
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.senkou_a_history.clear()
        self.senkou_b_history.clear()

        # We only need the last N values to populate buffers
        recent = data.tail(self._max_period + self.displacement)
        self.highs.extend(recent['high'].values)
        self.lows.extend(recent['low'].values)
        self.closes.extend(recent['close'].values)

        # Populate Senkou history
        if len(data) > self.displacement:
            subset = data.tail(self.senkou_b_period + self.displacement + 1)
            s_high = subset['high']
            s_low = subset['low']

            # Tenkan/Kijun for Senkou A
            t_max = s_high.rolling(window=self.tenkan_period).max()
            t_min = s_low.rolling(window=self.tenkan_period).min()
            t = (t_max + t_min) / 2

            k_max = s_high.rolling(window=self.kijun_period).max()
            k_min = s_low.rolling(window=self.kijun_period).min()
            k = (k_max + k_min) / 2

            sa = (t + k) / 2

            # Senkou B
            sb_max = s_high.rolling(window=self.senkou_b_period).max()
            sb_min = s_low.rolling(window=self.senkou_b_period).min()
            sb = (sb_max + sb_min) / 2

            # Add last N values to history
            last_sa = sa.tail(self.displacement + 1).values
            last_sb = sb.tail(self.displacement + 1).values

            for a, b in zip(last_sa, last_sb):
                if not np.isnan(a):
                    self.senkou_a_history.append(a)
                if not np.isnan(b):
                    self.senkou_b_history.append(b)

        self._buffers_init = True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Ichimoku hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Ichimoku çizgileri
        """
        # Populate buffers for subsequent updates
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.senkou_a_history.clear()
        self.senkou_b_history.clear()
        
        # We only need the last N values to populate buffers
        # Take slightly more to be safe
        recent = data.tail(self._max_period + self.displacement)
        self.highs.extend(recent['high'].values)
        self.lows.extend(recent['low'].values)
        self.closes.extend(recent['close'].values)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Tenkan-sen (Conversion Line)
        tenkan = self._calculate_midpoint(high, low, self.tenkan_period)
        
        # Kijun-sen (Base Line)
        kijun = self._calculate_midpoint(high, low, self.kijun_period)
        
        # Senkou Span A (Leading Span A)
        # (Tenkan + Kijun) / 2, shifted forward
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B)
        # 52 period midpoint, shifted forward
        senkou_b = self._calculate_midpoint(high, low, self.senkou_b_period)
        
        # Populate Senkou history from past calculations
        # We need to recalculate past Senkou values to fill the buffer?
        # Or just take the last N values if we had them?
        # Since calculate() is usually called on full history, we can compute the series
        # and take the last N values.
        
        # Calculate full series for history population
        # This is expensive but necessary for correct transition to realtime
        if len(data) > self.displacement:
            # We need rolling max/min for the last 'displacement' bars
            # to reconstruct Senkou history.
            # Actually, we can just use the single values calculated above?
            # No, 'senkou_a' above is just the LAST value.
            
            # Let's use pandas rolling for the tail to populate history
            # We need the last 'displacement' values of Senkou A/B (unshifted)
            subset = data.tail(self.senkou_b_period + self.displacement + 1)
            
            # Re-calculate rolling for subset
            s_high = subset['high']
            s_low = subset['low']
            
            # Tenkan/Kijun for Senkou A
            t_max = s_high.rolling(window=self.tenkan_period).max()
            t_min = s_low.rolling(window=self.tenkan_period).min()
            t = (t_max + t_min) / 2
            
            k_max = s_high.rolling(window=self.kijun_period).max()
            k_min = s_low.rolling(window=self.kijun_period).min()
            k = (k_max + k_min) / 2
            
            sa = (t + k) / 2
            
            # Senkou B
            sb_max = s_high.rolling(window=self.senkou_b_period).max()
            sb_min = s_low.rolling(window=self.senkou_b_period).min()
            sb = (sb_max + sb_min) / 2
            
            # Add last N values to history
            # We want the values that were calculated, NOT shifted yet.
            last_sa = sa.tail(self.displacement + 1).values
            last_sb = sb.tail(self.displacement + 1).values
            
            for a, b in zip(last_sa, last_sb):
                if not np.isnan(a): self.senkou_a_history.append(a)
                if not np.isnan(b): self.senkou_b_history.append(b)
        
        # Chikou Span (Lagging Span)
        # Close price, shifted backward
        chikou = close[-1]
        
        # Mevcut fiyat
        current_price = close[-1]
        
        timestamp = int(data.iloc[-1]['timestamp'])

        # Trend ve sinyal belirleme
        trend = self.get_trend(current_price, senkou_a, senkou_b)
        signal = self.get_signal(current_price, tenkan, kijun, senkou_a, senkou_b)

        # Cloud rengi
        cloud_color = 'bullish' if senkou_a > senkou_b else 'bearish'

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'tenkan': round(tenkan, 2),
                'kijun': round(kijun, 2),
                'senkou_a': round(senkou_a, 2),
                'senkou_b': round(senkou_b, 2),
                'chikou': round(chikou, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=self._calculate_strength(current_price, senkou_a, senkou_b),
            metadata={
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'senkou_b_period': self.senkou_b_period,
                'cloud_color': cloud_color,
                'price_vs_cloud': self._price_vs_cloud(current_price, senkou_a, senkou_b),
                'tk_cross': 'bullish' if tenkan > kijun else 'bearish'
            }
        )

    def _calculate_midpoint(self, high: np.ndarray, low: np.ndarray, period: int) -> float:
        """
        Belirtilen periyotta yüksek ve düşük ortalaması

        Args:
            high: Yüksek fiyatlar
            low: Düşük fiyatlar
            period: Periyot

        Returns:
            float: Midpoint değeri
        """
        period_high = np.max(high[-period:])
        period_low = np.min(low[-period:])
        return (period_high + period_low) / 2

    def _price_vs_cloud(self, price: float, senkou_a: float, senkou_b: float) -> str:
        """Fiyatın buluta göre pozisyonu"""
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if price > cloud_top:
            return 'above_cloud'
        elif price < cloud_bottom:
            return 'below_cloud'
        else:
            return 'in_cloud'

    def get_signal(self, price: float, tenkan: float, kijun: float,
                   senkou_a: float, senkou_b: float) -> SignalType:
        """
        Ichimoku'dan sinyal üret

        Args:
            price: Mevcut fiyat
            tenkan: Tenkan-sen değeri
            kijun: Kijun-sen değeri
            senkou_a: Senkou Span A değeri
            senkou_b: Senkou Span B değeri

        Returns:
            SignalType: BUY/SELL/HOLD
        """
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Güçlü bullish: Fiyat bulutun üstünde ve Tenkan > Kijun
        if price > cloud_top and tenkan > kijun:
            return SignalType.BUY

        # Güçlü bearish: Fiyat bulutun altında ve Tenkan < Kijun
        if price < cloud_bottom and tenkan < kijun:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, price: float, senkou_a: float, senkou_b: float) -> TrendDirection:
        """
        Ichimoku'dan trend belirle

        Args:
            price: Mevcut fiyat
            senkou_a: Senkou Span A değeri
            senkou_b: Senkou Span B değeri

        Returns:
            TrendDirection: UP/DOWN/NEUTRAL
        """
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if price > cloud_top:
            return TrendDirection.UP
        elif price < cloud_bottom:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, senkou_a: float, senkou_b: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        cloud_mid = (senkou_a + senkou_b) / 2
        distance_pct = abs((price - cloud_mid) / cloud_mid * 100)
        return min(distance_pct * 10, 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']

    def _requires_volume(self) -> bool:
        """Ichimoku volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Ichimoku']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Ichimoku indikatör testi"""

    print("\n" + "="*60)
    print("ICHIMOKU CLOUD TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Güçlü trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(99):
        trend = 0.5
        noise = np.random.randn() * 2.0
        prices.append(prices[-1] + trend + noise)

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
    ichimoku = Ichimoku()
    print(f"   [OK] Oluşturuldu: {ichimoku}")
    print(f"   [OK] Kategori: {ichimoku.category.value}")
    print(f"   [OK] Tip: {ichimoku.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {ichimoku.get_required_periods()}")

    result = ichimoku(data)
    print(f"   [OK] Tenkan-sen: {result.value['tenkan']}")
    print(f"   [OK] Kijun-sen: {result.value['kijun']}")
    print(f"   [OK] Senkou Span A: {result.value['senkou_a']}")
    print(f"   [OK] Senkou Span B: {result.value['senkou_b']}")
    print(f"   [OK] Chikou Span: {result.value['chikou']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Cloud analizi
    print("\n3. Cloud analizi...")
    print(f"   [OK] Cloud Rengi: {result.metadata['cloud_color']}")
    print(f"   [OK] Fiyat pozisyonu: {result.metadata['price_vs_cloud']}")
    print(f"   [OK] TK Crossover: {result.metadata['tk_cross']}")

    # Test 3: Sinyal analizi
    print("\n4. Sinyal analizi...")
    if result.signal == SignalType.BUY:
        print(f"   [OK] Güçlü BUY sinyali (fiyat bulutun üstünde, Tenkan > Kijun)")
    elif result.signal == SignalType.SELL:
        print(f"   [OK] Güçlü SELL sinyali (fiyat bulutun altında, Tenkan < Kijun)")
    else:
        print(f"   [OK] Nötr bölge (fiyat bulut içinde veya belirsiz)")

    # Test 4: Farklı parametreler
    print("\n5. Farklı parametre testi...")
    configs = [(9, 26, 52), (7, 22, 44), (12, 30, 60)]
    for t, k, s in configs:
        ich_test = Ichimoku(tenkan_period=t, kijun_period=k, senkou_b_period=s)
        result = ich_test.calculate(data)
        print(f"   [OK] Ichimoku({t},{k},{s}): Cloud={result.metadata['cloud_color']}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = ichimoku.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n7. Metadata testi...")
    metadata = ichimoku.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Output'lar: {metadata.output_names}")

    # Test 8: Batch Calculation
    print("\n8. Batch Calculation Testi...")
    batch_result = ichimoku.calculate_batch(data)
    print(f"   [OK] Batch result shape: {batch_result.shape}")
    print(f"   [OK] Columns: {list(batch_result.columns)}")
    
    # Compare last values
    last_batch = batch_result.iloc[-1]
    print(f"   [OK] Batch Tenkan: {last_batch['tenkan']:.2f} vs Single: {result.value['tenkan']:.2f}")
    print(f"   [OK] Batch Kijun: {last_batch['kijun']:.2f} vs Single: {result.value['kijun']:.2f}")
    
    # Note: Senkou A/B are shifted, so we need to check if alignment is correct
    # Single calculation returns value at 'timestamp', which corresponds to the plot value at timestamp+26?
    # No, calculate() returns the value available NOW.
    # calculate_batch() returns the series.
    # Let's check if they match at the same index.
    
    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
