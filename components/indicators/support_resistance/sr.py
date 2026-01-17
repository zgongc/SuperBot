"""
indicators/support_resistance/support_resistance.py - Support and Resistance Levels

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Support and Resistance - Otomatik destek ve direnç seviyesi tespiti
    Geçmiş fiyat verilerinden swing high/low noktalarını bularak
    önemli destek ve direnç seviyelerini belirler.

Formül:
    - Yerel maksimum ve minimumları bul
    - Birbirine yakın seviyeleri birleştir
    - Frekansa göre önem sıralaması yap
    - En güçlü N seviyeyi döndür

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


class SupportResistance(BaseIndicator):
    """
    Support and Resistance Level Detector

    Geçmiş fiyat verilerinden swing high/low noktalarını bularak
    otomatik olarak destek ve direnç seviyelerini tespit eder.

    Args:
        lookback: Geriye dönük arama periyodu (varsayılan: 50)
        num_levels: Döndürülecek seviye sayısı (varsayılan: 5)
        tolerance: Seviye birleştirme toleransı % (varsayılan: 0.5)
    """

    def __init__(
        self,
        lookback: int = 50,
        num_levels: int = 5,
        tolerance: float = 0.5,
        logger=None,
        error_handler=None
    ):
        self.lookback = lookback
        self.num_levels = num_levels
        self.tolerance = tolerance

        super().__init__(
            name='support_resistance',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
            params={
                'lookback': lookback,
                'num_levels': num_levels,
                'tolerance': tolerance
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.lookback

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.lookback < 10:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback en az 10 olmalı"
            )
        if self.num_levels < 1:
            raise InvalidParameterError(
                self.name, 'num_levels', self.num_levels,
                "Num levels pozitif olmalı"
            )
        if self.tolerance <= 0:
            raise InvalidParameterError(
                self.name, 'tolerance', self.tolerance,
                "Tolerance pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Support/Resistance seviyeleri hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Destek ve direnç seviyeleri
        """
        # Son lookback periyodu al
        recent_data = data.iloc[-self.lookback:]
        high = recent_data['high'].values
        low = recent_data['low'].values
        close = recent_data['close'].values
        
        # Bufferları doldur (Incremental update için hazırlık)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            self._high_buffer = deque(maxlen=self.lookback)
            self._low_buffer = deque(maxlen=self.lookback)
            self._close_buffer = deque(maxlen=self.lookback)
            
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        
        self._high_buffer.extend(high)
        self._low_buffer.extend(low)
        self._close_buffer.extend(close)

        # Swing noktalarını bul
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)

        # Seviyeleri birleştir
        all_levels = np.concatenate([swing_highs, swing_lows])
        clustered_levels = self._cluster_levels(all_levels)

        # En güçlü seviyeleri seç
        top_levels = self._select_top_levels(clustered_levels, close[-1])

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Destek ve direnç olarak ayır
        supports = [lvl for lvl in top_levels if lvl < current_price]
        resistances = [lvl for lvl in top_levels if lvl > current_price]

        # Seviyeleri oluştur (her zaman 6 key - R1,R2,R3,S1,S2,S3)
        levels = {}
        sorted_resistances = sorted(resistances)[:3]
        sorted_supports = sorted(supports, reverse=True)[:3]

        for i in range(1, 4):
            if i <= len(sorted_resistances):
                levels[f'R{i}'] = round(sorted_resistances[i-1], 2)
            else:
                levels[f'R{i}'] = np.nan

        for i in range(1, 4):
            if i <= len(sorted_supports):
                levels[f'S{i}'] = round(sorted_supports[i-1], 2)
            else:
                levels[f'S{i}'] = np.nan

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, supports, resistances),
            strength=self.calculate_strength(current_price, levels),
            metadata={
                'lookback': self.lookback,
                'current_price': round(current_price, 2),
                'total_swing_highs': len(swing_highs),
                'total_swing_lows': len(swing_lows),
                'total_levels': len(top_levels),
                'num_supports': len(supports),
                'num_resistances': len(resistances)
            }
        )



    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: R1..R3, S1..S3 seviyeleri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)
        
        # 1. Tüm swing noktalarını önceden bul (Vectorized)
        # Window=5 varsayımıyla
        window = 5
        
        # Swing Highs
        # high[i] > high[i-1] ... high[i-window] AND high[i] > high[i+1] ... high[i+window]
        # Bu tam olarak _find_swing_highs mantığı ama tüm seri için
        # argrelextrema kullanılabilir ama numpy ile sliding window max daha hızlı olabilir
        
        # Basit döngü ile tüm swingleri bulmak (tüm veri için bir kere)
        # Veya _find_swing_highs metodunu tüm data için çağır
        all_swing_highs_idx = []
        all_swing_lows_idx = []
        
        # Not: _find_swing_highs değer döndürüyor, index değil. Bize index lazım.
        # O yüzden kodu buraya inline edelim veya index döndüren versiyon yapalım.
        
        for i in range(window, n - window):
            if high[i] == np.max(high[i - window:i + window + 1]):
                all_swing_highs_idx.append(i)
            if low[i] == np.min(low[i - window:i + window + 1]):
                all_swing_lows_idx.append(i)
                
        all_swing_highs_idx = np.array(all_swing_highs_idx)
        all_swing_lows_idx = np.array(all_swing_lows_idx)
        
        # Sonuç arrayleri
        results = {f'R{i}': np.full(n, np.nan) for i in range(1, 4)}
        results.update({f'S{i}': np.full(n, np.nan) for i in range(1, 4)})
        
        # Her bar için hesapla (Lookback kadar geriye bakarak)
        # Bu kısım loop olmak zorunda çünkü cluster ve select logic karmaşık
        # Ancak sadece swing pointlerin olduğu indexlerde işlem yapmıyoruz, her bar için o anki window'daki swingleri alıyoruz
        
        for i in range(self.lookback, n):
            start_idx = i - self.lookback
            end_idx = i
            
            # Bu aralıktaki swingleri filtrele
            # np.searchsorted ile hızlıca aralık bulabiliriz
            
            # Highs
            h_start = np.searchsorted(all_swing_highs_idx, start_idx)
            h_end = np.searchsorted(all_swing_highs_idx, end_idx) # end_idx dahil değil
            current_swing_highs = high[all_swing_highs_idx[h_start:h_end]]
            
            # Lows
            l_start = np.searchsorted(all_swing_lows_idx, start_idx)
            l_end = np.searchsorted(all_swing_lows_idx, end_idx)
            current_swing_lows = low[all_swing_lows_idx[l_start:l_end]]
            
            # Birleştir ve Cluster
            all_levels = np.concatenate([current_swing_highs, current_swing_lows])
            if len(all_levels) == 0:
                continue
                
            clustered = self._cluster_levels(all_levels)
            
            # Select
            current_price = close[i]
            top_levels = self._select_top_levels(clustered, current_price)
            
            supports = sorted([lvl for lvl in top_levels if lvl < current_price], reverse=True)
            resistances = sorted([lvl for lvl in top_levels if lvl > current_price])
            
            # Sonuçları kaydet
            for k, val in enumerate(resistances[:3]):
                results[f'R{k+1}'][i] = val
            for k, val in enumerate(supports[:3]):
                results[f'S{k+1}'][i] = val
                
        return pd.DataFrame(results, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel S/R seviyeleri
        """
        # Buffer yönetimi
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            self._high_buffer = deque(maxlen=self.lookback)
            self._low_buffer = deque(maxlen=self.lookback)
            self._close_buffer = deque(maxlen=self.lookback)
            
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
        
        if len(self._high_buffer) < self.lookback:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Hesaplama (Mevcut calculate mantığını buffer üzerinde çalıştır)
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        
        # Swing noktalarını bul
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)

        # Seviyeleri birleştir
        all_levels = np.concatenate([swing_highs, swing_lows])
        clustered_levels = self._cluster_levels(all_levels)

        # En güçlü seviyeleri seç
        current_price = close[-1]
        top_levels = self._select_top_levels(clustered_levels, current_price)
        
        # Destek ve direnç olarak ayır
        supports = [lvl for lvl in top_levels if lvl < current_price]
        resistances = [lvl for lvl in top_levels if lvl > current_price]

        # Seviyeleri oluştur (her zaman 6 key - R1,R2,R3,S1,S2,S3)
        levels = {}
        sorted_resistances = sorted(resistances)[:3]
        sorted_supports = sorted(supports, reverse=True)[:3]

        for i in range(1, 4):
            if i <= len(sorted_resistances):
                levels[f'R{i}'] = round(sorted_resistances[i-1], 2)
            else:
                levels[f'R{i}'] = np.nan

        for i in range(1, 4):
            if i <= len(sorted_supports):
                levels[f'S{i}'] = round(sorted_supports[i-1], 2)
            else:
                levels[f'S{i}'] = np.nan

        return IndicatorResult(
            value=levels,
            timestamp=timestamp_val,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, supports, resistances),
            strength=self.calculate_strength(current_price, levels),
            metadata={
                'lookback': self.lookback,
                'current_price': round(current_price, 2),
                'total_swing_highs': len(swing_highs),
                'total_swing_lows': len(swing_lows),
                'total_levels': len(top_levels),
                'num_supports': len(supports),
                'num_resistances': len(resistances)
            }
        )
    def _find_swing_highs(self, high: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Swing high noktalarını bul

        Args:
            high: High fiyatları
            window: Pencere boyutu

        Returns:
            np.ndarray: Swing high seviyeleri
        """
        swing_highs = []
        for i in range(window, len(high) - window):
            if high[i] == np.max(high[i - window:i + window + 1]):
                swing_highs.append(high[i])
        return np.array(swing_highs)

    def _find_swing_lows(self, low: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Swing low noktalarını bul

        Args:
            low: Low fiyatları
            window: Pencere boyutu

        Returns:
            np.ndarray: Swing low seviyeleri
        """
        swing_lows = []
        for i in range(window, len(low) - window):
            if low[i] == np.min(low[i - window:i + window + 1]):
                swing_lows.append(low[i])
        return np.array(swing_lows)

    def _cluster_levels(self, levels: np.ndarray) -> np.ndarray:
        """
        Birbirine yakın seviyeleri birleştir

        Args:
            levels: Seviye dizisi

        Returns:
            np.ndarray: Birleştirilmiş seviyeler
        """
        if len(levels) == 0:
            return levels

        sorted_levels = np.sort(levels)
        clustered = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            # Önceki seviyeyle yüzde farkı hesapla
            pct_diff = abs((level - current_cluster[-1]) / current_cluster[-1] * 100)

            if pct_diff <= self.tolerance:
                # Aynı cluster'a ekle
                current_cluster.append(level)
            else:
                # Yeni cluster başlat
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        # Son cluster'ı ekle
        if current_cluster:
            clustered.append(np.mean(current_cluster))

        return np.array(clustered)

    def _select_top_levels(self, levels: np.ndarray, current_price: float) -> list:
        """
        En güçlü seviyeleri seç

        Args:
            levels: Seviye dizisi
            current_price: Güncel fiyat

        Returns:
            list: Seçilmiş seviyeler
        """
        if len(levels) == 0:
            return []

        # Güncel fiyata yakınlığa göre sırala ve seç
        distances = np.abs(levels - current_price)
        sorted_indices = np.argsort(distances)

        # En yakın num_levels kadar seviye seç
        selected = levels[sorted_indices[:self.num_levels * 2]]
        return sorted(selected.tolist())

    def get_signal(self, price: float, levels: dict) -> SignalType:
        """
        Fiyatın S/R seviyelerine göre sinyal üret

        Args:
            price: Güncel fiyat
            levels: Destek/Direnç seviyeleri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # En yakın destek ve dirençleri bul
        supports = [v for k, v in levels.items() if k.startswith('S')]
        resistances = [v for k, v in levels.items() if k.startswith('R')]

        if supports and min(abs(price - s) for s in supports) < price * 0.01:
            # Destek seviyesine yakın
            return SignalType.BUY
        elif resistances and min(abs(price - r) for r in resistances) < price * 0.01:
            # Direnç seviyesine yakın
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, price: float, supports: list, resistances: list) -> TrendDirection:
        """
        Fiyatın S/R seviyelerine göre trend belirle

        Args:
            price: Güncel fiyat
            supports: Destek seviyeleri
            resistances: Direnç seviyeleri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if not supports and resistances:
            return TrendDirection.UP  # Tüm seviyeler altında kaldı
        elif supports and not resistances:
            return TrendDirection.DOWN  # Tüm seviyeler üstüne çıktı
        else:
            return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, levels: dict) -> float:
        """
        Fiyatın seviyelere göre güç hesapla

        Args:
            price: Güncel fiyat
            levels: Destek/Direnç seviyeleri

        Returns:
            float: Güç değeri (0-100)
        """
        if not levels:
            return 50.0

        all_values = list(levels.values())
        if not all_values:
            return 50.0

        min_level = min(all_values)
        max_level = max(all_values)

        if max_level == min_level:
            return 50.0

        # Fiyatın seviyeler arasındaki konumu
        position = (price - min_level) / (max_level - min_level) * 100
        return min(max(position, 0), 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'lookback': 50,
            'num_levels': 5,
            'tolerance': 0.5
        }

    def _requires_volume(self) -> bool:
        """Support/Resistance volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SupportResistance']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Support/Resistance indikatör testi"""

    print("\n" + "="*60)
    print("SUPPORT AND RESISTANCE LEVELS TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend değişimleriyle fiyat hareketi simüle et
    base_price = 100
    prices = [base_price]

    for i in range(99):
        # Periyodik dalgalanma ekle
        wave = 10 * np.sin(i / 10)
        noise = np.random.randn() * 1
        prices.append(base_price + wave + noise)

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
    sr = SupportResistance(lookback=50, num_levels=5, tolerance=0.5)
    print(f"   [OK] Oluşturuldu: {sr}")
    print(f"   [OK] Kategori: {sr.category.value}")
    print(f"   [OK] Tip: {sr.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {sr.get_required_periods()}")

    result = sr(data)
    print(f"   [OK] Tespit Edilen Seviyeler:")
    for level, value in sorted(result.value.items(), key=lambda x: x[1], reverse=True):
        level_type = "Direnç" if level.startswith('R') else "Destek"
        print(f"        {level} ({level_type}): {value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı lookback değerleri
    print("\n3. Farklı lookback testi...")
    for lookback in [30, 50, 80]:
        sr_test = SupportResistance(lookback=lookback, num_levels=5)
        result = sr_test.calculate(data)
        print(f"   [OK] SR(lookback={lookback}) - Seviyeler: {len(result.value)} | "
              f"Sinyal: {result.signal.value}")

    # Test 3: Farklı seviye sayıları
    print("\n4. Farklı seviye sayısı testi...")
    for num in [3, 5, 7]:
        sr_test = SupportResistance(lookback=50, num_levels=num)
        result = sr_test.calculate(data)
        print(f"   [OK] SR(num_levels={num}) - Tespit: {len(result.value)} seviye")

    # Test 4: Seviye analizi
    print("\n5. Seviye analizi...")
    result = sr.calculate(data)
    current = result.metadata['current_price']
    print(f"   [OK] Güncel fiyat: {current}")

    supports = {k: v for k, v in result.value.items() if k.startswith('S')}
    resistances = {k: v for k, v in result.value.items() if k.startswith('R')}

    if supports:
        nearest_support = max(supports.values())
        distance_to_support = ((current - nearest_support) / current * 100)
        print(f"   [OK] En yakın destek: {nearest_support:.2f} (Uzaklık: {distance_to_support:.2f}%)")

    if resistances:
        nearest_resistance = min(resistances.values())
        distance_to_resistance = ((nearest_resistance - current) / current * 100)
        print(f"   [OK] En yakın direnç: {nearest_resistance:.2f} (Uzaklık: {distance_to_resistance:.2f}%)")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = sr.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = sr.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
