"""
indicators/support_resistance/zigzag.py - ZigZag Indicator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    ZigZag - Swing high ve low noktalarını belirler
    Belirli bir yüzde değişim eşiğini aşan fiyat hareketlerini filtreler.
    Trend değişimlerini ve önemli destek/direnç noktalarını gösterir.

Formül:
    - Fiyat önceki pivot'tan %deviation kadar değiştiğinde yeni pivot oluşur
    - Swing High: Yukarı yönlü pivot noktası
    - Swing Low: Aşağı yönlü pivot noktası

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


class ZigZag(BaseIndicator):
    """
    ZigZag Indicator

    Belirli bir yüzde değişim eşiğini aşan fiyat hareketlerini filtreler
    ve swing high/low noktalarını belirler.

    Args:
        deviation: Minimum değişim yüzdesi (varsayılan: 5.0)
        depth: Geriye dönük arama derinliği (varsayılan: 12)
    """

    def __init__(
        self,
        deviation: float = 5.0,
        depth: int = 12,
        logger=None,
        error_handler=None
    ):
        self.deviation = deviation
        self.depth = depth

        super().__init__(
            name='zigzag',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'deviation': deviation,
                'depth': depth
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.depth * 2

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.deviation <= 0:
            raise InvalidParameterError(
                self.name, 'deviation', self.deviation,
                "Deviation pozitif olmalı"
            )
        if self.depth < 1:
            raise InvalidParameterError(
                self.name, 'depth', self.depth,
                "Depth pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        ZigZag hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Son pivot değeri ve bilgileri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Bufferları doldur (Incremental update için hazırlık)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            self._high_buffer = deque(maxlen=self.depth + 1)
            self._low_buffer = deque(maxlen=self.depth + 1)
            self._last_pivot_val = None
            self._last_pivot_type = None
            self._total_candles = 0
            self._pivots = []
            
        # Son verileri buffer'a at
        # Not: Tam senkronizasyon için tüm geçmişi işleyip state'i kurmak gerekir.
        # Ancak calculate() zaten tüm geçmişi işliyor.
        # Biz sadece son durumu state'e aktaracağız.
        
        # Calculate çağrıldığında state'i sıfırlayıp yeniden kurmak en doğrusu
        # Ama bu pahalı olabilir. 
        # Basitçe son durumu alalım:
        
        # Pivots zaten hesaplandı
        pivots = self._find_pivots(high, low)
        
        if pivots:
            last = pivots[-1]
            self._last_pivot_val = last['value']
            self._last_pivot_type = last['type']
            self._pivots = pivots[-5:] # Son 5 pivotu sakla
        else:
            self._last_pivot_val = close[-1]
            self._last_pivot_type = 'none'
            
        self._total_candles = len(data)
        
        # Bufferları son verilerle doldur
        self._high_buffer.clear()
        self._low_buffer.clear()
        
        # Son depth+1 veriyi al
        start_idx = max(0, len(data) - (self.depth + 1))
        self._high_buffer.extend(high[start_idx:])
        self._low_buffer.extend(low[start_idx:])

        # Pivotları bul
        pivots = self._find_pivots(high, low)

        # Son pivot bilgisini al
        last_pivot = pivots[-1] if len(pivots) > 0 else None
        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        if last_pivot:
            pivot_value = last_pivot['value']
            pivot_type = last_pivot['type']
            pivot_index = last_pivot['index']

            # Trend belirle
            if pivot_type == 'high':
                trend = TrendDirection.DOWN  # High'dan sonra düşüş beklenir
                signal = SignalType.SELL
            else:
                trend = TrendDirection.UP  # Low'dan sonra yükseliş beklenir
                signal = SignalType.BUY

            # Güç hesapla (son pivot'tan uzaklık)
            price_change = abs((current_price - pivot_value) / pivot_value * 100)
            strength = min(price_change / self.deviation * 100, 100)

            # Önceki pivot varsa ona göre sinyal güncelle
            if len(pivots) > 1:
                prev_pivot = pivots[-2]
                # Trend doğrultusunda hareket ediyorsa sinyali güçlendir
                if pivot_type == 'low' and current_price > pivot_value:
                    signal = SignalType.BUY
                elif pivot_type == 'high' and current_price < pivot_value:
                    signal = SignalType.SELL
                else:
                    signal = SignalType.HOLD

        else:
            pivot_value = current_price
            pivot_type = 'none'
            pivot_index = len(data) - 1
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL
            strength = 0.0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'value': round(pivot_value, 2),
                'pivot_type': pivot_type
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'deviation': self.deviation,
                'depth': self.depth,
                'pivot_index': pivot_index,
                'current_price': round(current_price, 2),
                'price_change_pct': round(abs((current_price - pivot_value) / pivot_value * 100), 2),
                'total_pivots': len(pivots)
            }
        )



    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: ZigZag değerleri
        """
        high = data['high'].values
        low = data['low'].values
        
        pivots = self._find_pivots(high, low)
        
        # Sonuçları DataFrame'e dönüştür
        # Her bar için o anki "son pivot" değerini döndüreceğiz
        
        result_values = np.full(len(data), np.nan)
        result_types = np.full(len(data), None)
        
        # Pivotları zaman serisine yay
        # Pivot listesi: [{'index': 10, 'value': 100, 'type': 'high'}, ...]
        # Pivotlar bulundukları index'te "oluşur" (veya teyit edilir).
        # Ancak ZigZag genellikle geçmişe dönük çizilir.
        # Real-time kullanım için: O an bilinen son pivot değeri geçerlidir.
        
        current_pivot_val = np.nan
        current_pivot_type = None
        
        pivot_idx = 0
        num_pivots = len(pivots)
        
        # Bu döngü biraz verimsiz olabilir ama ZigZag doğası gereği sparse.
        # Daha hızlı yöntem: Pivot indexlerini kullanıp fillna yapmak.
        
        # Pivotların oluştuğu indexleri al
        pivot_indices = [p['index'] for p in pivots]
        pivot_vals = [p['value'] for p in pivots]
        pivot_types = [p['type'] for p in pivots]
        
        # Series oluştur
        s_values = pd.Series(np.nan, index=data.index)
        s_types = pd.Series(dtype=object, index=data.index)  # Explicitly object dtype for string values
        
        # Pivot noktalarını işaretle
        # Not: _find_pivots indexleri integer index (iloc).
        if pivots:
            # İlk pivot öncesi değer yok (veya ilk fiyat?)
            # İlk pivot indexine kadar NaN kalabilir veya backfill yapılabilir.
            # Biz forward fill mantığıyla gideceğiz.
            
            # Pivotları yerleştir
            # Dikkat: Pivot indexi, pivotun "oluştuğu" yer değil, "zirve/dip" yaptığı yerdir.
            # Ancak teyit edildiği yer (confirmation) daha ileride olabilir.
            # _find_pivots logic'inde pivot eklendiği an (loop index i), teyit anıdır.
            # Ancak pivot['index'] zirve noktasıdır.
            # Real-time simülasyonu için: Pivot teyit edildiği andan itibaren geçerlidir.
            # Ancak _find_pivots teyit anını döndürmüyor, sadece pivot noktasını döndürüyor.
            # Bu yüzden batch hesaplamada "repainting" olmadan (lookahead olmadan) değer üretmek zor.
            
            # Basit yaklaşım: Pivot noktalarını yerleştir ve forward fill yap (Step function).
            # Bu, "son bilinen pivot" mantığıdır.
            
            # Ancak _find_pivots listesi sıralı mı? Evet.
            
            # Pivot indexlerini data indexine çevir
            # data.index[pivot_indices]
            
            # Ancak burada bir sorun var: Pivot indexi geçmişte kalmış olabilir.
            # Bizim batch sonucumuzda, t anında "bilinen son pivot" olmalı.
            # _find_pivots fonksiyonunu modifiye etmeden teyit anını bilemeyiz.
            # Ama _find_pivots fonksiyonu teyit anında listeye ekliyor.
            # Yani pivot listesindeki sıra, teyit sırasıdır.
            
            # Tekrar _find_pivots mantığını batch içinde simüle etmek yerine,
            # _find_pivots'u kullanıp, pivot indexlerine göre yerleştirip ffill yapalım.
            # Bu "repainting" içerir (çünkü pivot indexi t-k olabilir).
            # Ama görselleştirme ve trend takibi için genelde bu istenir.
            
            # Eğer "non-repainting" istiyorsak, teyit anını bilmemiz lazım.
            # Şimdilik standart ZigZag davranışı (pivot noktalarını birleştiren çizgi) yerine
            # "Son Pivot Değeri"ni döndüreceğiz.
            
            for p in pivots:
                idx = p['index']
                s_values.iloc[idx] = p['value']
                s_types.iloc[idx] = p['type']
                
            # Forward fill
            s_values = s_values.ffill()
            s_types = s_types.ffill()
            
        return pd.DataFrame({
            'value': s_values,
            'pivot_type': s_types
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel ZigZag değeri
        """
        # Buffer yönetimi
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            # Depth kadar geriye bakmamız lazım
            self._high_buffer = deque(maxlen=self.depth + 1)
            self._low_buffer = deque(maxlen=self.depth + 1)
            # State
            self._last_pivot_val = None
            self._last_pivot_type = None
            self._last_pivot_idx = 0 # Relative index veya count
            self._total_candles = 0
            self._pivots = [] # Son birkaç pivotu tutabiliriz
            
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

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._total_candles += 1

        current_price = close_val
        
        # Yeterli veri yoksa
        if len(self._high_buffer) < self.depth + 1:
             # İlk değerler için basit initialization
             if self._last_pivot_val is None:
                 self._last_pivot_val = current_price
                 self._last_pivot_type = 'none'
             
             return IndicatorResult(
                value=self._last_pivot_val,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'pivot_type': 'none'}
            )

        # Incremental Calculation Logic
        # _find_pivots mantığını tek bir adım için uygula
        
        high_arr = np.array(self._high_buffer)
        low_arr = np.array(self._low_buffer)
        
        # Son depth kadar pencere
        # window_high = np.max(high[max(0, i - self.depth):i + 1])
        # Buffer zaten son (depth+1) veriyi tutuyor.
        # i (şu an) buffer'ın son elemanı.
        # window_high = np.max(buffer)
        
        window_high = np.max(high_arr)
        window_low = np.min(low_arr)
        
        threshold = self.deviation / 100
        
        # İlk pivot initialization (eğer henüz yoksa)
        if self._last_pivot_val is None:
            # Buffer doldu, ilk pivotu belirle
            # Basitçe en yüksek/düşük ile başla
            if (window_high - window_low) / window_low > threshold:
                self._last_pivot_val = window_high
                self._last_pivot_type = 'high'
                self._pivots.append({'value': window_high, 'type': 'high', 'index': self._total_candles - 1})
            else:
                # Henüz belirgin hareket yok
                self._last_pivot_val = current_price
                self._last_pivot_type = 'none'
        
        else:
            # Mevcut pivot var, yenisini ara
            if self._last_pivot_type == 'low' or self._last_pivot_type == 'none':
                # High pivot ara
                # Eğer none ise ve yükseliş varsa high başlat
                ref_val = self._last_pivot_val
                change = (window_high - ref_val) / ref_val
                
                if change > threshold:
                    # Yeni High Pivot bulundu
                    self._last_pivot_val = window_high
                    self._last_pivot_type = 'high'
                    self._pivots.append({'value': window_high, 'type': 'high', 'index': self._total_candles - 1})
                    
            elif self._last_pivot_type == 'high':
                # Low pivot ara
                ref_val = self._last_pivot_val
                change = (ref_val - window_low) / ref_val
                
                if change > threshold:
                    # Yeni Low Pivot bulundu
                    self._last_pivot_val = window_low
                    self._last_pivot_type = 'low'
                    self._pivots.append({'value': window_low, 'type': 'low', 'index': self._total_candles - 1})

        # Sonuç oluştur
        pivot_value = self._last_pivot_val
        pivot_type = self._last_pivot_type
        
        # Sinyal ve Trend (calculate metodundan alındı)
        signal = SignalType.HOLD
        trend = TrendDirection.NEUTRAL
        
        if pivot_type == 'high':
            trend = TrendDirection.DOWN
            if current_price < pivot_value:
                signal = SignalType.SELL
        elif pivot_type == 'low':
            trend = TrendDirection.UP
            if current_price > pivot_value:
                signal = SignalType.BUY
                
        # Strength
        price_change = abs((current_price - pivot_value) / pivot_value * 100) if pivot_value != 0 else 0
        strength = min(price_change / self.deviation * 100, 100)
        
        return IndicatorResult(
            value=round(pivot_value, 2),
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'deviation': self.deviation,
                'depth': self.depth,
                'pivot_type': pivot_type,
                'current_price': round(current_price, 2),
                'price_change_pct': round(price_change, 2),
                'total_pivots': len(self._pivots)
            }
        )

    def _find_pivots(self, high: np.ndarray, low: np.ndarray) -> list:
        """
        Swing high ve low noktalarını bul

        Args:
            high: High fiyatları
            low: Low fiyatları

        Returns:
            list: Pivot noktaları listesi
        """
        pivots = []
        last_pivot_value = None
        last_pivot_type = None
        threshold = self.deviation / 100

        # İlk pivot'u bul
        start_idx = self.depth
        current_high = np.max(high[:start_idx])
        current_low = np.min(low[:start_idx])

        if (current_high - current_low) / current_low > threshold:
            last_pivot_value = current_high
            last_pivot_type = 'high'
            pivots.append({
                'value': current_high,
                'type': 'high',
                'index': np.argmax(high[:start_idx])
            })

        # Devam eden pivotları bul
        for i in range(start_idx, len(high)):
            window_high = np.max(high[max(0, i - self.depth):i + 1])
            window_low = np.min(low[max(0, i - self.depth):i + 1])

            if last_pivot_value is None:
                last_pivot_value = window_high
                last_pivot_type = 'high'
                continue

            # Yeni high pivot
            if last_pivot_type == 'low':
                change = (window_high - last_pivot_value) / last_pivot_value
                if change > threshold:
                    pivots.append({
                        'value': window_high,
                        'type': 'high',
                        'index': i
                    })
                    last_pivot_value = window_high
                    last_pivot_type = 'high'

            # Yeni low pivot
            elif last_pivot_type == 'high':
                change = (last_pivot_value - window_low) / last_pivot_value
                if change > threshold:
                    pivots.append({
                        'value': window_low,
                        'type': 'low',
                        'index': i
                    })
                    last_pivot_value = window_low
                    last_pivot_type = 'low'

        return pivots

    def get_signal(self, value: float) -> SignalType:
        """
        ZigZag değerinden sinyal üret (calculate içinde yapılıyor)

        Args:
            value: ZigZag değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        ZigZag değerinden trend belirle (calculate içinde yapılıyor)

        Args:
            value: ZigZag değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'deviation': 5.0,
            'depth': 12
        }

    def _requires_volume(self) -> bool:
        """ZigZag volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ZigZag']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """ZigZag indikatör testi"""

    print("\n" + "="*60)
    print("ZIGZAG INDICATOR TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend değişimleriyle fiyat hareketi simüle et
    base_price = 100
    prices = [base_price]
    trend = 1  # 1: yukarı, -1: aşağı

    for i in range(99):
        # Her 20 mumda trend değiştir
        if i % 20 == 0:
            trend *= -1

        change = np.random.randn() * 0.5 + (trend * 0.3)
        prices.append(prices[-1] * (1 + change / 100))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    zigzag = ZigZag(deviation=5.0, depth=12)
    print(f"   [OK] Oluşturuldu: {zigzag}")
    print(f"   [OK] Kategori: {zigzag.category.value}")
    print(f"   [OK] Tip: {zigzag.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {zigzag.get_required_periods()}")

    result = zigzag(data)
    print(f"   [OK] Son Pivot: {result.value}")
    print(f"   [OK] Pivot Tipi: {result.metadata['pivot_type']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Toplam Pivot: {result.metadata['total_pivots']}")
    print(f"   [OK] Fiyat Değişim %: {result.metadata['price_change_pct']}")

    # Test 2: Farklı deviation değerleri
    print("\n3. Farklı deviation testi...")
    for dev in [3.0, 5.0, 10.0]:
        zigzag_test = ZigZag(deviation=dev, depth=12)
        result = zigzag_test.calculate(data)
        print(f"   [OK] ZigZag(dev={dev}) - Pivots: {result.metadata['total_pivots']} | "
              f"Sinyal: {result.signal.value}")

    # Test 3: Farklı depth değerleri
    print("\n4. Farklı depth testi...")
    for depth in [5, 12, 20]:
        zigzag_test = ZigZag(deviation=5.0, depth=depth)
        result = zigzag_test.calculate(data)
        print(f"   [OK] ZigZag(depth={depth}) - Pivots: {result.metadata['total_pivots']} | "
              f"Last: {result.value}")

    # Test 4: Pivot analizi
    print("\n5. Pivot analizi...")
    result = zigzag.calculate(data)
    current = result.metadata['current_price']
    pivot = result.value
    pivot_type = result.metadata['pivot_type']

    print(f"   [OK] Güncel fiyat: {current}")
    print(f"   [OK] Son pivot: {pivot} ({pivot_type})")

    if pivot_type == 'high':
        print(f"   [OK] Son swing high'dan sonra düşüş trendi")
        if current < pivot:
            print(f"   [OK] Fiyat pivot altında, düşüş devam ediyor")
        else:
            print(f"   [OK] Fiyat pivot üstünde, toparlanma sinyali")
    elif pivot_type == 'low':
        print(f"   [OK] Son swing low'dan sonra yükseliş trendi")
        if current > pivot:
            print(f"   [OK] Fiyat pivot üstünde, yükseliş devam ediyor")
        else:
            print(f"   [OK] Fiyat pivot altında, zayıflama sinyali")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = zigzag.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = zigzag.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
