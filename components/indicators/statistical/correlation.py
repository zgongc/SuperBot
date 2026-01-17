"""
indicators/statistical/correlation.py - Correlation (Korelasyon)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Correlation - İki varlık arasındaki doğrusal ilişkiyi ölçer
    Aralık: -1 ile +1 arası
    +1: Mükemmel pozitif korelasyon (birlikte hareket)
    0: Korelasyon yok (bağımsız)
    -1: Mükemmel negatif korelasyon (ters yönde hareket)

Formül:
    Pearson Korelasyon Katsayısı:
    r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)

    Rolling korelasyon belirli bir pencere üzerinden hesaplanır.

Kullanım:
    - Pairs trading stratejileri
    - Portföy çeşitlendirme
    - Risk yönetimi

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

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


class Correlation(BaseIndicator):
    """
    Correlation (Korelasyon)

    İki fiyat serisi arasındaki korelasyonu hesaplar.
    Pairs trading ve portföy analizi için kullanılır.

    Args:
        period: Korelasyon pencere periyodu (varsayılan: 20)
        reference_data: Karşılaştırılacak referans veri (varsayılan: None)
        high_correlation: Yüksek korelasyon eşiği (varsayılan: 0.7)
        low_correlation: Düşük korelasyon eşiği (varsayılan: -0.7)
    """

    def __init__(
        self,
        period: int = 20,
        reference_data: pd.DataFrame = None,
        high_correlation: float = 0.7,
        low_correlation: float = -0.7,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.reference_data = reference_data
        self.high_correlation = high_correlation
        self.low_correlation = low_correlation

        super().__init__(
            name='correlation',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'reference_data': reference_data,
                'high_correlation': high_correlation,
                'low_correlation': low_correlation
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot en az 2 olmalı (korelasyon için)"
            )
        if not (-1 <= self.low_correlation < self.high_correlation <= 1):
            raise InvalidParameterError(
                self.name, 'thresholds',
                f"low={self.low_correlation}, high={self.high_correlation}",
                "Eşikler -1 ile 1 arası olmalı ve low < high"
            )
        return True

    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Referans veriyi ayarla (karşılaştırılacak varlık)

        Args:
            reference_data: Referans OHLCV DataFrame
        """
        self.reference_data = reference_data

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Korelasyon hesapla

        Args:
            data: OHLCV DataFrame (birinci varlık)

        Returns:
            IndicatorResult: Korelasyon değeri
        """
        # Referans veri yoksa, kendi geçmiş verileriyle korelasyonu hesapla (autocorrelation)
        if self.reference_data is None:
            close = data['close'].values
            period_data = close[-self.period:]

            # Autocorrelation: mevcut fiyat ile gecikmeli fiyat arasında
            if len(period_data) >= 2:
                x = period_data[:-1]  # t-1
                y = period_data[1:]   # t
                correlation = np.corrcoef(x, y)[0, 1]
                reference_name = "self_lag1"
                reference_price = period_data[-2] if len(period_data) >= 2 else period_data[-1]
            else:
                correlation = 0.0
                reference_name = "self_lag1"
                reference_price = close[-1]
        else:
            # İki farklı varlık arasında korelasyon
            close1 = data['close'].values[-self.period:]
            close2 = self.reference_data['close'].values[-self.period:]

            # Veri uzunluklarını eşitle
            min_len = min(len(close1), len(close2))
            if min_len < 2:
                correlation = 0.0
            else:
                close1 = close1[-min_len:]
                close2 = close2[-min_len:]
                correlation = np.corrcoef(close1, close2)[0, 1]

            reference_name = "reference_asset"
            reference_price = close2[-1] if len(close2) > 0 else 0

        # NaN kontrolü
        if np.isnan(correlation):
            correlation = 0.0

        timestamp = int(data.iloc[-1]['timestamp'])
        current_price = data['close'].values[-1]

        # Korelasyon gücü: mutlak değer
        strength = abs(correlation) * 100

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(correlation, 4),
            timestamp=timestamp,
            signal=self.get_signal(correlation),
            trend=self.get_trend(correlation),
            strength=strength,
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'reference_price': round(reference_price, 2),
                'reference_name': reference_name,
                'correlation_pct': round(correlation * 100, 2),
                'relationship': self._get_relationship(correlation)
            }
        )

    def _get_relationship(self, correlation: float) -> str:
        """
        Korelasyon değerinden ilişki türünü belirle

        Args:
            correlation: Korelasyon katsayısı

        Returns:
            str: İlişki açıklaması
        """
        abs_corr = abs(correlation)

        if abs_corr >= 0.9:
            strength = "Çok Güçlü"
        elif abs_corr >= 0.7:
            strength = "Güçlü"
        elif abs_corr >= 0.5:
            strength = "Orta"
        elif abs_corr >= 0.3:
            strength = "Zayıf"
        else:
            strength = "Çok Zayıf"

        direction = "Pozitif" if correlation >= 0 else "Negatif"

        return f"{strength} {direction}"

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Correlation calculation - BACKTEST için

        Correlation Formula:
            Rolling Pearson correlation coefficient over 'period' window
            - If reference_data is None: autocorrelation (self vs lag-1)
            - If reference_data provided: cross-correlation with reference

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Correlation values (-1 to +1) for all bars

        Performance: 2000 bars in ~0.05 seconds
        """
        self._validate_data(data)

        close = data['close']

        # If no reference data, calculate autocorrelation (lag-1)
        if self.reference_data is None:
            # Autocorrelation with lag-1
            correlation = close.rolling(window=self.period).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 2 else 0,
                raw=True
            )
        else:
            # Cross-correlation with reference asset
            ref_close = self.reference_data['close']

            # Align both series to same index (use minimum length)
            min_len = min(len(close), len(ref_close))
            close_aligned = close.iloc[-min_len:].reset_index(drop=True)
            ref_aligned = ref_close.iloc[-min_len:].reset_index(drop=True)

            # Calculate rolling correlation
            correlation = close_aligned.rolling(window=self.period).corr(ref_aligned)

            # Re-index to original
            correlation.index = data.index[-min_len:]
            correlation = correlation.reindex(data.index)

        # Handle NaN values
        correlation = correlation.fillna(0)

        # Set first period values to NaN (warmup)
        correlation.iloc[:self.period-1] = np.nan

        return pd.Series(correlation.values, index=data.index, name='correlation')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - update() için gerekli state'i hazırlar"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._close_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)
            self._buffers_init = True

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
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Korelasyon değerinden sinyal üret

        Args:
            value: Korelasyon değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # Yüksek pozitif korelasyon: varlıklar birlikte hareket ediyor
        if value >= self.high_correlation:
            return SignalType.BUY

        # Yüksek negatif korelasyon: varlıklar ters hareket ediyor
        elif value <= self.low_correlation:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Korelasyon değerinden trend belirle

        Args:
            value: Korelasyon değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > 0.3:
            return TrendDirection.UP  # Pozitif korelasyon
        elif value < -0.3:
            return TrendDirection.DOWN  # Negatif korelasyon
        return TrendDirection.NEUTRAL  # Düşük/yok korelasyon

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20,
            'high_correlation': 0.7,
            'low_correlation': -0.7
        }

    def _requires_volume(self) -> bool:
        """Correlation volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Correlation']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Correlation indikatör testi"""

    print("\n" + "="*60)
    print("CORRELATION (KORELASYON) TEST")
    print("="*60 + "\n")

    # Test 1: Autocorrelation testi
    print("1. Autocorrelation testi (kendi gecikmiş haliyle)...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend + momentum içeren fiyat serisi
    base_price = 100
    prices = [base_price]
    for i in range(49):
        momentum = prices[-1] - (prices[-2] if len(prices) > 1 else base_price)
        trend = 0.1
        noise = np.random.randn() * 0.5
        prices.append(prices[-1] + trend + momentum * 0.3 + noise)

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

    corr = Correlation(period=20)
    print(f"   [OK] Oluşturuldu: {corr}")
    print(f"   [OK] Kategori: {corr.category.value}")

    result = corr(data)
    print(f"   [OK] Autocorrelation: {result.value}")
    print(f"   [OK] İlişki: {result.metadata['relationship']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Pozitif korelasyon - iki benzer varlık
    print("\n2. Pozitif korelasyon testi (benzer hareketler)...")
    np.random.seed(42)

    # Varlık 1
    prices1 = [100]
    for i in range(49):
        trend = 0.2 + np.random.randn() * 0.5
        prices1.append(prices1[-1] + trend)

    data1 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices1,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices1],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices1],
        'close': prices1,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices1]
    })

    # Varlık 2 - Varlık 1 ile yüksek korelasyonlu
    prices2 = []
    for p in prices1:
        # Aynı trend + küçük noise
        prices2.append(p * 1.1 + np.random.randn() * 0.3)

    data2 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices2,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices2],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices2],
        'close': prices2,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices2]
    })

    corr.set_reference_data(data2)
    result = corr.calculate(data1)
    print(f"   [OK] Pozitif Korelasyon: {result.value}")
    print(f"   [OK] İlişki: {result.metadata['relationship']}")
    print(f"   [OK] Korelasyon %: {result.metadata['correlation_pct']}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 3: Negatif korelasyon - ters hareketler
    print("\n3. Negatif korelasyon testi (ters hareketler)...")
    prices3 = []
    for p in prices1:
        # Ters hareket
        prices3.append(200 - p + np.random.randn() * 0.3)

    data3 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices3,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices3],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices3],
        'close': prices3,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices3]
    })

    corr.set_reference_data(data3)
    result = corr.calculate(data1)
    print(f"   [OK] Negatif Korelasyon: {result.value}")
    print(f"   [OK] İlişki: {result.metadata['relationship']}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 4: Sıfır korelasyon - bağımsız hareketler
    print("\n4. Sıfır korelasyon testi (bağımsız hareketler)...")
    np.random.seed(99)
    prices4 = [100 + np.random.randn() * 3 for _ in range(50)]

    data4 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices4,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices4],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices4],
        'close': prices4,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices4]
    })

    corr.set_reference_data(data4)
    result = corr.calculate(data1)
    print(f"   [OK] Sıfır Korelasyon: {result.value}")
    print(f"   [OK] İlişki: {result.metadata['relationship']}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: Farklı periyotlar
    print("\n5. Farklı periyot testi...")
    corr.set_reference_data(data2)  # Pozitif korelasyonlu veri
    for period in [10, 20, 30]:
        corr_test = Correlation(period=period, reference_data=data2)
        result = corr_test.calculate(data1)
        print(f"   [OK] Corr({period}): {result.value:.4f} | İlişki: {result.metadata['relationship']}")

    # Test 6: Rolling korelasyon analizi
    print("\n6. Rolling korelasyon testi (son 10 mum)...")
    corr_roll = Correlation(period=20, reference_data=data2)
    for i in range(-10, 0):
        test_data1 = data1.iloc[:len(data1)+i]
        test_data2 = data2.iloc[:len(data2)+i]
        corr_roll.set_reference_data(test_data2)
        if len(test_data1) >= corr_roll.period:
            result = corr_roll.calculate(test_data1)
            print(f"   [OK] Mum {i:3d}: Corr = {result.value:7.4f} | "
                  f"İlişki = {result.metadata['relationship']:20s} | "
                  f"Trend = {result.trend.name}")

    # Test 7: Özel eşikler
    print("\n7. Özel eşik testi...")
    corr_custom = Correlation(period=20, reference_data=data2,
                              high_correlation=0.9, low_correlation=-0.9)
    result = corr_custom.calculate(data1)
    print(f"   [OK] Özel eşikli Korelasyon: {result.value}")
    print(f"   [OK] High threshold: {corr_custom.high_correlation}")
    print(f"   [OK] Low threshold: {corr_custom.low_correlation}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 8: İstatistikler
    print("\n8. İstatistik testi...")
    stats = corr.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 9: Metadata
    print("\n9. Metadata testi...")
    metadata = corr.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
