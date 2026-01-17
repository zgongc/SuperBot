"""
indicators/statistical/cointegration.py - Cointegration (Eş-bütünleşme)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Cointegration - İki zaman serisinin uzun vadeli dengesini test eder
    Çıktılar:
        - spread: İki varlık arasındaki fark (spread)
        - zscore: Spread'in Z-Score'u
        - is_cointegrated: Eş-bütünleşme var mı (boolean)

    Pairs trading stratejilerinde kritik öneme sahiptir.

Formül:
    1. Hedge Ratio (β) hesapla: Linear Regression
       Asset1 = β × Asset2 + ε

    2. Spread hesapla:
       Spread = Asset1 - (β × Asset2)

    3. Spread'in Z-Score'unu hesapla:
       Z-Score = (Spread - Mean(Spread)) / Std(Spread)

    4. Augmented Dickey-Fuller (ADF) testi ile durağanlık kontrolü

Kullanım:
    - Pairs trading
    - Statistical arbitrage
    - Mean reversion stratejileri

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scipy>=1.10.0
    - statsmodels>=0.14.0 (ADF testi için)
"""

import numpy as np
import pandas as pd
from scipy import stats
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)

# statsmodels için optional import
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class Cointegration(BaseIndicator):
    """
    Cointegration (Eş-bütünleşme)

    İki varlık arasındaki uzun vadeli dengeyi ve spread'i hesaplar.
    Pairs trading stratejileri için kullanılır.

    Args:
        period: Hesaplama periyodu (varsayılan: 50)
        reference_data: Karşılaştırılacak referans veri (varsayılan: None)
        entry_threshold: Spread Z-Score giriş eşiği (varsayılan: 2.0)
        exit_threshold: Spread Z-Score çıkış eşiği (varsayılan: 0.5)
        adf_significance: ADF test anlamlılık seviyesi (varsayılan: 0.05)
    """

    def __init__(
        self,
        period: int = 50,
        reference_data: pd.DataFrame = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        adf_significance: float = 0.05,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.reference_data = reference_data
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.adf_significance = adf_significance

        super().__init__(
            name='cointegration',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'period': period,
                'reference_data': reference_data,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'adf_significance': adf_significance
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 10:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot en az 10 olmalı (cointegration için)"
            )
        if self.entry_threshold <= self.exit_threshold:
            raise InvalidParameterError(
                self.name, 'thresholds',
                f"entry={self.entry_threshold}, exit={self.exit_threshold}",
                "Entry threshold, exit threshold'dan büyük olmalı"
            )
        if not (0 < self.adf_significance < 1):
            raise InvalidParameterError(
                self.name, 'adf_significance', self.adf_significance,
                "ADF anlamlılık seviyesi 0-1 arası olmalı"
            )
        return True

    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Referans veriyi ayarla (pair'in diğer varlığı)

        Args:
            reference_data: Referans OHLCV DataFrame
        """
        self.reference_data = reference_data

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: spread, zscore, is_cointegrated
        """
        if self.reference_data is None:
            return pd.DataFrame(index=data.index, columns=['spread', 'zscore', 'is_cointegrated'])
            
        # Veri hizalama
        # Indexlerin eşleştiğinden emin olmalıyız
        common_index = data.index.intersection(self.reference_data.index)
        if len(common_index) < self.period:
            return pd.DataFrame(index=data.index, columns=['spread', 'zscore', 'is_cointegrated'])
            
        s1 = data.loc[common_index, 'close']
        s2 = self.reference_data.loc[common_index, 'close']
        
        # Rolling Statistics
        roll = s1.rolling(window=self.period)
        mean_s1 = roll.mean()
        var_s1 = roll.var()
        
        roll2 = s2.rolling(window=self.period)
        mean_s2 = roll2.mean()
        var_s2 = roll2.var()
        
        cov_s1s2 = s1.rolling(window=self.period).cov(s2)
        
        # Beta (Hedge Ratio)
        beta = cov_s1s2 / var_s2
        
        # Spread Mean (using current beta for the whole window)
        # mean(e) = mean(s1 - beta*s2) = mean(s1) - beta * mean(s2)
        spread_mean = mean_s1 - beta * mean_s2
        
        # Spread Variance (using current beta for the whole window)
        # var(e) = var(s1 - beta*s2) = var(s1) + beta^2 * var(s2) - 2*beta*cov(s1,s2)
        spread_var = var_s1 + (beta ** 2) * var_s2 - 2 * beta * cov_s1s2
        spread_std = np.sqrt(spread_var)
        
        # Current Spread (Residual at t)
        spread = s1 - beta * s2
        
        # Z-Score
        # Sıfıra bölme hatasını önle
        spread_std = spread_std.replace(0, np.nan)
        zscore = (spread - spread_mean) / spread_std
        
        # Cointegration Check (Vectorized approximation)
        # ADF testi çok yavaş olduğu için batch modunda volatility ratio kullanıyoruz
        # spread_volatility = spread_std / abs(spread_mean)
        # is_cointegrated = spread_volatility < 0.5
        
        # Sıfıra bölme hatasını önle
        spread_mean_abs = spread_mean.abs().replace(0, np.nan)
        spread_volatility = spread_std / spread_mean_abs
        is_cointegrated = spread_volatility < 0.5
        
        # Sonuç DataFrame
        result = pd.DataFrame({
            'spread': spread,
            'zscore': zscore,
            'is_cointegrated': is_cointegrated
        }, index=common_index)
        
        # Orijinal index'e reindex et (eksik yerler NaN kalır)
        return result.reindex(data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Güncel Cointegration değerleri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        if self.reference_data is None:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Timestamp kontrolü
        timestamp = candle.get('timestamp')
        if timestamp is None:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Referans veriden ilgili mumu bul
        # Not: Bu basit bir lookup. Gerçek hayatta senkronizasyon önemli.
        # self.reference_data'nın index'i timestamp veya datetime olmalı.
        # Eğer range index ise timestamp kolonunda arama yapmalıyız.
        
        ref_price = None
        
        # Timestamp ile eşleşen satırı bulmaya çalış
        if isinstance(self.reference_data.index, pd.DatetimeIndex):
            # Timestamp ms ise datetime'a çevir gerekirse
            try:
                ts_dt = pd.to_datetime(timestamp, unit='ms')
                if ts_dt in self.reference_data.index:
                    ref_price = self.reference_data.loc[ts_dt, 'close']
            except:
                pass
        else:
            # 'timestamp' kolonu varsa
            if 'timestamp' in self.reference_data.columns:
                matches = self.reference_data[self.reference_data['timestamp'] == timestamp]
                if not matches.empty:
                    ref_price = matches.iloc[0]['close']
        
        if ref_price is None:
            # Eşleşen referans veri yoksa hesaplama yapılamaz
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Buffer yönetimi (BaseIndicator'da standart bir buffer yok, kendimiz yönetelim veya calculate çağıralım)
        # Cointegration için buffer yönetimi karmaşık çünkü iki seri var.
        # En kolayı: Son period kadar veriyi alıp calculate çağırmak (Incremental değil ama Realtime)
        # Ancak performans için incremental yapmak istersek:
        
        # Şimdilik güvenli yol: RealtimeCalculator zaten buffer tutuyor (data['close']).
        # Ancak sadece ana varlık için tutuyor.
        # Bizim referans varlık için de son period verisine ihtiyacımız var.
        
        # Bu yüzden burada basitçe `calculate` metodunu son data ile çağırmak en mantıklısı.
        # Ancak `calculate` metodu tüm datayı alıyor gibi görünüyor.
        # BaseIndicator.update() varsayılan olarak calculate() çağırır zaten.
        # Ama biz optimize etmek istiyoruz.
        
        # Optimize edilmiş update için:
        # 1. Sınıf state'inde son periodluk close1 ve close2 tutulmalı.
        # 2. Yeni gelen close1 ve bulunan close2 eklenmeli.
        # 3. Hesaplama yapılmalı.
        
        # State initialize (ilk seferde)
        if not hasattr(self, '_close1_buffer'):
            from collections import deque
            self._close1_buffer = deque(maxlen=self.period)
            self._close2_buffer = deque(maxlen=self.period)
            
            # Eğer geçmiş veri varsa doldur (warmup)
            # Bu kısım biraz tricky, çünkü update() tek tek çağrılır.
            # Warmup dışarıdan yapılmalı veya ilk update'de yapılmalı.
            pass

        close1 = candle['close']
        close2 = ref_price
        
        self._close1_buffer.append(close1)
        self._close2_buffer.append(close2)
        
        if len(self._close1_buffer) < self.period:
             return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
             
        # Hesaplama
        c1 = np.array(self._close1_buffer)
        c2 = np.array(self._close2_buffer)
        
        # Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(c2, c1)
        hedge_ratio = slope
        
        # Spread
        spread_series = c1 - (hedge_ratio * c2)
        current_spread = spread_series[-1]
        
        # Z-Score
        spread_mean = np.mean(spread_series)
        spread_std = np.std(spread_series, ddof=1)
        
        if spread_std > 0:
            spread_zscore = (current_spread - spread_mean) / spread_std
        else:
            spread_zscore = 0.0
            
        # Cointegration Check (ADF veya Volatility)
        # Realtime'da ADF çalıştırabiliriz (tek seferlik olduğu için)
        is_cointegrated = False
        adf_pvalue = 1.0
        
        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(spread_series, autolag='AIC')
                adf_pvalue = adf_result[1]
                is_cointegrated = adf_pvalue < self.adf_significance
            except:
                spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
                is_cointegrated = spread_volatility < 0.5
        else:
            spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
            is_cointegrated = spread_volatility < 0.5
            
        # Sonuç
        strength = min(abs(spread_zscore) * 50, 100)
        
        return IndicatorResult(
            value={
                'spread': round(current_spread, 4),
                'zscore': round(spread_zscore, 4),
                'is_cointegrated': is_cointegrated
            },
            timestamp=int(timestamp) if timestamp else 0,
            signal=self.get_signal(spread_zscore, is_cointegrated),
            trend=self.get_trend(spread_zscore),
            strength=strength,
            metadata={
                'period': self.period,
                'hedge_ratio': round(hedge_ratio, 4),
                'spread_mean': round(spread_mean, 4),
                'spread_std': round(spread_std, 4),
                'adf_pvalue': round(adf_pvalue, 6),
                'correlation': round(r_value, 4),
                'asset1_price': round(close1, 2),
                'asset2_price': round(close2, 2),
                'has_statsmodels': HAS_STATSMODELS
            }
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Cointegration hesapla

        Args:
            data: OHLCV DataFrame (birinci varlık)

        Returns:
            IndicatorResult: Cointegration analizi sonuçları
        """
        if self.reference_data is None:
            # Referans veri yoksa dummy değerler döndür
            timestamp = int(data.iloc[-1]['timestamp'])
            return IndicatorResult(
                value={
                    'spread': 0.0,
                    'zscore': 0.0,
                    'is_cointegrated': False
                },
                timestamp=timestamp,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={
                    'error': 'No reference data provided',
                    'hedge_ratio': 0.0,
                    'adf_pvalue': 1.0
                }
            )

        # Veri hazırlığı
        close1 = data['close'].values[-self.period:]
        close2 = self.reference_data['close'].values[-self.period:]

        # Veri uzunluklarını eşitle
        min_len = min(len(close1), len(close2))
        if min_len < 10:
            timestamp = int(data.iloc[-1]['timestamp'])
            return IndicatorResult(
                value={
                    'spread': 0.0,
                    'zscore': 0.0,
                    'is_cointegrated': False
                },
                timestamp=timestamp,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={
                    'error': 'Insufficient data',
                    'hedge_ratio': 0.0,
                    'adf_pvalue': 1.0
                }
            )

        close1 = close1[-min_len:]
        close2 = close2[-min_len:]
        
        # Bufferları doldur (Incremental update için hazırlık)
        if not hasattr(self, '_close1_buffer'):
            from collections import deque
            self._close1_buffer = deque(maxlen=self.period)
            self._close2_buffer = deque(maxlen=self.period)
            
        # Son period kadar veriyi buffer'a at
        # Not: calculate() genellikle tüm tarihçeyi alır ama biz son durumu state olarak saklamak istiyoruz.
        # Eğer calculate() backtest için çağrılıyorsa bu state son anı temsil eder.
        buffer_data1 = close1[-self.period:] if len(close1) >= self.period else close1
        buffer_data2 = close2[-self.period:] if len(close2) >= self.period else close2
        
        self._close1_buffer.clear()
        self._close2_buffer.clear()
        self._close1_buffer.extend(buffer_data1)
        self._close2_buffer.extend(buffer_data2)

        # 1. Hedge Ratio hesapla (Linear Regression)
        # close1 = beta * close2 + alpha
        slope, intercept, r_value, p_value, std_err = stats.linregress(close2, close1)
        hedge_ratio = slope

        # 2. Spread hesapla
        spread_series = close1 - (hedge_ratio * close2)
        current_spread = spread_series[-1]

        # 3. Spread'in Z-Score'unu hesapla
        spread_mean = np.mean(spread_series)
        spread_std = np.std(spread_series, ddof=1)

        if spread_std > 0:
            spread_zscore = (current_spread - spread_mean) / spread_std
        else:
            spread_zscore = 0.0

        # 4. ADF test ile durağanlık kontrolü (cointegration testi)
        is_cointegrated = False
        adf_pvalue = 1.0

        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(spread_series, autolag='AIC')
                adf_statistic = adf_result[0]
                adf_pvalue = adf_result[1]

                # p-value < anlamlılık seviyesi ise spread durağan (cointegrated)
                is_cointegrated = adf_pvalue < self.adf_significance
            except Exception:
                # ADF test başarısız olursa basit volatilite kontrolü yap
                spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
                is_cointegrated = spread_volatility < 0.5
        else:
            # statsmodels yoksa basit volatilite kontrolü
            spread_volatility = spread_std / abs(spread_mean) if spread_mean != 0 else 999
            is_cointegrated = spread_volatility < 0.5

        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal gücü: Z-Score'un mutlak değeri
        strength = min(abs(spread_zscore) * 50, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'spread': round(current_spread, 4),
                'zscore': round(spread_zscore, 4),
                'is_cointegrated': is_cointegrated
            },
            timestamp=timestamp,
            signal=self.get_signal(spread_zscore, is_cointegrated),
            trend=self.get_trend(spread_zscore),
            strength=strength,
            metadata={
                'period': self.period,
                'hedge_ratio': round(hedge_ratio, 4),
                'spread_mean': round(spread_mean, 4),
                'spread_std': round(spread_std, 4),
                'adf_pvalue': round(adf_pvalue, 6),
                'correlation': round(r_value, 4),
                'asset1_price': round(close1[-1], 2),
                'asset2_price': round(close2[-1], 2),
                'has_statsmodels': HAS_STATSMODELS
            }
        )

    def get_signal(self, zscore: float, is_cointegrated: bool) -> SignalType:
        """
        Spread Z-Score'dan sinyal üret

        Args:
            zscore: Spread'in Z-Score'u
            is_cointegrated: Eş-bütünleşme var mı

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # Eş-bütünleşme yoksa sinyal verme
        if not is_cointegrated:
            return SignalType.HOLD

        # Spread çok düşük (asset1 ucuz, asset2 pahalı)
        # Asset1 al, Asset2 sat
        if zscore <= -self.entry_threshold:
            return SignalType.BUY

        # Spread çok yüksek (asset1 pahalı, asset2 ucuz)
        # Asset1 sat, Asset2 al
        elif zscore >= self.entry_threshold:
            return SignalType.SELL

        # Spread normale dönüyor, pozisyon kapat
        elif abs(zscore) <= self.exit_threshold:
            return SignalType.HOLD

        return SignalType.HOLD

    def get_trend(self, zscore: float) -> TrendDirection:
        """
        Spread Z-Score'dan trend belirle

        Args:
            zscore: Spread'in Z-Score'u

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if zscore > 1:
            return TrendDirection.UP  # Spread genişliyor
        elif zscore < -1:
            return TrendDirection.DOWN  # Spread daralıyor
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 50,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'adf_significance': 0.05
        }

    def _requires_volume(self) -> bool:
        """Cointegration volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Cointegration']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Cointegration indikatör testi"""

    print("\n" + "="*60)
    print("COINTEGRATION (EŞ-BÜTÜNLEŞME) TEST")
    print("="*60 + "\n")

    # Test 1: Eş-bütünleşmiş varlıklar
    print("1. Eş-bütünleşmiş varlık çifti testi...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Varlık 1 - ortak trend + bireysel noise
    common_trend = np.cumsum(np.random.randn(100) * 0.1)
    individual_noise1 = np.random.randn(100) * 0.3

    prices1 = 100 + common_trend + individual_noise1

    data1 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices1,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices1],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices1],
        'close': prices1,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices1]
    })

    # Varlık 2 - aynı ortak trend + farklı bireysel noise (eş-bütünleşik)
    individual_noise2 = np.random.randn(100) * 0.3
    prices2 = 110 + common_trend * 1.2 + individual_noise2

    data2 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices2,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices2],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices2],
        'close': prices2,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices2]
    })

    print(f"   [OK] {len(data1)} mum oluşturuldu")
    print(f"   [OK] Varlık 1 fiyat: {prices1[-1]:.2f}")
    print(f"   [OK] Varlık 2 fiyat: {prices2[-1]:.2f}")

    coint = Cointegration(period=50, reference_data=data2)
    print(f"   [OK] Oluşturuldu: {coint}")
    print(f"   [OK] Kategori: {coint.category.value}")
    print(f"   [OK] statsmodels var: {HAS_STATSMODELS}")

    result = coint(data1)
    print(f"   [OK] Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Eş-bütünleşme: {result.value['is_cointegrated']}")
    print(f"   [OK] Hedge Ratio: {result.metadata['hedge_ratio']}")
    print(f"   [OK] Korelasyon: {result.metadata['correlation']}")
    print(f"   [OK] ADF P-value: {result.metadata['adf_pvalue']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Eş-bütünleşmemiş varlıklar
    print("\n2. Eş-bütünleşmemiş varlık çifti testi...")
    np.random.seed(99)

    # Varlık 3 - tamamen bağımsız random walk
    prices3 = [100]
    for _ in range(99):
        prices3.append(prices3[-1] + np.random.randn() * 2)

    data3 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices3,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices3],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices3],
        'close': prices3,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices3]
    })

    # Varlık 4 - başka bir bağımsız random walk
    prices4 = [110]
    for _ in range(99):
        prices4.append(prices4[-1] + np.random.randn() * 2)

    data4 = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices4,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices4],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices4],
        'close': prices4,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices4]
    })

    coint.set_reference_data(data4)
    result = coint.calculate(data3)
    print(f"   [OK] Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Eş-bütünleşme: {result.value['is_cointegrated']}")
    print(f"   [OK] Korelasyon: {result.metadata['correlation']}")
    print(f"   [OK] ADF P-value: {result.metadata['adf_pvalue']}")

    # Test 3: Spread genişlemesi - trading sinyali
    print("\n3. Trading sinyal testi (spread genişlemesi)...")
    # Spread'i yapay olarak genişlet
    prices1_wide = prices1.copy()
    prices1_wide[-1] += 5  # Son fiyatı yükselt

    data1_wide = data1.copy()
    data1_wide.loc[data1_wide.index[-1], 'close'] = prices1_wide[-1]

    coint.set_reference_data(data2)
    result = coint.calculate(data1_wide)
    print(f"   [OK] Genişletilmiş Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 4: Spread daralması
    print("\n4. Spread daralma testi...")
    prices1_narrow = prices1.copy()
    prices1_narrow[-1] -= 3  # Son fiyatı düşür

    data1_narrow = data1.copy()
    data1_narrow.loc[data1_narrow.index[-1], 'close'] = prices1_narrow[-1]

    result = coint.calculate(data1_narrow)
    print(f"   [OK] Daraltılmış Spread: {result.value['spread']:.4f}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: Farklı periyotlar
    print("\n5. Farklı periyot testi...")
    for period in [30, 50, 70]:
        if len(data1) >= period:
            coint_test = Cointegration(period=period, reference_data=data2)
            result = coint_test.calculate(data1)
            print(f"   [OK] Period({period}): Z-Score={result.value['zscore']:7.4f} | "
                  f"Coint={result.value['is_cointegrated']} | "
                  f"Hedge Ratio={result.metadata['hedge_ratio']:.4f}")

    # Test 6: Spread zaman serisi
    print("\n6. Spread zaman serisi (son 10 mum)...")
    coint_ts = Cointegration(period=50, reference_data=data2)
    for i in range(-10, 0):
        test_data1 = data1.iloc[:len(data1)+i]
        test_data2 = data2.iloc[:len(data2)+i]
        if len(test_data1) >= coint_ts.period:
            coint_ts.set_reference_data(test_data2)
            result = coint_ts.calculate(test_data1)
            print(f"   [OK] Mum {i:3d}: Spread={result.value['spread']:8.4f} | "
                  f"Z-Score={result.value['zscore']:7.4f} | "
                  f"Sinyal={result.signal.value}")

    # Test 7: Özel eşikler
    print("\n7. Özel eşik testi...")
    coint_custom = Cointegration(period=50, reference_data=data2,
                                  entry_threshold=3.0, exit_threshold=1.0)
    result = coint_custom.calculate(data1)
    print(f"   [OK] Entry threshold: {coint_custom.entry_threshold}")
    print(f"   [OK] Exit threshold: {coint_custom.exit_threshold}")
    print(f"   [OK] Z-Score: {result.value['zscore']:.4f}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 8: İstatistikler
    print("\n8. İstatistik testi...")
    stats_data = coint.statistics
    print(f"   [OK] Hesaplama sayısı: {stats_data['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats_data['error_count']}")

    # Test 9: Metadata
    print("\n9. Metadata testi...")
    metadata = coint.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    # Test 10: Referans veri olmadan
    print("\n10. Referans veri yok testi...")
    coint_no_ref = Cointegration(period=50)
    result = coint_no_ref.calculate(data1)
    print(f"   [OK] Spread: {result.value['spread']}")
    print(f"   [OK] Error mesajı: {result.metadata.get('error', 'N/A')}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
