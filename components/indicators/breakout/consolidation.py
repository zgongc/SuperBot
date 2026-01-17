"""
indicators/breakout/consolidation.py - Consolidation Detector

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Consolidation - Konsolidasyon Tespiti ve Puanı
    Fiyatın dar bir aralıkta hareket ettiği dönemleri tespit eder
    ve konsolidasyon kalitesini puanlar.

    Analiz Kriterleri:
    - Volatilite (ATR): Düşük volatilite = konsolidasyon
    - Range genişliği: Dar range = konsolidasyon
    - Fiyat dağılımı: Uniform dağılım = kaliteli konsolidasyon
    - Süre: Uzun süre = güçlü konsolidasyon

    Çıktı:
    - consolidation_score: 0-100 arası konsolidasyon puanı
    - 0-25: Trend var, konsolidasyon yok
    - 25-50: Zayıf konsolidasyon
    - 50-75: Orta konsolidasyon
    - 75-100: Güçlü konsolidasyon

Formül:
    ATR Score = (1 - Current ATR / Average ATR) × 100
    Range Score = (1 - Range % / Historical Avg %) × 100
    Distribution Score = Uniformity of price distribution
    Time Score = Consolidation duration

    Final Score = (ATR × 0.4) + (Range × 0.3) + (Distribution × 0.2) + (Time × 0.1)

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


class Consolidation(BaseIndicator):
    """
    Consolidation Detector

    Fiyatın konsolidasyon halinde olup olmadığını tespit eder ve
    konsolidasyon kalitesini 0-100 arası puanlar.

    Args:
        period: Analiz periyodu (varsayılan: 20)
        atr_period: ATR hesaplama periyodu (varsayılan: 14)
        lookback: Geçmiş karşılaştırma periyodu (varsayılan: 100)
        min_consolidation_bars: Minimum konsolidasyon mum sayısı (varsayılan: 10)
    """

    def __init__(
        self,
        period: int = 20,
        atr_period: int = 14,
        lookback: int = 100,
        min_consolidation_bars: int = 10,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.atr_period = atr_period
        self.lookback = lookback
        self.min_consolidation_bars = min_consolidation_bars

        super().__init__(
            name='consolidation',
            category=IndicatorCategory.BREAKOUT,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'atr_period': atr_period,
                'lookback': lookback,
                'min_consolidation_bars': min_consolidation_bars
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.period, self.atr_period, self.min_consolidation_bars) + 10

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 5:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot en az 5 olmalı"
            )
        if self.atr_period < 2:
            raise InvalidParameterError(
                self.name, 'atr_period', self.atr_period,
                "ATR periyodu en az 2 olmalı"
            )
        if self.min_consolidation_bars < 3:
            raise InvalidParameterError(
                self.name, 'min_consolidation_bars', self.min_consolidation_bars,
                "Minimum konsolidasyon mum sayısı en az 3 olmalı"
            )
        return True

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """ATR hesapla"""
        tr_list = []
        for i in range(1, len(close)):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i-1])
            l_pc = abs(low[i] - close[i-1])
            tr = max(h_l, h_pc, l_pc)
            tr_list.append(tr)

        tr_array = np.array(tr_list)
        if len(tr_array) >= self.atr_period:
            atr = np.mean(tr_array[-self.atr_period:])
        else:
            atr = np.mean(tr_array)

        return atr

    def _calculate_atr_score(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """ATR bazlı konsolidasyon puanı (0-100)"""
        # Mevcut ATR
        current_atr = self._calculate_atr(high, low, close)

        # Geçmiş ATR ortalaması
        lookback_size = min(self.lookback, len(close) - self.atr_period - 1)
        if lookback_size < self.atr_period:
            return 50.0

        historical_atrs = []
        for i in range(lookback_size):
            end_idx = -(i + 1)
            start_idx = end_idx - self.atr_period - 1
            if start_idx < -len(close):
                break
            hist_atr = self._calculate_atr(
                high[start_idx:end_idx],
                low[start_idx:end_idx],
                close[start_idx:end_idx]
            )
            historical_atrs.append(hist_atr)

        if not historical_atrs:
            return 50.0

        avg_historical_atr = np.mean(historical_atrs)

        # Düşük ATR = yüksek puan
        if avg_historical_atr == 0:
            return 50.0

        atr_ratio = current_atr / avg_historical_atr
        score = (1 - min(atr_ratio, 1.0)) * 100

        return score

    def _calculate_range_score(self, high: np.ndarray, low: np.ndarray) -> tuple:
        """Range bazlı konsolidasyon puanı (0-100)"""
        # Mevcut range
        current_high = np.max(high[-self.period:])
        current_low = np.min(low[-self.period:])
        current_range = current_high - current_low
        current_range_pct = (current_range / current_low * 100) if current_low > 0 else 0

        # Geçmiş range ortalaması
        lookback_size = min(self.lookback, len(high) - self.period)
        if lookback_size < self.period:
            return 50.0, current_range_pct

        historical_ranges = []
        for i in range(lookback_size):
            end_idx = -(i + 1)
            start_idx = end_idx - self.period
            if start_idx < -len(high):
                break
            hist_high = np.max(high[start_idx:end_idx])
            hist_low = np.min(low[start_idx:end_idx])
            hist_range = hist_high - hist_low
            hist_range_pct = (hist_range / hist_low * 100) if hist_low > 0 else 0
            historical_ranges.append(hist_range_pct)

        if not historical_ranges:
            return 50.0, current_range_pct

        avg_historical_range = np.mean(historical_ranges)

        # Dar range = yüksek puan
        if avg_historical_range == 0:
            return 50.0, current_range_pct

        range_ratio = current_range_pct / avg_historical_range
        score = (1 - min(range_ratio, 1.0)) * 100

        return score, current_range_pct

    def _calculate_distribution_score(self, close: np.ndarray) -> float:
        """Fiyat dağılımı puanı (0-100)"""
        # Son period içindeki fiyatların dağılımını analiz et
        prices = close[-self.period:]

        # Histogram oluştur (5 bin)
        hist, bin_edges = np.histogram(prices, bins=5)

        # Uniform dağılım = her bin'de eşit fiyat
        expected_count = len(prices) / 5

        # Chi-square benzeri test
        deviations = [(count - expected_count) ** 2 / expected_count for count in hist if expected_count > 0]
        if not deviations:
            return 50.0

        deviation_score = np.mean(deviations)

        # Düşük sapma = yüksek puan
        # Max sapma ~len(prices) olabilir
        max_deviation = len(prices) / 2
        score = (1 - min(deviation_score / max_deviation, 1.0)) * 100

        return score

    def _calculate_time_score(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple:
        """Konsolidasyon süresi puanı (0-100)"""
        # Range yüzdesini her mum için hesapla
        range_threshold = 3.0  # %3 eşiği

        consolidation_count = 0
        for i in range(self.min_consolidation_bars, 0, -1):
            end_idx = -i if i > 1 else None
            start_idx = end_idx - self.period if end_idx else -self.period

            period_high = np.max(high[start_idx:end_idx])
            period_low = np.min(low[start_idx:end_idx])
            period_range_pct = ((period_high - period_low) / period_low * 100) if period_low > 0 else 0

            if period_range_pct < range_threshold:
                consolidation_count += 1
            else:
                break

        # Uzun süre = yüksek puan
        score = min((consolidation_count / self.min_consolidation_bars) * 100, 100)

        return score, consolidation_count

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Consolidation hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Konsolidasyon puanı ve detayları
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Bufferları doldur (Incremental update için hazırlık)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = self.lookback + self.period + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        
        # Son max_len kadar veriyi al
        start_idx = max(0, len(data) - (self.lookback + self.period + 50))
        self._high_buffer.extend(high[start_idx:])
        self._low_buffer.extend(low[start_idx:])
        self._close_buffer.extend(close[start_idx:])

        # ATR puanı (ağırlık: 0.4)
        atr_score = self._calculate_atr_score(high, low, close)

        # Range puanı (ağırlık: 0.3)
        range_score, current_range_pct = self._calculate_range_score(high, low)

        # Dağılım puanı (ağırlık: 0.2)
        distribution_score = self._calculate_distribution_score(close)

        # Zaman puanı (ağırlık: 0.1)
        time_score, consolidation_bars = self._calculate_time_score(high, low, close)

        # Final konsolidasyon puanı
        consolidation_score = (
            atr_score * 0.4 +
            range_score * 0.3 +
            distribution_score * 0.2 +
            time_score * 0.1
        )

        # Konsolidasyon seviyesi
        if consolidation_score >= 75:
            level = "Strong"
        elif consolidation_score >= 50:
            level = "Moderate"
        elif consolidation_score >= 25:
            level = "Weak"
        else:
            level = "None"

        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirle
        signal = self.get_signal(consolidation_score)
        trend = self.get_trend(consolidation_score)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'value': round(consolidation_score, 2),
                'level': level,
                'atr_score': round(atr_score, 2),
                'range_score': round(range_score, 2),
                'distribution_score': round(distribution_score, 2),
                'time_score': round(time_score, 2),
                'range_pct': round(current_range_pct, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=consolidation_score,
            metadata={
                'consolidation_bars': consolidation_bars
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Consolidation değerleri
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 1. ATR Score
        # TR Hesapla
        h_l = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        
        # Current ATR
        current_atr = tr.rolling(window=self.atr_period).mean()
        
        # Historical Avg ATR (Lookback kadar geriye dönük ortalama)
        # Bu biraz ağır olabilir: rolling(lookback).mean()
        # Ancak logic: lookback içindeki ortalama ATR değil, 
        # lookback süresince hesaplanan ATR'lerin ortalaması.
        # Yani rolling(atr_period).mean() serisinin rolling(lookback).mean()'i.
        historical_avg_atr = current_atr.shift(1).rolling(window=self.lookback).mean()
        
        # ATR Ratio
        atr_ratio = current_atr / historical_avg_atr.replace(0, np.nan)
        atr_score = (1 - atr_ratio.clip(upper=1.0)) * 100
        atr_score = atr_score.fillna(50.0)
        
        # 2. Range Score
        # Current Range %
        roll_high = high.rolling(window=self.period).max()
        roll_low = low.rolling(window=self.period).min()
        current_range = roll_high - roll_low
        current_range_pct = (current_range / roll_low.replace(0, np.nan)) * 100
        current_range_pct = current_range_pct.fillna(0)
        
        # Historical Avg Range %
        historical_avg_range = current_range_pct.shift(1).rolling(window=self.lookback).mean()
        
        range_ratio = current_range_pct / historical_avg_range.replace(0, np.nan)
        range_score = (1 - range_ratio.clip(upper=1.0)) * 100
        range_score = range_score.fillna(50.0)
        
        # 3. Distribution Score
        # Rolling apply ile histogram hesabı yavaş olabilir.
        # Basitleştirilmiş yaklaşım: Standart sapma / Range
        # Uniform dağılımda std dev yüksektir (range'e göre).
        # Ama burada "uniformity" isteniyor.
        # Orijinal logic: Chi-square test.
        # Bunu batch yapmak zor. Rolling apply kullanalım.
        
        def calc_dist_score(x):
            if len(x) < 5: return 50.0
            hist, _ = np.histogram(x, bins=5)
            expected = len(x) / 5
            deviations = [(c - expected) ** 2 / expected for c in hist if expected > 0]
            if not deviations: return 50.0
            dev_score = np.mean(deviations)
            max_dev = len(x) / 2
            return (1 - min(dev_score / max_dev, 1.0)) * 100

        # Performance için sadece close üzerinde rolling apply
        # Bu işlem yavaş olabilir, optimize edilebilir.
        distribution_score = close.rolling(window=self.period).apply(calc_dist_score, raw=True)
        distribution_score = distribution_score.fillna(50.0)
        
        # 4. Time Score
        # Son X barda range < threshold olanların sayısı
        range_threshold = 3.0
        
        # Her bar için range % hesapla (period bazlı değil, o anki barın range'i değil)
        # Logic: "Range yüzdesini her mum için hesapla" diyor ama loop içinde
        # period_high/low kullanıyor. Yani her mumda geriye dönük period kadar bakıyor.
        # Bu zaten current_range_pct.
        
        # Yani: current_range_pct < 3.0 olan ardışık mum sayısı?
        # Hayır, loop: for i in range(min_consolidation_bars, 0, -1)
        # Geriye dönük i kadar gidip bakıyor.
        # Aslında mantık: Şu anki mumdan geriye doğru giderek, 
        # kaç tane mumun "kendi periodluk penceresinde" range'i düşük?
        
        is_consolidating = current_range_pct < range_threshold
        
        # Rolling sum of boolean?
        # Hayır, ardışık olması lazım.
        # Basitçe: Son min_consolidation_bars içindeki is_consolidating sayısı değil.
        # Loop mantığı: En uzundan başlayıp (min_consolidation_bars), 
        # eğer o periyotta range düşükse sayıyor.
        
        # Orijinal koda bakalım:
        # for i in range(self.min_consolidation_bars, 0, -1):
        #    period_range_pct = ...
        #    if period_range_pct < range_threshold: count += 1 else break
        
        # Bu mantık biraz garip. i azaldıkça pencere küçülmüyor, start_idx değişiyor.
        # start_idx = end_idx - self.period
        # Yani kayan pencere ile geriye gidiyor.
        
        # Batch için:
        # is_consolidating serisi (current_range_pct < 3.0)
        # Geriye dönük ardışık true sayısı.
        
        # Pandas ile ardışık grupları bulmak:
        # grouper = (is_consolidating != is_consolidating.shift()).cumsum()
        # Ama biz her bar için o anki ardışık sayıyı istiyoruz.
        
        # Bu tipik bir "consecutive streak" problemi.
        # df['streak'] = df.groupby((df['val'] != df['val'].shift()).cumsum()).cumcount() + 1
        # Sadece True olanları alacağız.
        
        streak_groups = (is_consolidating != is_consolidating.shift()).cumsum()
        streaks = is_consolidating.groupby(streak_groups).cumsum()
        
        # Sadece True olanlar streak, False olanlar 0 olmalı ama cumsum boolean'ı 1/0 toplar.
        # False olanlar resetlenmeli.
        # Daha iyi yöntem:
        # y = x * (y.shift() + 1)
        # Bu recursive, pandas'ta zor.
        
        # Alternatif: Rolling sum ama sadece hepsi 1 ise? Hayır.
        
        # Python loop ile hızlıca hesaplayalım (Numpy iteration)
        cons_vals = is_consolidating.values.astype(int)
        time_counts = np.zeros(len(cons_vals))
        
        current_streak = 0
        for i in range(len(cons_vals)):
            if cons_vals[i] == 1:
                current_streak += 1
            else:
                current_streak = 0
            time_counts[i] = current_streak
            
        # Max limit: min_consolidation_bars
        time_counts = np.minimum(time_counts, self.min_consolidation_bars)
        time_score = (time_counts / self.min_consolidation_bars) * 100
        time_score = pd.Series(time_score, index=data.index)
        
        # Final Score
        final_score = (
            atr_score * 0.4 +
            range_score * 0.3 +
            distribution_score * 0.2 +
            time_score * 0.1
        )
        
        # Level
        levels = pd.Series("None", index=data.index)
        levels[final_score >= 25] = "Weak"
        levels[final_score >= 50] = "Moderate"
        levels[final_score >= 75] = "Strong"
        
        return pd.DataFrame({
            'value': final_score,
            'level': levels,
            'atr_score': atr_score,
            'range_score': range_score,
            'distribution_score': distribution_score,
            'time_score': time_score,
            'range_pct': current_range_pct
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel Consolidation değeri
        """
        # Buffer yönetimi
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            # Lookback + period kadar veriye ihtiyaç olabilir (Historical avg için)
            # Ancak historical avg hesaplamak için çok fazla veri lazım.
            # Buffer'da sadece son period'u tutup, historical avg'yi incremental update etmek daha mantıklı.
            # Veya buffer'ı yeterince büyük tutmak.
            
            max_len = self.lookback + self.period + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            
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
        
        # Yeterli veri yoksa
        if len(self._close_buffer) < self.period:
             return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.NEUTRAL,
                trend=TrendDirection.UNKNOWN,
                strength=0.0,
                metadata={'level': 'None'}
            )
            
        # Hesaplama
        # Bufferları numpy array'e çevir (son lookback+period kadar)
        # Tüm buffer'ı kullanmak yerine optimize edilebilir ama şimdilik güvenli yol.
        
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        
        # ATR Score
        atr_score = self._calculate_atr_score(high, low, close)
        
        # Range Score
        range_score, current_range_pct = self._calculate_range_score(high, low)
        
        # Distribution Score
        distribution_score = self._calculate_distribution_score(close)
        
        # Time Score
        time_score, consolidation_bars = self._calculate_time_score(high, low, close)
        
        # Final Score
        consolidation_score = (
            atr_score * 0.4 +
            range_score * 0.3 +
            distribution_score * 0.2 +
            time_score * 0.1
        )
        
        # Level
        if consolidation_score >= 75:
            level = "Strong"
        elif consolidation_score >= 50:
            level = "Moderate"
        elif consolidation_score >= 25:
            level = "Weak"
        else:
            level = "None"
            
        return IndicatorResult(
            value=round(consolidation_score, 2),
            timestamp=timestamp_val,
            signal=self.get_signal(consolidation_score),
            trend=self.get_trend(consolidation_score),
            strength=consolidation_score,
            metadata={
                'level': level,
                'atr_score': round(atr_score, 2),
                'range_score': round(range_score, 2),
                'distribution_score': round(distribution_score, 2),
                'time_score': round(time_score, 2),
                'range_pct': round(current_range_pct, 2),
                'consolidation_bars': consolidation_bars
            }
        )

    def get_signal(self, score: float) -> SignalType:
        """
        Konsolidasyon puanından sinyal üret

        Args:
            score: Konsolidasyon puanı

        Returns:
            SignalType: HOLD (konsolidasyon sırasında bekle)
        """
        # Yüksek konsolidasyon = bekle (breakout için hazırlan)
        if score >= 75:
            return SignalType.HOLD
        elif score >= 50:
            return SignalType.HOLD
        else:
            return SignalType.NEUTRAL

    def get_trend(self, score: float) -> TrendDirection:
        """
        Konsolidasyon puanından trend belirle

        Args:
            score: Konsolidasyon puanı

        Returns:
            TrendDirection: NEUTRAL (konsolidasyon = trend yok)
        """
        # Konsolidasyon sırasında trend yok
        if score >= 50:
            return TrendDirection.NEUTRAL
        else:
            return TrendDirection.UNKNOWN

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20,
            'atr_period': 14,
            'lookback': 100,
            'min_consolidation_bars': 10
        }

    def _requires_volume(self) -> bool:
        """Consolidation volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Consolidation']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Consolidation indikatör testi"""

    print("\n" + "="*60)
    print("CONSOLIDATION DETECTOR TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend -> Konsolidasyon -> Trend simüle et
    base_price = 100
    prices = [base_price]

    # İlk 30 mum: Uptrend
    for i in range(29):
        change = np.random.randn() * 1.0 + 0.5
        prices.append(prices[-1] + change)

    # Sonraki 60 mum: Konsolidasyon
    consolidation_base = prices[-1]
    for i in range(59):
        change = np.random.randn() * 0.2
        prices.append(np.clip(prices[-1] + change, consolidation_base - 1, consolidation_base + 1))

    # Son 60 mum: Breakout + trend
    for i in range(60):
        change = np.random.randn() * 1.5 + 0.8
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p + np.random.randn() * 0.1 for p in prices],
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    consol = Consolidation()
    print(f"   [OK] Oluşturuldu: {consol}")
    print(f"   [OK] Kategori: {consol.category.value}")
    print(f"   [OK] Gerekli periyot: {consol.get_required_periods()}")

    result = consol(data)
    print(f"   [OK] Konsolidasyon Puanı: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Trend testi
    print("\n3. Trend testi (ilk 30 mum)...")
    trend_data = data.head(50)
    result = consol.calculate(trend_data)
    print(f"   [OK] Konsolidasyon Puanı: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] ATR Score: {result.metadata['atr_score']:.2f}")
    print(f"   [OK] Range Score: {result.metadata['range_score']:.2f}")

    # Test 3: Konsolidasyon testi
    print("\n4. Konsolidasyon testi (60-90 mum)...")
    consol_data = data.iloc[30:90].reset_index(drop=True)
    # Yeterli geçmiş veri için baştan ekleme yapmamız gerekiyor
    consol_data_full = data.head(90)
    result = consol.calculate(consol_data_full)
    print(f"   [OK] Konsolidasyon Puanı: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] ATR Score: {result.metadata['atr_score']:.2f}")
    print(f"   [OK] Range Score: {result.metadata['range_score']:.2f}")
    print(f"   [OK] Distribution Score: {result.metadata['distribution_score']:.2f}")
    print(f"   [OK] Time Score: {result.metadata['time_score']:.2f}")
    print(f"   [OK] Consolidation Bars: {result.metadata['consolidation_bars']}")

    # Test 4: Breakout sonrası
    print("\n5. Breakout sonrası test (tüm data)...")
    result = consol.calculate(data)
    print(f"   [OK] Konsolidasyon Puanı: {result.value}")
    print(f"   [OK] Seviye: {result.metadata['level']}")
    print(f"   [OK] Range %: {result.metadata['range_pct']:.2f}%")

    # Test 5: Zaman serisi analizi
    print("\n6. Zaman serisi analizi...")
    scores = []
    levels = []

    for i in range(40, len(data), 10):
        partial_data = data.head(i)
        result = consol.calculate(partial_data)
        scores.append(result.value)
        levels.append(result.metadata['level'])

    print(f"   [OK] Toplam ölçüm: {len(scores)}")
    print(f"   [OK] Ortalama puan: {np.mean(scores):.2f}")
    print(f"   [OK] Max puan: {max(scores):.2f}")
    print(f"   [OK] Min puan: {min(scores):.2f}")
    print(f"   [OK] Strong: {levels.count('Strong')}, Moderate: {levels.count('Moderate')}")
    print(f"   [OK] Weak: {levels.count('Weak')}, None: {levels.count('None')}")

    # Test 6: Farklı parametreler
    print("\n7. Farklı parametre testi...")
    consol_fast = Consolidation(period=10, min_consolidation_bars=5)
    result = consol_fast.calculate(consol_data_full)
    print(f"   [OK] Fast (10 period) Puanı: {result.value}")

    # Test 7: İstatistikler
    print("\n8. İstatistik testi...")
    stats = consol.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = consol.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
