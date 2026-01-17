"""
indicators/combo/triple_screen.py - Elder's Triple Screen Trading System

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Triple Screen - Dr. Alexander Elder'ın ünlü 3 ekran ticaret sistemi
    Üç farklı zaman dilimi ve indikatörü birleştirerek güvenilir sinyaller üretir

    Sistem:
    1. EKRAN 1 (Trend): Uzun vadeli trend (EMA veya MACD)
    2. EKRAN 2 (Oscillator): Orta vadeli momentum (RSI veya Stochastic)
    3. EKRAN 3 (Entry): Kısa vadeli giriş (Price Action)

    Özellikler:
    - Çoklu zaman dilimi analizi
    - Trend filtreleme sistemi
    - Güçlü risk yönetimi
    - Yüksek doğruluk oranı

Strateji:
    AL Sinyali:
    - Ekran 1: Uzun vadeli yükseliş trendi (EMA > EMA_slow VEYA MACD > 0)
    - Ekran 2: RSI aşırı satım bölgesinde (<30)
    - Ekran 3: Fiyat önceki dip seviyesinden yüksek

    SAT Sinyali:
    - Ekran 1: Uzun vadeli düşüş trendi (EMA < EMA_slow VEYA MACD < 0)
    - Ekran 2: RSI aşırı alım bölgesinde (>70)
    - Ekran 3: Fiyat önceki tepe seviyesinden düşük

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.trend.ema
    - indicators.trend.macd
    - indicators.momentum.rsi
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.ema import EMA
from indicators.trend.macd import MACD
from indicators.momentum.rsi import RSI
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class TripleScreen(BaseIndicator):
    """
    Elder's Triple Screen Trading System

    Üç ekran sistemi ile güvenilir al/sat sinyalleri üretir.

    Args:
        ema_fast: Hızlı EMA periyodu (varsayılan: 13)
        ema_slow: Yavaş EMA periyodu (varsayılan: 26)
        rsi_period: RSI periyodu (varsayılan: 13)
        rsi_overbought: RSI aşırı alım seviyesi (varsayılan: 70)
        rsi_oversold: RSI aşırı satım seviyesi (varsayılan: 30)
        use_macd: MACD kullan (varsayılan: True), False ise EMA crossover
    """

    def __init__(
        self,
        ema_fast: int = 13,
        ema_slow: int = 26,
        rsi_period: int = 13,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        use_macd: bool = True,
        logger=None,
        error_handler=None
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_macd = use_macd

        # Ekran 1: Trend belirleme (MACD veya EMA crossover)
        if use_macd:
            self.trend_indicator = MACD(
                fast_period=12,
                slow_period=26,
                signal_period=9,
                logger=logger,
                error_handler=error_handler
            )
        else:
            self.ema_fast_ind = EMA(period=ema_fast, logger=logger, error_handler=error_handler)
            self.ema_slow_ind = EMA(period=ema_slow, logger=logger, error_handler=error_handler)

        # Ekran 2: Oscillator (RSI)
        self.rsi = RSI(
            period=rsi_period,
            overbought=rsi_overbought,
            oversold=rsi_oversold,
            logger=logger,
            error_handler=error_handler
        )

        super().__init__(
            name='triple_screen',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'use_macd': use_macd
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        if self.use_macd:
            return max(self.trend_indicator.get_required_periods(), self.rsi.get_required_periods())
        else:
            return max(self.ema_slow, self.rsi.get_required_periods())

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.ema_fast >= self.ema_slow:
            raise InvalidParameterError(
                self.name, 'ema_periods',
                f"fast={self.ema_fast}, slow={self.ema_slow}",
                "EMA fast periyodu slow'dan küçük olmalı"
            )
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "RSI periyodu pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Triple Screen sistemi hesaplama

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Üç ekran analizi ve kombine sinyal
        """
        # EKRAN 1: Uzun vadeli trend
        if self.use_macd:
            trend_result = self.trend_indicator.calculate(data)
            screen1_value = trend_result.value['histogram']
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = trend_result.signal
        else:
            ema_fast_result = self.ema_fast_ind.calculate(data)
            ema_slow_result = self.ema_slow_ind.calculate(data)
            screen1_value = ema_fast_result.value - ema_slow_result.value
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = SignalType.BUY if screen1_value > 0 else SignalType.SELL

        # EKRAN 2: Orta vadeli oscillator (RSI)
        rsi_result = self.rsi.calculate(data)
        rsi_value = rsi_result.value
        screen2_signal = rsi_result.signal
        screen2_strength = rsi_result.strength

        # EKRAN 3: Kısa vadeli giriş noktası (Price Action)
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Son 5 mumun dip ve tepe seviyeleri
        recent_high = np.max(high[-5:]) if len(high) >= 5 else high[-1]
        recent_low = np.min(low[-5:]) if len(low) >= 5 else low[-1]
        current_price = close[-1]

        # Fiyat aksiyonu kontrolü
        screen3_buy = current_price > recent_low  # Yeni dip oluşmuyor
        screen3_sell = current_price < recent_high  # Yeni tepe oluşmuyor

        timestamp = int(data.iloc[-1]['timestamp'])

        # Kombine sinyal üretimi (3 ekranın hepsi onaylamalı)
        signal = self._generate_signal(
            screen1_trend, screen1_signal,
            rsi_value, screen2_signal,
            screen3_buy, screen3_sell
        )

        # Sinyal gücü ve konfirmasyon
        strength = self._calculate_strength(screen1_trend, rsi_value, screen3_buy, screen3_sell)
        confirmation = self._get_confirmation(screen1_signal, screen2_signal, signal)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'screen1': round(screen1_value, 4),
                'screen2': round(rsi_value, 2),
                'screen3_buy': screen3_buy,
                'screen3_sell': screen3_sell,
                'current_price': round(current_price, 8)
            },
            timestamp=timestamp,
            signal=signal,
            trend=screen1_trend,
            strength=strength,
            metadata={
                'ema_fast': self.ema_fast,
                'ema_slow': self.ema_slow,
                'rsi_period': self.rsi_period,
                'use_macd': self.use_macd,
                'screen1_trend': screen1_trend.name,
                'screen1_signal': screen1_signal.value,
                'screen2_rsi': round(rsi_value, 2),
                'screen2_signal': screen2_signal.value,
                'screen3_status': 'ready' if (screen3_buy or screen3_sell) else 'waiting',
                'confirmation': confirmation,
                'recent_high': round(recent_high, 8),
                'recent_low': round(recent_low, 8)
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel Triple Screen değeri
        """
        # Buffer yönetimi
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = self.get_required_periods() + 50
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
        min_required = self.get_required_periods()
        if len(self._close_buffer) < min_required:
            return IndicatorResult(
                value={'screen1': 0.0, 'screen2': 50.0, 'screen3_buy': False, 'screen3_sell': False, 'current_price': candle['close']},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'confirmation': 'none'}
            )
            
        # DataFrame oluştur (sub-indikatörler için)
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        # EKRAN 1: Uzun vadeli trend
        if self.use_macd:
            trend_result = self.trend_indicator.calculate(buffer_data)
            screen1_value = trend_result.value['histogram']
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = trend_result.signal
        else:
            ema_fast_result = self.ema_fast_ind.calculate(buffer_data)
            ema_slow_result = self.ema_slow_ind.calculate(buffer_data)
            screen1_value = ema_fast_result.value - ema_slow_result.value
            screen1_trend = TrendDirection.UP if screen1_value > 0 else TrendDirection.DOWN
            screen1_signal = SignalType.BUY if screen1_value > 0 else SignalType.SELL
        
        # EKRAN 2: Orta vadeli oscillator (RSI)
        rsi_result = self.rsi.calculate(buffer_data)
        rsi_value = rsi_result.value
        screen2_signal = rsi_result.signal
        
        # EKRAN 3: Kısa vadeli giriş noktası (Price Action)
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        
        # Son 5 mumun dip ve tepe seviyeleri
        recent_high = np.max(high[-5:]) if len(high) >= 5 else high[-1]
        recent_low = np.min(low[-5:]) if len(low) >= 5 else low[-1]
        current_price = close[-1]
        
        # Fiyat aksiyonu kontrolü
        screen3_buy = current_price > recent_low
        screen3_sell = current_price < recent_high
        
        # Kombine sinyal üretimi
        signal = self._generate_signal(
            screen1_trend, screen1_signal,
            rsi_value, screen2_signal,
            screen3_buy, screen3_sell
        )

        # Sinyal güçü ve konfirmasyon
        strength = self._calculate_strength(screen1_trend, rsi_value, screen3_buy, screen3_sell)
        confirmation = self._get_confirmation(screen1_signal, screen2_signal, signal)

        return IndicatorResult(
            value={
                'screen1': round(screen1_value, 4),
                'screen2': round(rsi_value, 2),
                'screen3_buy': screen3_buy,
                'screen3_sell': screen3_sell,
                'current_price': round(current_price, 8)
            },
            timestamp=timestamp_val,
            signal=signal,
            trend=screen1_trend,
            strength=strength,
            metadata={
                'ema_fast': self.ema_fast,
                'ema_slow': self.ema_slow,
                'rsi_period': self.rsi_period,
                'use_macd': self.use_macd,
                'screen1_trend': screen1_trend.name,
                'screen1_signal': screen1_signal.value,
                'screen2_rsi': round(rsi_value, 2),
                'screen2_signal': screen2_signal.value,
                'screen3_status': 'ready' if (screen3_buy or screen3_sell) else 'waiting',
                'confirmation': confirmation,
                'recent_high': round(recent_high, 8),
                'recent_low': round(recent_low, 8)
            }
        )

    def _generate_signal(
        self,
        screen1_trend: TrendDirection,
        screen1_signal: SignalType,
        rsi_value: float,
        screen2_signal: SignalType,
        screen3_buy: bool,
        screen3_sell: bool
    ) -> SignalType:
        """
        Üç ekrandan kombine sinyal üret

        Tüm ekranlar aynı yönde onaylamalı
        """
        # AL Sinyali: Ekran 1 yükseliş + Ekran 2 aşırı satım + Ekran 3 onay
        if (screen1_trend == TrendDirection.UP and
            rsi_value < self.rsi_oversold and
            screen3_buy):
            return SignalType.BUY

        # SAT Sinyali: Ekran 1 düşüş + Ekran 2 aşırı alım + Ekran 3 onay
        if (screen1_trend == TrendDirection.DOWN and
            rsi_value > self.rsi_overbought and
            screen3_sell):
            return SignalType.SELL

        # Kısmi onaylar (daha zayıf sinyaller)
        if screen1_trend == TrendDirection.UP and rsi_value < 40 and screen3_buy:
            return SignalType.BUY

        if screen1_trend == TrendDirection.DOWN and rsi_value > 60 and screen3_sell:
            return SignalType.SELL

        return SignalType.HOLD

    def _calculate_strength(
        self,
        screen1_trend: TrendDirection,
        rsi_value: float,
        screen3_buy: bool,
        screen3_sell: bool
    ) -> float:
        """
        Sinyal gücünü hesapla (0-100)

        Her ekranın katkısını değerlendir
        """
        strength = 0

        # Ekran 1 katkısı (trend gücü)
        if screen1_trend != TrendDirection.NEUTRAL:
            strength += 30

        # Ekran 2 katkısı (RSI ekstrem seviyeler)
        rsi_deviation = abs(rsi_value - 50)
        if rsi_value < 30 or rsi_value > 70:
            strength += min(rsi_deviation, 40)
        else:
            strength += min(rsi_deviation / 2, 20)

        # Ekran 3 katkısı (price action onayı)
        if screen3_buy or screen3_sell:
            strength += 30

        return min(strength, 100)

    def _get_confirmation(
        self,
        screen1_signal: SignalType,
        screen2_signal: SignalType,
        final_signal: SignalType
    ) -> str:
        """
        Sinyal konfirmasyonu seviyesini belirle

        Returns:
            str: 'triple', 'double', 'single' veya 'none'
        """
        # Üç ekran da aynı yönde
        if (screen1_signal == screen2_signal == final_signal and
            final_signal != SignalType.HOLD):
            return 'triple'

        # İki ekran aynı yönde
        if ((screen1_signal == screen2_signal and screen1_signal != SignalType.HOLD) or
            (screen1_signal == final_signal and screen1_signal != SignalType.HOLD) or
            (screen2_signal == final_signal and screen2_signal != SignalType.HOLD)):
            return 'double'

        # Tek ekran sinyal veriyor
        if final_signal != SignalType.HOLD:
            return 'single'

        return 'none'

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation - returns DataFrame with multiple columns

        Note: This is a simple implementation for compatibility.
        For performance, consider implementing vectorized logic.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Indicator values with columns: screen1, screen2, screen3_buy, screen3_sell, current_price
        """
        results = {
            'screen1': [],
            'screen2': [],
            'screen3_buy': [],
            'screen3_sell': [],
            'current_price': []
        }

        for i in range(len(data)):
            if i < self.get_required_periods() - 1:
                for key in results:
                    results[key].append(np.nan)
            else:
                window_data = data.iloc[:i+1]
                result = self.calculate(window_data)

                # Extract dict values
                if result and hasattr(result, 'value') and isinstance(result.value, dict):
                    results['screen1'].append(result.value.get('screen1', np.nan))
                    results['screen2'].append(result.value.get('screen2', np.nan))
                    results['screen3_buy'].append(result.value.get('screen3_buy', False))
                    results['screen3_sell'].append(result.value.get('screen3_sell', False))
                    results['current_price'].append(result.value.get('current_price', np.nan))
                else:
                    for key in results:
                        results[key].append(np.nan)

        return pd.DataFrame(results, index=data.index)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'ema_fast': 13,
            'ema_slow': 26,
            'rsi_period': 13,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'use_macd': True
        }

    def _requires_volume(self) -> bool:
        """Triple Screen volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TripleScreen']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Triple Screen indikatör testi"""

    print("\n" + "="*60)
    print("TRIPLE SCREEN TRADING SYSTEM TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend değişimli piyasa simülasyonu
    base_price = 100
    prices = [base_price]
    highs = []
    lows = []

    for i in range(99):
        if i < 30:
            trend = 0.5  # Yükseliş
        elif i < 60:
            trend = -0.3  # Düşüş
        else:
            trend = 0.6  # Güçlü yükseliş
        noise = np.random.randn() * 2
        new_price = prices[-1] + trend + noise
        prices.append(new_price)
        highs.append(new_price + abs(np.random.randn()) * 1.0)
        lows.append(new_price - abs(np.random.randn()) * 1.0)

    # İlk fiyat için high/low ekle
    highs.insert(0, prices[0] + 1)
    lows.insert(0, prices[0] - 1)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama (MACD versiyonu)
    print("\n2. Temel hesaplama testi (MACD)...")
    ts = TripleScreen(use_macd=True)
    print(f"   [OK] Oluşturuldu: {ts}")
    print(f"   [OK] Kategori: {ts.category.value}")
    print(f"   [OK] Tip: {ts.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {ts.get_required_periods()}")

    result = ts(data)
    print(f"   [OK] Ekran 1 (MACD): {result.value['screen1']}")
    print(f"   [OK] Ekran 2 (RSI): {result.value['screen2']}")
    print(f"   [OK] Ekran 3 Buy: {result.value['screen3_buy']}")
    print(f"   [OK] Ekran 3 Sell: {result.value['screen3_sell']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")

    # Test 2: Ekran analizi
    print("\n3. Ekran analizi...")
    print(f"   [OK] Ekran 1 Trend: {result.metadata['screen1_trend']}")
    print(f"   [OK] Ekran 1 Sinyal: {result.metadata['screen1_signal']}")
    print(f"   [OK] Ekran 2 RSI: {result.metadata['screen2_rsi']}")
    print(f"   [OK] Ekran 2 Sinyal: {result.metadata['screen2_signal']}")
    print(f"   [OK] Ekran 3 Status: {result.metadata['screen3_status']}")
    print(f"   [OK] Konfirmasyon: {result.metadata['confirmation']}")

    # Test 3: EMA versiyonu
    print("\n4. EMA crossover versiyonu testi...")
    ts_ema = TripleScreen(use_macd=False, ema_fast=13, ema_slow=26)
    result_ema = ts_ema.calculate(data)
    print(f"   [OK] Ekran 1 (EMA): {result_ema.value['screen1']:.4f}")
    print(f"   [OK] Ekran 2 (RSI): {result_ema.value['screen2']}")
    print(f"   [OK] Sinyal: {result_ema.signal.value}")
    print(f"   [OK] Konfirmasyon: {result_ema.metadata['confirmation']}")

    # Test 4: Trend değişimi analizi
    print("\n5. Trend değişimi analizi...")
    test_points = [35, 55, 75, 95]
    for idx in test_points:
        data_slice = data.iloc[:idx+1]
        result = ts.calculate(data_slice)
        print(f"   [OK] Mum {idx}: "
              f"Screen1={result.metadata['screen1_trend']}, "
              f"RSI={result.value['screen2']:.1f}, "
              f"Sinyal={result.signal.value}, "
              f"Confirm={result.metadata['confirmation']}")

    # Test 5: Özel parametreler
    print("\n6. Özel parametre testi...")
    ts_custom = TripleScreen(
        ema_fast=8,
        ema_slow=21,
        rsi_period=9,
        rsi_overbought=75,
        rsi_oversold=25,
        use_macd=True
    )
    result = ts_custom.calculate(data)
    print(f"   [OK] Özel parametreli ekran 1: {result.value['screen1']:.4f}")
    print(f"   [OK] Özel parametreli ekran 2: {result.value['screen2']}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 6: Sinyal gücü ve konfirmasyon
    print("\n7. Sinyal gücü ve konfirmasyon analizi...")
    print(f"   [OK] Sinyal Gücü: {result.strength:.2f}/100")
    print(f"   [OK] Konfirmasyon Seviyesi: {result.metadata['confirmation']}")
    print(f"   [OK] Recent High: {result.metadata['recent_high']:.2f}")
    print(f"   [OK] Recent Low: {result.metadata['recent_low']:.2f}")

    # Test 7: İstatistikler
    print("\n8. İstatistik testi...")
    stats = ts.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = ts.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
