"""
indicators/breakout/breakout_scanner.py - Multi-Timeframe Breakout Scanner

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Breakout Scanner - Multi-Timeframe Breakout Analizi
    Birden fazla zaman diliminde breakout tespiti yapar
    Destek/direnç seviyelerini, hacmi ve momentum'u analiz eder

    Analiz Kriterleri:
    - Fiyat range dışına çıkış (High/Low breakout)
    - Hacim artışı (ortalama hacmin üzerinde)
    - Momentum doğrulaması (RSI benzeri)
    - Multi-candle confirmation

    Çıktı:
    - breakout_score: 0-100 arası breakout puanı
    - direction: Breakout yönü (up/down/none)
    - resistance: Direnç seviyesi
    - support: Destek seviyesi

Formül:
    Range High = MAX(High, lookback)
    Range Low = MIN(Low, lookback)
    Breakout UP = Close > Range High (previous)
    Breakout DOWN = Close < Range Low (previous)

    Score = (Price Movement × 0.4) + (Volume × 0.3) + (Momentum × 0.3)

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


class BreakoutScanner(BaseIndicator):
    """
    Multi-Timeframe Breakout Scanner

    Destek/direnç seviyelerini, hacim ve momentum analizini kullanarak
    breakout'ları tespit eder ve puanlar.

    Args:
        lookback: Geriye bakış periyodu (varsayılan: 20)
        confirmation: Doğrulama mum sayısı (varsayılan: 2)
        volume_threshold: Hacim eşiği (ortalama hacmin katı) (varsayılan: 1.5)
        momentum_period: Momentum hesaplama periyodu (varsayılan: 14)
    """

    def __init__(
        self,
        lookback: int = 20,
        confirmation: int = 2,
        volume_threshold: float = 1.5,
        momentum_period: int = 14,
        logger=None,
        error_handler=None
    ):
        self.lookback = lookback
        self.confirmation = confirmation
        self.volume_threshold = volume_threshold
        self.momentum_period = momentum_period

        super().__init__(
            name='breakout_scanner',
            category=IndicatorCategory.BREAKOUT,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'lookback': lookback,
                'confirmation': confirmation,
                'volume_threshold': volume_threshold,
                'momentum_period': momentum_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.lookback, self.momentum_period) + self.confirmation + 5

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.lookback < 5:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback en az 5 olmalı"
            )
        if self.confirmation < 1:
            raise InvalidParameterError(
                self.name, 'confirmation', self.confirmation,
                "Confirmation en az 1 olmalı"
            )
        if self.volume_threshold <= 0:
            raise InvalidParameterError(
                self.name, 'volume_threshold', self.volume_threshold,
                "Volume threshold pozitif olmalı"
            )
        return True

    def _calculate_momentum(self, close: np.ndarray) -> float:
        """Momentum hesapla (RSI benzeri)"""
        if len(close) < self.momentum_period + 1:
            return 50.0

        changes = np.diff(close)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[-self.momentum_period:])
        avg_loss = np.mean(losses[-self.momentum_period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        momentum = 100 - (100 / (1 + rs))

        return momentum

    def _detect_breakout(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> tuple:
        """
        Breakout tespit et

        Returns:
            (breakout_type, score, resistance, support)
            breakout_type: 'up', 'down', 'none'
        """
        # Range belirleme (lookback periyodundan önceki)
        range_start = -(self.lookback + self.confirmation)
        range_end = -self.confirmation

        range_high = np.max(high[range_start:range_end])
        range_low = np.min(low[range_start:range_end])

        # Son confirmation mumlarını kontrol et
        recent_high = np.max(high[-self.confirmation:])
        recent_low = np.min(low[-self.confirmation:])
        current_close = close[-1]

        # Breakout kontrolü
        breakout_up = recent_high > range_high and current_close > range_high
        breakout_down = recent_low < range_low and current_close < range_low

        # Hacim analizi
        avg_volume = np.mean(volume[-(self.lookback + self.confirmation):-self.confirmation])
        recent_volume = np.mean(volume[-self.confirmation:])
        volume_confirm = recent_volume > (avg_volume * self.volume_threshold)

        # Momentum analizi
        momentum = self._calculate_momentum(close)

        # Score hesaplama
        score = 0.0
        breakout_type = 'none'

        if breakout_up:
            breakout_type = 'up'

            # Fiyat hareketi puanı (0-40)
            price_move = ((current_close - range_high) / range_high) * 100
            price_score = min(price_move * 10, 40)

            # Hacim puanı (0-30)
            volume_score = 30 if volume_confirm else 15

            # Momentum puanı (0-30)
            momentum_score = (momentum / 100) * 30

            score = price_score + volume_score + momentum_score

        elif breakout_down:
            breakout_type = 'down'

            # Fiyat hareketi puanı (0-40)
            price_move = ((range_low - current_close) / range_low) * 100
            price_score = min(price_move * 10, 40)

            # Hacim puanı (0-30)
            volume_score = 30 if volume_confirm else 15

            # Momentum puanı (0-30)
            momentum_score = ((100 - momentum) / 100) * 30

            score = price_score + volume_score + momentum_score

        return breakout_type, min(score, 100), range_high, range_low, momentum, volume_confirm

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Breakout Scanner hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Breakout analizi
        """
        # Bufferları doldur (Incremental update için hazırlık)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = max(self.lookback, self.momentum_period) + self.confirmation + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        self._volume_buffer.clear()
        
        # Son max_len kadar veriyi al
        start_idx = max(0, len(data) - (max(self.lookback, self.momentum_period) + self.confirmation + 50))
        self._high_buffer.extend(data['high'].values[start_idx:])
        self._low_buffer.extend(data['low'].values[start_idx:])
        self._close_buffer.extend(data['close'].values[start_idx:])
        self._volume_buffer.extend(data['volume'].values[start_idx:])
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Breakout tespit et
        breakout_type, score, resistance, support, momentum, volume_confirm = self._detect_breakout(
            high, low, close, volume
        )

        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirle
        signal = self.get_signal(breakout_type, score)
        trend = self.get_trend(breakout_type)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'breakout_score': round(score, 2),
                'direction': breakout_type,
                'resistance': round(resistance, 2),
                'support': round(support, 2),
                'momentum': round(momentum, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=score,
            metadata={
                'breakout_type': breakout_type,
                'volume_confirm': volume_confirm,
                'range_width': round(resistance - support, 2),
                'price': round(close[-1], 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Breakout değerleri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        n = len(data)
        
        # Sonuç dizileri (calculate() ile uyumlu - sadece 5 key)
        results = {
            'breakout_score': np.zeros(n),
            'direction': np.full(n, 'none', dtype=object),
            'resistance': np.zeros(n),
            'support': np.zeros(n),
            'momentum': np.full(n, 50.0)
        }
        
        # Her bar için hesapla
        min_required = max(self.lookback, self.momentum_period) + self.confirmation
        for i in range(min_required, n):
            # Gerekli veriyi al
            start_idx = max(0, i - min_required - 10)
            h = high[start_idx:i+1]
            l = low[start_idx:i+1]
            c = close[start_idx:i+1]
            v = volume[start_idx:i+1]
            
            # Breakout tespit et
            breakout_type, score, resistance, support, momentum, volume_confirm = self._detect_breakout(h, l, c, v)
            
            results['breakout_score'][i] = score
            results['direction'][i] = breakout_type
            results['resistance'][i] = resistance
            results['support'][i] = support
            results['momentum'][i] = momentum
            # volume_confirm metadata'da, output'ta değil
            
        return pd.DataFrame(results, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel Breakout değeri
        """
        # Buffer yönetimi
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = max(self.lookback, self.momentum_period) + self.confirmation + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            
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
        self._volume_buffer.append(volume_val)
        
        # Yeterli veri yoksa
        min_required = max(self.lookback, self.momentum_period) + self.confirmation
        if len(self._close_buffer) < min_required:
            return IndicatorResult(
                value={'breakout_score': 0.0, 'direction': 'none', 'resistance': 0.0, 'support': 0.0, 'momentum': 50.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'breakout_type': 'none', 'volume_confirm': False}
            )
            
        # Hesaplama
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        volume = np.array(self._volume_buffer)
        
        # Breakout tespit et
        breakout_type, score, resistance, support, momentum, volume_confirm = self._detect_breakout(
            high, low, close, volume
        )
        
        # Sinyal belirle
        signal = self.get_signal(breakout_type, score)
        trend = self.get_trend(breakout_type)

        return IndicatorResult(
            value={
                'breakout_score': round(score, 2),
                'direction': breakout_type,
                'resistance': round(resistance, 2),
                'support': round(support, 2),
                'momentum': round(momentum, 2)
            },
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=score,
            metadata={
                'breakout_type': breakout_type,
                'volume_confirm': volume_confirm,
                'range_width': round(resistance - support, 2),
                'price': round(close[-1], 2)
            }
        )

    def get_signal(self, breakout_type: str, score: float) -> SignalType:
        """
        Breakout tipinden sinyal üret

        Args:
            breakout_type: Breakout yönü ('up', 'down', 'none')
            score: Breakout puanı

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if breakout_type == 'up':
            if score >= 70:
                return SignalType.STRONG_BUY
            elif score >= 50:
                return SignalType.BUY
        elif breakout_type == 'down':
            if score >= 70:
                return SignalType.STRONG_SELL
            elif score >= 50:
                return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, breakout_type: str) -> TrendDirection:
        """
        Breakout tipinden trend belirle

        Args:
            breakout_type: Breakout yönü

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if breakout_type == 'up':
            return TrendDirection.UP
        elif breakout_type == 'down':
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'lookback': 20,
            'confirmation': 2,
            'volume_threshold': 1.5,
            'momentum_period': 14
        }

    def _requires_volume(self) -> bool:
        """Breakout Scanner volume gerektirir"""
        return True

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['breakout_score', 'direction', 'resistance', 'support', 'momentum']


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['BreakoutScanner']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Breakout Scanner indikatör testi"""

    print("\n" + "="*60)
    print("BREAKOUT SCANNER TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Range -> Breakout simüle et
    base_price = 100
    prices = [base_price]

    # İlk 40 mum: Dar range
    for i in range(39):
        change = np.random.randn() * 0.3
        prices.append(np.clip(prices[-1] + change, 98, 102))

    # Sonraki 30 mum: Breakout up
    for i in range(30):
        change = np.random.randn() * 0.5 + 0.8
        prices.append(prices[-1] + change)

    # Son 30 mum: Devam veya konsolidasyon
    for i in range(30):
        change = np.random.randn() * 0.5
        prices.append(prices[-1] + change)

    volumes = [1000 + np.random.randint(0, 500) for _ in range(40)]
    volumes.extend([2000 + np.random.randint(0, 1000) for _ in range(30)])  # Yüksek hacim
    volumes.extend([1000 + np.random.randint(0, 500) for _ in range(30)])

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    scanner = BreakoutScanner()
    print(f"   [OK] Oluşturuldu: {scanner}")
    print(f"   [OK] Kategori: {scanner.category.value}")
    print(f"   [OK] Gerekli periyot: {scanner.get_required_periods()}")

    result = scanner(data)
    print(f"   [OK] Breakout Score: {result.value['breakout_score']}")
    print(f"   [OK] Direction: {result.value['direction']}")
    print(f"   [OK] Resistance: {result.value['resistance']}")
    print(f"   [OK] Support: {result.value['support']}")
    print(f"   [OK] Momentum: {result.value['momentum']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Range testi
    print("\n3. Range testi (ilk 40 mum)...")
    range_data = data.head(50)
    result = scanner.calculate(range_data)
    print(f"   [OK] Breakout Score: {result.value['breakout_score']}")
    print(f"   [OK] Direction: {result.value['direction']}")
    print(f"   [OK] Range Width: {result.metadata['range_width']:.2f}")

    # Test 3: Breakout testi
    print("\n4. Breakout testi (70 mum)...")
    breakout_data = data.head(70)
    result = scanner.calculate(breakout_data)
    print(f"   [OK] Breakout Score: {result.value['breakout_score']}")
    print(f"   [OK] Direction: {result.value['direction']}")
    print(f"   [OK] Volume Confirm: {result.metadata['volume_confirm']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Güç: {result.strength:.2f}")

    # Test 4: Farklı parametreler
    print("\n5. Farklı parametre testi...")
    scanner_fast = BreakoutScanner(lookback=10, confirmation=1)
    result = scanner_fast.calculate(data)
    print(f"   [OK] Fast Scanner Score: {result.value['breakout_score']}")
    print(f"   [OK] Direction: {result.value['direction']}")

    # Test 5: Zaman serisi analizi
    print("\n6. Zaman serisi analizi...")
    scores = []
    directions = []

    for i in range(40, len(data), 5):
        partial_data = data.head(i)
        result = scanner.calculate(partial_data)
        scores.append(result.value['breakout_score'])
        directions.append(result.value['direction'])

    up_count = directions.count('up')
    down_count = directions.count('down')
    none_count = directions.count('none')

    print(f"   [OK] Toplam ölçüm: {len(scores)}")
    print(f"   [OK] Breakout UP: {up_count}")
    print(f"   [OK] Breakout DOWN: {down_count}")
    print(f"   [OK] No Breakout: {none_count}")
    print(f"   [OK] Ortalama score: {np.mean(scores):.2f}")
    print(f"   [OK] Max score: {max(scores):.2f}")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = scanner.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = scanner.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")
    print(f"   [OK] Output names: {metadata.output_names}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
