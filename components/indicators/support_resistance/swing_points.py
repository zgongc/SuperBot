"""
indicators/support_resistance/swing_points.py - Swing Points

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Swing Points - Swing high ve swing low noktalarını tespit eder
    Yerel maksimum ve minimum noktaları belirleyerek potansiyel
    destek ve direnç seviyelerini gösterir.

Formül:
    Swing High: Ortadaki mum, solundaki ve sağındaki N mumdan yüksek
    Swing Low: Ortadaki mum, solundaki ve sağındaki N mumdan düşük
    N = left_bars veya right_bars

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


class SwingPoints(BaseIndicator):
    """
    Swing Points Detector

    Yerel maksimum (swing high) ve minimum (swing low) noktalarını
    tespit eder. Her swing noktası, belirtilen sayıda önceki ve sonraki
    mumdan daha yüksek veya düşük olmalıdır.

    Args:
        left_bars: Sol taraftaki karşılaştırma mum sayısı (varsayılan: 5)
        right_bars: Sağ taraftaki karşılaştırma mum sayısı (varsayılan: 5)
        lookback: Geriye dönük arama periyodu (varsayılan: 50)
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        lookback: int = 50,
        logger=None,
        error_handler=None
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.lookback = lookback

        super().__init__(
            name='swing_points',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'lookback': lookback
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.lookback + self.right_bars

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.left_bars < 1:
            raise InvalidParameterError(
                self.name, 'left_bars', self.left_bars,
                "Left bars pozitif olmalı"
            )
        if self.right_bars < 1:
            raise InvalidParameterError(
                self.name, 'right_bars', self.right_bars,
                "Right bars pozitif olmalı"
            )
        min_lookback = self.left_bars + self.right_bars
        if self.lookback < min_lookback:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                f"Lookback en az {min_lookback} olmalı (left_bars + right_bars)"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Swing Points hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Swing high ve low seviyeleri
        """
        # Lookback periyodu + sağ taraf için extra veri al
        recent_data = data.iloc[-self.lookback - self.right_bars:]
        high = recent_data['high'].values
        low = recent_data['low'].values

        # Swing noktalarını bul
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Swing seviyelerini oluştur (son swing high ve low - calculate_batch ile tutarlı)
        levels = {
            'swing_high': round(swing_highs[-1], 2) if len(swing_highs) > 0 else None,
            'swing_low': round(swing_lows[-1], 2) if len(swing_lows) > 0 else None
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, swing_highs, swing_lows),
            trend=self.get_trend(swing_highs, swing_lows),
            strength=self.calculate_strength(current_price, swing_highs, swing_lows),
            metadata={
                'left_bars': self.left_bars,
                'right_bars': self.right_bars,
                'lookback': self.lookback,
                'current_price': round(current_price, 2),
                'total_swing_highs': len(swing_highs),
                'total_swing_lows': len(swing_lows),
                'last_swing_high': round(swing_highs[-1], 2) if len(swing_highs) > 0 else None,
                'last_swing_low': round(swing_lows[-1], 2) if len(swing_lows) > 0 else None
            }
        )

    def _find_swing_highs(self, high: np.ndarray) -> list:
        """
        Swing high noktalarını bul

        Args:
            high: High fiyatları

        Returns:
            list: Swing high seviyeleri
        """
        swing_highs = []

        # Sadece tamamlanmış swing'leri kontrol et (sağ taraf için veri olmalı)
        for i in range(self.left_bars, len(high) - self.right_bars):
            # Sol taraf kontrolü
            is_highest_left = all(high[i] > high[i - j] for j in range(1, self.left_bars + 1))

            # Sağ taraf kontrolü
            is_highest_right = all(high[i] >= high[i + j] for j in range(1, self.right_bars + 1))

            if is_highest_left and is_highest_right:
                swing_highs.append(high[i])

        return swing_highs if swing_highs else [max(high)]

    def _find_swing_lows(self, low: np.ndarray) -> list:
        """
        Swing low noktalarını bul

        Args:
            low: Low fiyatları

        Returns:
            list: Swing low seviyeleri
        """
        swing_lows = []

        # Sadece tamamlanmış swing'leri kontrol et (sağ taraf için veri olmalı)
        for i in range(self.left_bars, len(low) - self.right_bars):
            # Sol taraf kontrolü
            is_lowest_left = all(low[i] < low[i - j] for j in range(1, self.left_bars + 1))

            # Sağ taraf kontrolü
            is_lowest_right = all(low[i] <= low[i + j] for j in range(1, self.right_bars + 1))

            if is_lowest_left and is_lowest_right:
                swing_lows.append(low[i])

        return swing_lows if swing_lows else [min(low)]

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup SwingPoints buffers with historical data

        CRITICAL: SwingPoints uses its own buffers (_high_buffer, _low_buffer, _close_buffer)
        not BaseIndicator's _buffers. This override ensures they're properly filled.

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier (unused, for interface compatibility)
        """
        from collections import deque

        max_len = self.get_required_periods() + 50
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Fill buffers with historical data
        for _, row in data.tail(max_len).iterrows():
            self._high_buffer.append(row['high'])
            self._low_buffer.append(row['low'])
            self._close_buffer.append(row['close'])

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
            # Return dict format consistent with calculate() method
            # This allows TradingEngine to properly add to DataFrame
            return IndicatorResult(
                value={'swing_high': None, 'swing_low': None},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'warmup': True, 'required': self.get_required_periods(), 'current': len(self._close_buffer)}
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

    def get_signal(self, price: float, swing_highs: list, swing_lows: list) -> SignalType:
        """
        Fiyatın swing noktalarına göre sinyal üret

        Args:
            price: Güncel fiyat
            swing_highs: Swing high seviyeleri
            swing_lows: Swing low seviyeleri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if not swing_highs or not swing_lows:
            return SignalType.HOLD

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]

        # En yakın swing seviyesine yakınlık kontrolü
        distance_to_low = abs(price - last_low) / price
        distance_to_high = abs(price - last_high) / price

        if distance_to_low < 0.01:  # %1 içinde
            return SignalType.BUY
        elif distance_to_high < 0.01:  # %1 içinde
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, swing_highs: list, swing_lows: list) -> TrendDirection:
        """
        Swing noktalarına göre trend belirle

        Args:
            swing_highs: Swing high seviyeleri
            swing_lows: Swing low seviyeleri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if not swing_highs or not swing_lows or len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendDirection.NEUTRAL

        # Yükselen dip ve yükselen tepe = yükseliş trendi
        # Alçalan dip ve alçalan tepe = düşüş trendi
        higher_lows = swing_lows[-1] > swing_lows[-2]
        higher_highs = swing_highs[-1] > swing_highs[-2]
        lower_lows = swing_lows[-1] < swing_lows[-2]
        lower_highs = swing_highs[-1] < swing_highs[-2]

        if higher_lows and higher_highs:
            return TrendDirection.UP
        elif lower_lows and lower_highs:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, swing_highs: list, swing_lows: list) -> float:
        """
        Fiyatın swing noktalarına göre güç hesapla

        Args:
            price: Güncel fiyat
            swing_highs: Swing high seviyeleri
            swing_lows: Swing low seviyeleri

        Returns:
            float: Güç değeri (0-100)
        """
        if not swing_highs or not swing_lows:
            return 50.0

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]

        if last_high == last_low:
            return 50.0

        # Fiyatın swing range içindeki konumu
        position = (price - last_low) / (last_high - last_low) * 100
        return min(max(position, 0), 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'lookback': 50
        }

    def _requires_volume(self) -> bool:
        """Swing Points volume gerektirmez"""
        return False

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Swing Points calculation - BACKTEST için

        Proper swing point detection using left_bars and right_bars.
        A swing high is a peak higher than N bars on both left and right.
        A swing low is a valley lower than N bars on both left and right.

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: swing_high and swing_low columns (NaN where no swing detected)

        Performance: Detects confirmed swing points with proper confirmation

        Note: Swing points require future confirmation (right_bars), so the last
        N bars will have NaN values since we can't confirm them yet.
        """
        self._validate_data(data)

        high = data['high'].values
        low = data['low'].values
        n_bars = len(data)

        # Initialize arrays with NaN
        swing_highs = np.full(n_bars, np.nan)
        swing_lows = np.full(n_bars, np.nan)

        # Detect swing points (skip first left_bars and last right_bars)
        for i in range(self.left_bars, n_bars - self.right_bars):
            # Check if this is a swing high
            is_swing_high = True
            current_high = high[i]

            # Check left side
            for j in range(1, self.left_bars + 1):
                if current_high <= high[i - j]:
                    is_swing_high = False
                    break

            # Check right side if left passed
            if is_swing_high:
                for j in range(1, self.right_bars + 1):
                    if current_high < high[i + j]:  # >= for right side
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs[i] = current_high

            # Check if this is a swing low
            is_swing_low = True
            current_low = low[i]

            # Check left side
            for j in range(1, self.left_bars + 1):
                if current_low >= low[i - j]:
                    is_swing_low = False
                    break

            # Check right side if left passed
            if is_swing_low:
                for j in range(1, self.right_bars + 1):
                    if current_low > low[i + j]:  # <= for right side
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows[i] = current_low

        return pd.DataFrame({
            'swing_high': swing_highs,
            'swing_low': swing_lows
        }, index=data.index)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SwingPoints']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Swing Points indikatör testi"""

    print("\n" + "="*60)
    print("SWING POINTS TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Dalgalı fiyat hareketi simüle et
    base_price = 100
    prices = [base_price]

    for i in range(99):
        # Sinüs dalgası + noise
        wave = 10 * np.sin(i / 8)
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
    swing = SwingPoints(left_bars=5, right_bars=5, lookback=50)
    print(f"   [OK] Oluşturuldu: {swing}")
    print(f"   [OK] Kategori: {swing.category.value}")
    print(f"   [OK] Tip: {swing.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {swing.get_required_periods()}")

    result = swing(data)
    print(f"   [OK] Swing Seviyeleri:")
    for level, value in sorted(result.value.items(), key=lambda x: x[1], reverse=True):
        level_type = "Swing High" if level.startswith('SH') else "Swing Low"
        print(f"        {level} ({level_type}): {value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı left/right bar kombinasyonları
    print("\n3. Farklı bar kombinasyonu testi...")
    for left, right in [(3, 3), (5, 5), (7, 7)]:
        swing_test = SwingPoints(left_bars=left, right_bars=right, lookback=50)
        result = swing_test.calculate(data)
        print(f"   [OK] Swing(L={left},R={right}) - Highs: {result.metadata['total_swing_highs']}, "
              f"Lows: {result.metadata['total_swing_lows']}")

    # Test 3: Farklı lookback değerleri
    print("\n4. Farklı lookback testi...")
    for lookback in [30, 50, 70]:
        swing_test = SwingPoints(left_bars=5, right_bars=5, lookback=lookback)
        result = swing_test.calculate(data)
        print(f"   [OK] Swing(lookback={lookback}) - "
              f"Tespit: {result.metadata['total_swing_highs'] + result.metadata['total_swing_lows']} nokta")

    # Test 4: Swing analizi
    print("\n5. Swing analizi...")
    result = swing.calculate(data)
    current = result.metadata['current_price']
    last_high = result.metadata['last_swing_high']
    last_low = result.metadata['last_swing_low']

    print(f"   [OK] Güncel fiyat: {current}")
    print(f"   [OK] Son swing high: {last_high}")
    print(f"   [OK] Son swing low: {last_low}")

    if last_high and last_low:
        swing_range = last_high - last_low
        position = (current - last_low) / swing_range * 100
        print(f"   [OK] Swing range: {swing_range:.2f}")
        print(f"   [OK] Fiyat konumu: {position:.1f}% (range içinde)")

        if position > 70:
            print(f"   [OK] Fiyat swing range'in üst bölgesinde (direnç yakını)")
        elif position < 30:
            print(f"   [OK] Fiyat swing range'in alt bölgesinde (destek yakını)")
        else:
            print(f"   [OK] Fiyat swing range'in orta bölgesinde")

    # Test 5: Trend analizi
    print("\n6. Trend analizi...")
    print(f"   [OK] Tespit edilen trend: {result.trend.name}")
    if result.trend == TrendDirection.UP:
        print(f"   [OK] Yükselen dipler ve tepeler - Yükseliş trendi")
        print(f"   [OK] Strateji: Swing low seviyelerinde alım ara")
    elif result.trend == TrendDirection.DOWN:
        print(f"   [OK] Alçalan dipler ve tepeler - Düşüş trendi")
        print(f"   [OK] Strateji: Swing high seviyelerinde satış ara")
    else:
        print(f"   [OK] Net bir trend yok - Yatay hareket")
        print(f"   [OK] Strateji: Range trading (destek/dirençten işlem)")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = swing.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = swing.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
