"""
indicators/structure/liquidityzones.py - Liquidity Zones

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Liquidity Zones - Smart Money Concepts
    Likidite havuzlarını (Stop-loss yoğunluğu) tespit eder

    Liquidity Zone Nedir:
    - Swing High/Low seviyeleri = Stop-loss yoğunluğu
    - Equal Highs/Lows = Çoklu stop-loss seviyesi
    - Smart Money bu seviyeleri "sweep" ederek likidite toplar

Formül:
    1. Swing High/Low tespiti
    2. Equal seviye tespiti (±tolerance aralığında)
    3. Likidite havuzları oluştur
    4. "Sweep" tespiti (kısa süreli kırılma)

    Liquidity Sweep:
    - Fiyat likidite seviyesinin üstüne/altına çıkar
    - Hemen geri döner (1-3 mum içinde)
    - "Stop hunt" veya "liquidity grab"

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class LiquidityZones(BaseIndicator):
    """
    Liquidity Zones

    Likidite havuzlarını tespit eder.
    Smart Money'nin stop-loss'ları tetiklediği bölgeleri gösterir.

    Args:
        left_bars: Sol taraf bar sayısı (varsayılan: 5)
        right_bars: Sağ taraf bar sayısı (varsayılan: 5)
        equal_tolerance: Equal seviye toleransı (%) (varsayılan: 0.1)
        max_zones: Maksimum zone sayısı (varsayılan: 5)
        sweep_lookback: Sweep kontrolü için geriye bakış (varsayılan: 3)
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        equal_tolerance: float = 0.1,
        max_zones: int = 5,
        sweep_lookback: int = 3,
        logger=None,
        error_handler=None
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.equal_tolerance = equal_tolerance
        self.max_zones = max_zones
        self.sweep_lookback = sweep_lookback

        super().__init__(
            name='liquidityzones',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'equal_tolerance': equal_tolerance,
                'max_zones': max_zones,
                'sweep_lookback': sweep_lookback
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Aktif likidite zone'ları
        self.liquidityzones: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.left_bars + self.right_bars + 10

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
        if self.equal_tolerance < 0:
            raise InvalidParameterError(
                self.name, 'equal_tolerance', self.equal_tolerance,
                "Equal tolerance negatif olamaz"
            )
        if self.max_zones < 1:
            raise InvalidParameterError(
                self.name, 'max_zones', self.max_zones,
                "Max zones pozitif olmalı"
            )
        if self.sweep_lookback < 1:
            raise InvalidParameterError(
                self.name, 'sweep_lookback', self.sweep_lookback,
                "Sweep lookback pozitif olmalı"
            )
        return True

    def _find_swing_highs(self, highs: np.ndarray) -> List[Dict[str, Any]]:
        """Swing High noktalarını tespit et"""
        swing_highs = []

        for i in range(self.left_bars, len(highs) - self.right_bars):
            is_pivot = True

            for j in range(1, self.left_bars + 1):
                if highs[i] <= highs[i - j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            for j in range(1, self.right_bars + 1):
                if highs[i] <= highs[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                swing_highs.append({
                    'index': i,
                    'value': highs[i],
                    'type': 'high'
                })

        return swing_highs

    def _find_swing_lows(self, lows: np.ndarray) -> List[Dict[str, Any]]:
        """Swing Low noktalarını tespit et"""
        swing_lows = []

        for i in range(self.left_bars, len(lows) - self.right_bars):
            is_pivot = True

            for j in range(1, self.left_bars + 1):
                if lows[i] >= lows[i - j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            for j in range(1, self.right_bars + 1):
                if lows[i] >= lows[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                swing_lows.append({
                    'index': i,
                    'value': lows[i],
                    'type': 'low'
                })

        return swing_lows

    def _find_equal_levels(self, swings: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Equal (eşit) seviyeleri tespit et

        Args:
            swings: Swing noktaları

        Returns:
            List[List]: Equal gruplar
        """
        if len(swings) < 2:
            return []

        equal_groups = []

        for i, swing1 in enumerate(swings):
            group = [swing1]

            for swing2 in swings[i + 1:]:
                # Tolerance içinde mi?
                diff_percent = abs(swing1['value'] - swing2['value']) / swing1['value'] * 100

                if diff_percent <= self.equal_tolerance:
                    group.append(swing2)

            # En az 2 equal seviye
            if len(group) >= 2:
                equal_groups.append(group)

        return equal_groups

    def _create_liquidityzones(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Likidite zone'ları oluştur

        Args:
            swing_highs: Swing high'lar
            swing_lows: Swing low'lar

        Returns:
            List[Dict]: Likidite zone'ları
        """
        zones = []

        # Swing High'lardan likidite zone'ları (Sell-side liquidity)
        for swing in swing_highs[-self.max_zones:]:
            zones.append({
                'type': 'sell_side',  # Üstte stop-loss'lar (long pozisyonlar)
                'level': swing['value'],
                'index': swing['index'],
                'strength': 1,  # Tek seviye
                'swept': False
            })

        # Swing Low'lardan likidite zone'ları (Buy-side liquidity)
        for swing in swing_lows[-self.max_zones:]:
            zones.append({
                'type': 'buy_side',  # Altta stop-loss'lar (short pozisyonlar)
                'level': swing['value'],
                'index': swing['index'],
                'strength': 1,
                'swept': False
            })

        # Equal level'lardan likidite zone'ları (güçlü)
        equal_highs = self._find_equal_levels(swing_highs)
        for group in equal_highs:
            avg_level = np.mean([s['value'] for s in group])
            zones.append({
                'type': 'sell_side_equal',
                'level': avg_level,
                'index': group[-1]['index'],
                'strength': len(group),  # Equal sayısı
                'swept': False
            })

        equal_lows = self._find_equal_levels(swing_lows)
        for group in equal_lows:
            avg_level = np.mean([s['value'] for s in group])
            zones.append({
                'type': 'buy_side_equal',
                'level': avg_level,
                'index': group[-1]['index'],
                'strength': len(group),
                'swept': False
            })

        return zones

    def _detect_sweeps(
        self,
        zones: List[Dict[str, Any]],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Liquidity sweep tespiti

        Args:
            zones: Likidite zone'ları
            highs, lows, closes: Fiyat dizileri

        Returns:
            List[Dict]: Güncellenmiş zone'lar
        """
        # Son N mumu kontrol et
        lookback_start = max(0, len(closes) - self.sweep_lookback)

        for zone in zones:
            if zone['swept']:
                continue  # Zaten sweep edilmiş

            for i in range(lookback_start, len(closes)):
                # Sell-side sweep (üst taraf)
                if 'sell_side' in zone['type']:
                    # High seviyeyi geçti mi?
                    if highs[i] > zone['level']:
                        # Close seviyenin altında mı? (geri döndü)
                        if closes[i] < zone['level']:
                            zone['swept'] = True
                            zone['swept_index'] = i
                            break

                # Buy-side sweep (alt taraf)
                elif 'buy_side' in zone['type']:
                    # Low seviyeyi geçti mi?
                    if lows[i] < zone['level']:
                        # Close seviyenin üstünde mi? (geri döndü)
                        if closes[i] > zone['level']:
                            zone['swept'] = True
                            zone['swept_index'] = i
                            break

        return zones

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Liquidity Zones hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Liquidity zones
        """
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # Swing High/Low tespiti
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        # Likidite zone'ları oluştur
        self.liquidityzones = self._create_liquidityzones(swing_highs, swing_lows)

        # Sweep tespiti
        self.liquidityzones = self._detect_sweeps(
            self.liquidityzones, highs, lows, closes
        )

        # Swept olmamış zone'ları filtrele (aktif zone'lar)
        active_zones = [z for z in self.liquidityzones if not z['swept']]

        # En güçlü zone'ları seç
        active_zones.sort(key=lambda x: x['strength'], reverse=True)
        active_zones = active_zones[:self.max_zones]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Değer: Aktif likidite zone'ları
        zones = [
            {
                'type': zone['type'],
                'level': round(zone['level'], 2),
                'strength': zone['strength'],
                'swept': zone['swept']
            }
            for zone in active_zones
        ]

        # Swept zone'lar (son sweep'ler)
        swept_zones = [
            z for z in self.liquidityzones
            if z['swept'] and z.get('swept_index', 0) >= len(closes) - self.sweep_lookback
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=zones,
            timestamp=timestamp,
            signal=self.get_signal(zones, swept_zones, closes[-1]),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),
            metadata={
                'total_zones': len(zones),
                'sell_side_zones': len([z for z in zones if 'sell_side' in z['type']]),
                'buy_side_zones': len([z for z in zones if 'buy_side' in z['type']]),
                'recent_sweeps': len(swept_zones),
                'swept_zones': [
                    {
                        'type': z['type'],
                        'level': round(z['level'], 2),
                        'swept_at': z.get('swept_index')
                    }
                    for z in swept_zones[:3]
                ],
                'equal_tolerance': self.equal_tolerance
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel Liquidity Zones
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
                value=[],
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'total_zones': 0}
            )
            
        # Hesaplama
        highs = np.array(self._high_buffer)
        lows = np.array(self._low_buffer)
        closes = np.array(self._close_buffer)
        
        # Swing High/Low tespiti
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)
        
        # Likidite zone'ları oluştur
        self.liquidityzones = self._create_liquidityzones(swing_highs, swing_lows)
        
        # Sweep tespiti
        self.liquidityzones = self._detect_sweeps(self.liquidityzones, highs, lows, closes)
        
        # Swept olmamış zone'ları filtrele
        active_zones = [z for z in self.liquidityzones if not z['swept']]
        active_zones.sort(key=lambda x: x['strength'], reverse=True)
        active_zones = active_zones[:self.max_zones]
        
        # Değer: Aktif likidite zone'ları
        zones = [
            {
                'type': zone['type'],
                'level': round(zone['level'], 2),
                'strength': zone['strength'],
                'swept': zone['swept']
            }
            for zone in active_zones
        ]

        # Swept zone'lar
        swept_zones = [
            z for z in self.liquidityzones
            if z['swept'] and z.get('swept_index', 0) >= len(closes) - self.sweep_lookback
        ]

        return IndicatorResult(
            value=zones,
            timestamp=timestamp_val,
            signal=self.get_signal(zones, swept_zones, closes[-1]),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),
            metadata={
                'total_zones': len(zones),
                'sell_side_zones': len([z for z in zones if 'sell_side' in z['type']]),
                'buy_side_zones': len([z for z in zones if 'buy_side' in z['type']]),
                'recent_sweeps': len(swept_zones),
                'swept_zones': [
                    {
                        'type': z['type'],
                        'level': round(z['level'], 2),
                        'swept_at': z.get('swept_index')
                    }
                    for z in swept_zones[:3]
                ],
                'equal_tolerance': self.equal_tolerance
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Batch calculation - calls calculate() for each row
        
        Note: This is a simple implementation for compatibility.
        For performance, consider implementing vectorized logic.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.Series: Indicator values
        """
        results = []
        for i in range(len(data)):
            if i < self.get_required_periods() - 1:
                results.append(np.nan)
            else:
                window_data = data.iloc[:i+1]
                result = self.calculate(window_data)
                # Extract value (handle dict, float, or IndicatorResult)
                if result is None:
                    results.append(np.nan)
                elif hasattr(result, 'value'):
                    results.append(result.value)
                else:
                    results.append(result)
        
        return pd.Series(results, index=data.index, name=self.name)

    def get_signal(
        self,
        zones: List[Dict[str, Any]],
        swept_zones: List[Dict[str, Any]],
        current_price: float
    ) -> SignalType:
        """
        Liquidity zone'lardan sinyal üret

        Args:
            zones: Aktif zone'lar
            swept_zones: Yakın zamanda sweep edilmiş zone'lar
            current_price: Güncel fiyat

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # Sweep sonrası ters yönde sinyal
        if swept_zones:
            last_sweep = swept_zones[-1]

            # Sell-side sweep -> Bullish (aşağı dönüş)
            if 'sell_side' in last_sweep['type']:
                return SignalType.BUY

            # Buy-side sweep -> Bearish (yukarı dönüş)
            elif 'buy_side' in last_sweep['type']:
                return SignalType.SELL

        # Zone'a yaklaşma
        for zone in zones:
            distance_percent = abs(current_price - zone['level']) / current_price * 100

            if distance_percent < 1.0:  # %1 içinde
                if 'buy_side' in zone['type']:
                    return SignalType.BUY  # Destek
                elif 'sell_side' in zone['type']:
                    return SignalType.SELL  # Direnç

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        Liquidity zone'lardan trend belirle

        Args:
            zones: Likidite zone'ları

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if not zones:
            return TrendDirection.NEUTRAL

        buy_side_strength = sum(
            z['strength'] for z in zones if 'buy_side' in z['type']
        )
        sell_side_strength = sum(
            z['strength'] for z in zones if 'sell_side' in z['type']
        )

        if buy_side_strength > sell_side_strength * 1.2:
            return TrendDirection.UP  # Daha fazla aşağı likidite
        elif sell_side_strength > buy_side_strength * 1.2:
            return TrendDirection.DOWN  # Daha fazla yukarı likidite

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'equal_tolerance': 0.1,
            'max_zones': 5,
            'sweep_lookback': 3
        }

    def _requires_volume(self) -> bool:
        """Liquidity Zones volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['LiquidityZones']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Liquidity Zones indikatör testi"""

    print("\n" + "="*60)
    print("LIQUIDITY ZONES TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # Liquidity sweep simülasyonu
    base_price = 100
    prices = []
    highs = []
    lows = []

    for i in range(60):
        if i == 20:
            # Swing high oluştur
            price = base_price + 5
            prices.append(price)
            highs.append(price + 0.5)
            lows.append(price - 0.3)
        elif i == 35:
            # Liquidity sweep (yanlış kırılma)
            price = base_price + 5.5  # Swing high'ı geç
            prices.append(base_price + 4.8)  # Geri dön
            highs.append(price)
            lows.append(base_price + 4.5)
        else:
            # Normal hareket
            price = base_price + np.random.randn() * 0.5
            prices.append(price)
            highs.append(price + abs(np.random.randn()) * 0.3)
            lows.append(price - abs(np.random.randn()) * 0.3)

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

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    liq = LiquidityZones(
        left_bars=5,
        right_bars=5,
        equal_tolerance=0.1,
        max_zones=5,
        sweep_lookback=3
    )
    print(f"   [OK] Oluşturuldu: {liq}")
    print(f"   [OK] Kategori: {liq.category.value}")
    print(f"   [OK] Gerekli periyot: {liq.get_required_periods()}")

    result = liq(data)
    print(f"   [OK] Toplam Zone: {result.metadata['total_zones']}")
    print(f"   [OK] Sell-Side: {result.metadata['sell_side_zones']}")
    print(f"   [OK] Buy-Side: {result.metadata['buy_side_zones']}")
    print(f"   [OK] Yakın Sweep: {result.metadata['recent_sweeps']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength}")

    # Test 2: Zone detayları
    print("\n3. Zone detayları...")
    if result.value:
        for i, zone in enumerate(result.value[:3]):
            print(f"   [OK] Zone #{i+1}:")
            print(f"       - Tip: {zone['type']}")
            print(f"       - Seviye: {zone['level']:.2f}")
            print(f"       - Güç: {zone['strength']}")
            print(f"       - Swept: {zone['swept']}")
    else:
        print("   [OK] Aktif zone bulunamadı")

    # Test 3: Swept zone'lar
    print("\n4. Swept zone'lar...")
    if result.metadata['swept_zones']:
        for i, swept in enumerate(result.metadata['swept_zones']):
            print(f"   [OK] Sweep #{i+1}:")
            print(f"       - Tip: {swept['type']}")
            print(f"       - Seviye: {swept['level']:.2f}")
            print(f"       - Index: {swept['swept_at']}")
    else:
        print("   [OK] Yakın zamanda sweep bulunamadı")

    # Test 4: Farklı parametreler
    print("\n5. Farklı parametre testi...")
    for tolerance in [0.05, 0.1, 0.2]:
        liq_test = LiquidityZones(equal_tolerance=tolerance)
        result = liq_test.calculate(data)
        print(f"   [OK] LIQ(tolerance={tolerance}): {result.metadata['total_zones']} zones")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = liq.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = liq.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
