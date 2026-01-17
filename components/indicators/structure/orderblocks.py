"""
indicators/structure/orderblocks.py - Order Blocks

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Order Blocks - Smart Money Concepts
    Kurumsal emirlerin bıraktığı izleri tespit eder

    Order Block Nedir:
    - Güçlü fiyat hareketinden önceki son zıt hareket mumunun bölgesi
    - Smart Money'nin emir bıraktığı bölge
    - Güçlü destek/direnç oluşturur

Formül:
    Bullish Order Block:
    1. Güçlü yükseliş hareketi tespit et (threshold üzeri)
    2. Bu hareketten önceki son düşüş mumunu bul
    3. O mumun low-high aralığı = Order Block

    Bearish Order Block:
    1. Güçlü düşüş hareketi tespit et
    2. Bu hareketten önceki son yükseliş mumunu bul
    3. O mumun high-low aralığı = Order Block

    Test & Validation:
    - Fiyat OB'ye döndüğünde genellikle reaksiyon alır
    - Break edilirse geçerliliğini kaybeder

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


class OrderBlocks(BaseIndicator):
    """
    Order Blocks (OB)

    Kurumsal emirlerin bıraktığı bölgeleri tespit eder.
    Güçlü destek/direnç seviyeleri olarak kullanılır.

    Args:
        strength_threshold: Güçlü hareket eşiği (% değişim) (varsayılan: 1.0)
        max_blocks: Maksimum aktif block sayısı (varsayılan: 5)
        lookback: Geriye bakış periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        strength_threshold: float = 1.0,
        max_blocks: int = 5,
        lookback: int = 20,
        logger=None,
        error_handler=None
    ):
        self.strength_threshold = strength_threshold
        self.max_blocks = max_blocks
        self.lookback = lookback

        super().__init__(
            name='orderblocks',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'strength_threshold': strength_threshold,
                'max_blocks': max_blocks,
                'lookback': lookback
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Aktif Order Block'ları takip et
        self.active_blocks: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.lookback + 5

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.strength_threshold <= 0:
            raise InvalidParameterError(
                self.name, 'strength_threshold', self.strength_threshold,
                "Strength threshold pozitif olmalı"
            )
        if self.max_blocks < 1:
            raise InvalidParameterError(
                self.name, 'max_blocks', self.max_blocks,
                "Max blocks pozitif olmalı"
            )
        if self.lookback < 5:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback en az 5 olmalı"
            )
        return True

    def _is_bullish_candle(self, open_price: float, close_price: float) -> bool:
        """Mum bullish mi?"""
        return close_price > open_price

    def _is_bearish_candle(self, open_price: float, close_price: float) -> bool:
        """Mum bearish mi?"""
        return close_price < open_price

    def _calculate_candle_change(self, open_price: float, close_price: float) -> float:
        """Mum değişim yüzdesi"""
        if open_price == 0:
            return 0.0
        return ((close_price - open_price) / open_price) * 100

    def _detect_strong_moves(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Güçlü fiyat hareketlerini tespit et

        Args:
            opens: Open fiyat dizisi
            closes: Close fiyat dizisi
            highs: High fiyat dizisi
            lows: Low fiyat dizisi

        Returns:
            List[Dict]: Güçlü hareketler
        """
        strong_moves = []

        # Son lookback period içinde tara
        start_idx = max(0, len(closes) - self.lookback)

        for i in range(start_idx, len(closes)):
            change_percent = self._calculate_candle_change(opens[i], closes[i])

            # Güçlü yükseliş
            if change_percent >= self.strength_threshold:
                strong_moves.append({
                    'type': 'bullish',
                    'index': i,
                    'change_percent': change_percent,
                    'high': highs[i],
                    'low': lows[i]
                })

            # Güçlü düşüş
            elif change_percent <= -self.strength_threshold:
                strong_moves.append({
                    'type': 'bearish',
                    'index': i,
                    'change_percent': change_percent,
                    'high': highs[i],
                    'low': lows[i]
                })

        return strong_moves

    def _find_order_block(
        self,
        move: Dict[str, Any],
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Dict[str, Any]:
        """
        Güçlü hareketten order block çıkar

        Args:
            move: Güçlü hareket bilgisi
            opens, closes, highs, lows: Fiyat dizileri

        Returns:
            Dict: Order block bilgisi
        """
        move_index = move['index']
        move_type = move['type']

        # Geriye doğru tara
        for i in range(move_index - 1, max(0, move_index - 10), -1):
            is_bullish = self._is_bullish_candle(opens[i], closes[i])
            is_bearish = self._is_bearish_candle(opens[i], closes[i])

            # Bullish move için son bearish mum
            if move_type == 'bullish' and is_bearish:
                return {
                    'type': 'bullish',
                    'top': highs[i],
                    'bottom': lows[i],
                    'index': i,
                    'move_index': move_index,
                    'strength': move['change_percent'],
                    'status': 'active',
                    'test_count': 0
                }

            # Bearish move için son bullish mum
            if move_type == 'bearish' and is_bullish:
                return {
                    'type': 'bearish',
                    'top': highs[i],
                    'bottom': lows[i],
                    'index': i,
                    'move_index': move_index,
                    'strength': abs(move['change_percent']),
                    'status': 'active',
                    'test_count': 0
                }

        return None

    def _update_block_status(
        self,
        block: Dict[str, Any],
        current_high: float,
        current_low: float,
        current_close: float
    ) -> Dict[str, Any]:
        """
        Order block durumunu güncelle

        Args:
            block: Order block bilgisi
            current_high, current_low, current_close: Güncel fiyatlar

        Returns:
            Dict: Güncellenmiş block
        """
        # Fiyat block içine girdi mi?
        in_block = (current_low <= block['top'] and current_high >= block['bottom'])

        if in_block:
            block['test_count'] += 1

        # Block kırıldı mı?
        if block['type'] == 'bullish':
            # Bullish OB altına kapanış -> broken
            if current_close < block['bottom']:
                block['status'] = 'broken'
        else:
            # Bearish OB üstüne kapanış -> broken
            if current_close > block['top']:
                block['status'] = 'broken'

        return block

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Order Blocks hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Order Block zones
        """
        opens = data['open'].values
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values

        # Güçlü hareketleri tespit et
        strong_moves = self._detect_strong_moves(opens, closes, highs, lows)

        # Her güçlü hareketten order block çıkar
        new_blocks = []
        for move in strong_moves:
            ob = self._find_order_block(move, opens, closes, highs, lows)
            if ob:
                # Duplicate kontrolü
                is_duplicate = any(
                    existing['index'] == ob['index']
                    for existing in self.active_blocks
                )
                if not is_duplicate:
                    new_blocks.append(ob)

        # Yeni block'ları ekle
        self.active_blocks.extend(new_blocks)

        # Mevcut block'ların durumunu güncelle
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]

        for block in self.active_blocks:
            self._update_block_status(block, current_high, current_low, current_close)

        # Kırılmış block'ları kaldır
        self.active_blocks = [
            block for block in self.active_blocks
            if block['status'] == 'active'
        ]

        # Maksimum block sayısını uygula (en güçlü olanları tut)
        if len(self.active_blocks) > self.max_blocks:
            self.active_blocks.sort(key=lambda x: x['strength'], reverse=True)
            self.active_blocks = self.active_blocks[:self.max_blocks]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Değer: Aktif order block'ların listesi
        zones = [
            {
                'type': block['type'],
                'top': round(block['top'], 2),
                'bottom': round(block['bottom'], 2),
                'strength': round(block['strength'], 2),
                'test_count': block['test_count'],
                'status': block['status']
            }
            for block in self.active_blocks
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=zones,
            timestamp=timestamp,
            signal=self.get_signal(zones, current_close),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),  # Her block 20 puan
            metadata={
                'total_blocks': len(zones),
                'bullish_blocks': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_blocks': len([z for z in zones if z['type'] == 'bearish']),
                'strength_threshold': self.strength_threshold,
                'max_blocks': self.max_blocks
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Güncel Order Blocks
        """
        # Buffer yönetimi
        if not hasattr(self, '_open_buffer'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._open_buffer = deque(maxlen=max_len)
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

        self._open_buffer.append(open_val)
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
                metadata={'total_blocks': 0}
            )
            
        # Hesaplama
        opens = np.array(self._open_buffer)
        closes = np.array(self._close_buffer)
        highs = np.array(self._high_buffer)
        lows = np.array(self._low_buffer)
        
        # Güçlü hareketleri tespit et
        strong_moves = self._detect_strong_moves(opens, closes, highs, lows)
        
        # Her güçlü hareketten order block çıkar
        new_blocks = []
        for move in strong_moves:
            ob = self._find_order_block(move, opens, closes, highs, lows)
            if ob:
                # Duplicate kontrolü
                is_duplicate = any(
                    existing['index'] == ob['index']
                    for existing in self.active_blocks
                )
                if not is_duplicate:
                    new_blocks.append(ob)
        
        # Yeni block'ları ekle
        self.active_blocks.extend(new_blocks)
        
        # Mevcut block'ların durumunu güncelle
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]
        
        for block in self.active_blocks:
            self._update_block_status(block, current_high, current_low, current_close)
        
        # Kırılmış block'ları kaldır
        self.active_blocks = [
            block for block in self.active_blocks
            if block['status'] == 'active'
        ]
        
        # Maksimum block sayısını uygula
        if len(self.active_blocks) > self.max_blocks:
            self.active_blocks.sort(key=lambda x: x['strength'], reverse=True)
            self.active_blocks = self.active_blocks[:self.max_blocks]
        
        # Değer: Aktif order block'lar
        zones = [
            {
                'type': block['type'],
                'top': round(block['top'], 2),
                'bottom': round(block['bottom'], 2),
                'strength': round(block['strength'], 2),
                'test_count': block['test_count'],
                'status': block['status']
            }
            for block in self.active_blocks
        ]

        return IndicatorResult(
            value=zones,
            timestamp=timestamp_val,
            signal=self.get_signal(zones, current_close),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),
            metadata={
                'total_blocks': len(zones),
                'bullish_blocks': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_blocks': len([z for z in zones if z['type'] == 'bearish']),
                'strength_threshold': self.strength_threshold,
                'max_blocks': self.max_blocks
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

    def get_signal(self, zones: List[Dict[str, Any]], current_price: float) -> SignalType:
        """
        Order Block'lardan sinyal üret

        Args:
            zones: Order block zone'ları
            current_price: Güncel fiyat

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if not zones:
            return SignalType.HOLD

        # Fiyat bir OB zone'una yakın mı?
        for zone in zones:
            # Zone ortası
            zone_mid = (zone['top'] + zone['bottom']) / 2
            distance = abs(current_price - zone_mid)
            zone_size = zone['top'] - zone['bottom']

            # Zone içindeyse veya %2 yakınındaysa
            if distance <= zone_size * 0.5 or (distance / current_price) * 100 < 2.0:
                if zone['type'] == 'bullish':
                    return SignalType.BUY  # Bullish OB'de destek bekle
                elif zone['type'] == 'bearish':
                    return SignalType.SELL  # Bearish OB'de direnç bekle

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        Order Block'lardan trend belirle

        Args:
            zones: Order block zone'ları

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if not zones:
            return TrendDirection.NEUTRAL

        bullish_count = len([z for z in zones if z['type'] == 'bullish'])
        bearish_count = len([z for z in zones if z['type'] == 'bearish'])

        # Güçlü block'lar daha fazla ağırlık taşır
        bullish_strength = sum(
            z['strength'] for z in zones if z['type'] == 'bullish'
        )
        bearish_strength = sum(
            z['strength'] for z in zones if z['type'] == 'bearish'
        )

        if bullish_strength > bearish_strength * 1.2:
            return TrendDirection.UP
        elif bearish_strength > bullish_strength * 1.2:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'strength_threshold': 1.0,
            'max_blocks': 5,
            'lookback': 20
        }

    def _requires_volume(self) -> bool:
        """Order Blocks volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['OrderBlocks']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Order Blocks indikatör testi"""

    print("\n" + "="*60)
    print("ORDER BLOCKS TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Güçlü hareketli fiyat simülasyonu
    base_price = 100
    prices = []
    opens = []
    highs = []
    lows = []

    for i in range(50):
        if i == 15:
            # Güçlü yükseliş (Order Block oluşturur)
            open_p = base_price
            close_p = base_price + 2.5  # %2.5 yükseliş
            opens.append(open_p)
            prices.append(close_p)
            highs.append(close_p + 0.2)
            lows.append(open_p - 0.1)
            base_price = close_p
        elif i == 35:
            # Güçlü düşüş (Order Block oluşturur)
            open_p = base_price
            close_p = base_price - 2.0  # %2 düşüş
            opens.append(open_p)
            prices.append(close_p)
            highs.append(open_p + 0.1)
            lows.append(close_p - 0.2)
            base_price = close_p
        else:
            # Normal hareket
            open_p = base_price
            close_p = base_price + np.random.randn() * 0.3
            opens.append(open_p)
            prices.append(close_p)
            highs.append(max(open_p, close_p) + abs(np.random.randn()) * 0.2)
            lows.append(min(open_p, close_p) - abs(np.random.randn()) * 0.2)
            base_price = close_p

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    ob = OrderBlocks(strength_threshold=1.0, max_blocks=5, lookback=20)
    print(f"   [OK] Oluşturuldu: {ob}")
    print(f"   [OK] Kategori: {ob.category.value}")
    print(f"   [OK] Gerekli periyot: {ob.get_required_periods()}")

    result = ob(data)
    print(f"   [OK] Toplam Block: {result.metadata['total_blocks']}")
    print(f"   [OK] Bullish Block: {result.metadata['bullish_blocks']}")
    print(f"   [OK] Bearish Block: {result.metadata['bearish_blocks']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength}")

    # Test 2: Block detayları
    print("\n3. Block detayları...")
    if result.value:
        for i, block in enumerate(result.value[:3]):
            print(f"   [OK] Block #{i+1}:")
            print(f"       - Tip: {block['type']}")
            print(f"       - Top: {block['top']:.2f}")
            print(f"       - Bottom: {block['bottom']:.2f}")
            print(f"       - Güç: {block['strength']:.2f}%")
            print(f"       - Test: {block['test_count']} kez")
            print(f"       - Durum: {block['status']}")
    else:
        print("   [OK] Aktif block bulunamadı")

    # Test 3: Farklı parametreler
    print("\n4. Farklı parametre testi...")
    for threshold in [0.5, 1.0, 1.5]:
        ob_test = OrderBlocks(strength_threshold=threshold)
        result = ob_test.calculate(data)
        print(f"   [OK] OB(threshold={threshold}): {result.metadata['total_blocks']} blocks")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = ob.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = ob.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
