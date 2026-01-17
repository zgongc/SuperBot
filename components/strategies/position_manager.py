#!/usr/bin/env python3
"""
components/strategies/position_manager.py
SuperBot - Position Manager (NEW DESIGN)
Yazar: SuperBot Team
Tarih: 2025-12-07
Versiyon: 2.0.0

Position Manager - Strategy template parametrelerini işleyen position yönetimi

ÖNEMLİ: Bu yeni tasarım, eski 819 satırlık kullanılmayan position_manager.py'nin yerini alır.
Amaç: BacktestEngine ve TradingEngine'deki duplicate inline kodları merkezi bir yere taşımak.

İşlenen Parametreler (simple_rsi.py PositionManagement'tan):
- max_positions_per_symbol: Symbol başına max pozisyon sayısı
- pyramiding_enabled: Aynı yönde birden fazla pozisyon açılabilir mi
- pyramiding_max_entries: Max pyramiding entry sayısı
- pyramiding_scale_factor: Sonraki entry'lerin size çarpanı
- allow_hedging: Zıt yönde pozisyon açılabilir mi (hedge)
- position_timeout_enabled: Pozisyon timeout aktif mi
- position_timeout: Timeout süresi (dakika)

Kullanım:
    pm = PositionManager(strategy, logger)

    # Pozisyon açılabilir mi kontrolü
    can_open, reason, scale = pm.can_open_position(
        symbol='BTCUSDT',
        side='LONG',
        positions=current_positions
    )

    # Timeout kontrolü
    should_close, reason = pm.check_position_timeout(position, current_timestamp)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class PositionOpenResult:
    """Pozisyon açma kontrolü sonucu"""
    can_open: bool
    reason: str
    pyramiding_scale: float = 1.0  # Pyramiding için size scale factor
    should_close_opposite: bool = False  # Zıt pozisyonlar kapatılmalı mı (one-way mode)
    opposite_positions: List[Dict] = None  # Kapatılacak zıt pozisyonlar

    def __post_init__(self):
        if self.opposite_positions is None:
            self.opposite_positions = []


class PositionManager:
    """
    Position Manager - Strategy template parametrelerini işler

    Workflow:
    1. can_open_position() - Pozisyon açılabilir mi kontrol et
    2. check_position_timeout() - Timeout kontrolü
    3. get_positions_for_symbol() - Symbol için pozisyonları getir

    NOT: Bu sınıf sadece KARAR VERİR, pozisyon açma/kapama işlemini engine yapar.
    """

    def __init__(
        self,
        strategy: Any,
        logger: Optional[Any] = None
    ):
        """
        Args:
            strategy: BaseStrategy instance (position_management parametreleri için)
            logger: Logger instance
        """
        self.strategy = strategy
        self.logger = logger

        # Cache strategy parameters
        pm = strategy.position_management
        self.max_positions_per_symbol = pm.max_positions_per_symbol
        self.pyramiding_enabled = pm.pyramiding_enabled
        self.pyramiding_max_entries = pm.pyramiding_max_entries
        self.pyramiding_scale_factor = pm.pyramiding_scale_factor
        self.allow_hedging = pm.allow_hedging
        self.position_timeout_enabled = pm.position_timeout_enabled
        self.position_timeout = pm.position_timeout  # Minutes

        if self.logger:
            self.logger.debug(
                f"PositionManager initialized: max_per_symbol={self.max_positions_per_symbol}, "
                f"pyramiding={self.pyramiding_enabled}, hedging={self.allow_hedging}"
            )

    # ========================================================================
    # MAIN METHODS - Engine'ler bu methodları kullanır
    # ========================================================================

    def can_open_position(
        self,
        symbol: str,
        side: str,
        positions: List[Dict]
    ) -> PositionOpenResult:
        """
        Pozisyon açılabilir mi kontrol et

        Bu method BacktestEngine (line 825-939) ve TradingEngine (line 822-864)
        içindeki duplicate kodun yerine geçer.

        Args:
            symbol: Sembol (örn: 'BTCUSDT')
            side: 'LONG' veya 'SHORT'
            positions: Mevcut açık pozisyonlar listesi
                      Her pozisyon dict: {'symbol': str, 'side': str, ...}

        Returns:
            PositionOpenResult:
                - can_open: Pozisyon açılabilir mi
                - reason: Açıklama
                - pyramiding_scale: Pyramiding için size çarpanı (1.0 = normal)
                - should_close_opposite: Zıt pozisyonlar kapatılmalı mı
                - opposite_positions: Kapatılacak zıt pozisyon listesi

        Örnek:
            >>> result = pm.can_open_position('BTCUSDT', 'LONG', current_positions)
            >>> if result.can_open:
            ...     if result.should_close_opposite:
            ...         for pos in result.opposite_positions:
            ...             engine.close_position(pos)
            ...     quantity = base_quantity * result.pyramiding_scale
            ...     engine.open_position(symbol, side, quantity)
        """
        # 1. Symbol için mevcut pozisyonları filtrele
        positions_for_symbol = self._get_positions_for_symbol(symbol, positions)
        same_side_positions = [p for p in positions_for_symbol if p.get('side') == side]
        opposite_side = 'SHORT' if side == 'LONG' else 'LONG'
        opposite_positions = [p for p in positions_for_symbol if p.get('side') == opposite_side]

        # 2. Max positions per symbol kontrolü
        if len(positions_for_symbol) >= self.max_positions_per_symbol:
            return PositionOpenResult(
                can_open=False,
                reason=f"Max positions reached for {symbol} ({len(positions_for_symbol)}/{self.max_positions_per_symbol})"
            )

        # 3. Pyramiding kontrolü (aynı yönde birden fazla pozisyon)
        can_open_same_side = False
        pyramiding_scale = 1.0

        if self.pyramiding_enabled:
            # Pyramiding aktif: max_entries'e kadar aynı yönde pozisyon açılabilir
            if len(same_side_positions) < self.pyramiding_max_entries:
                can_open_same_side = True
                # Scale factor hesapla: 1st=1.0, 2nd=factor, 3rd=factor^2, ...
                if len(same_side_positions) > 0:
                    pyramiding_scale = self.pyramiding_scale_factor ** len(same_side_positions)
        else:
            # Pyramiding kapalı: sadece 1 pozisyon açılabilir
            if not same_side_positions:
                can_open_same_side = True

        # 4. Hedging modu kontrolü
        if self.allow_hedging:
            # HEDGE MODE: Zıt pozisyonlar da aynı anda açık kalabilir
            if can_open_same_side:
                return PositionOpenResult(
                    can_open=True,
                    reason="OK (hedge mode)",
                    pyramiding_scale=pyramiding_scale,
                    should_close_opposite=False
                )
            else:
                reason = f"Pyramiding limit reached ({len(same_side_positions)}/{self.pyramiding_max_entries})"
                return PositionOpenResult(can_open=False, reason=reason)
        else:
            # ONE-WAY MODE: Zıt pozisyonları önce kapat
            if opposite_positions:
                # Zıt pozisyonlar var - önce kapat, sonra aç
                if can_open_same_side:
                    return PositionOpenResult(
                        can_open=True,
                        reason="OK (close opposite first)",
                        pyramiding_scale=pyramiding_scale,
                        should_close_opposite=True,
                        opposite_positions=opposite_positions
                    )
                else:
                    # Pyramiding limiti doldu
                    reason = f"Pyramiding limit reached ({len(same_side_positions)}/{self.pyramiding_max_entries})"
                    return PositionOpenResult(can_open=False, reason=reason)
            else:
                # Zıt pozisyon yok
                if can_open_same_side:
                    return PositionOpenResult(
                        can_open=True,
                        reason="OK",
                        pyramiding_scale=pyramiding_scale,
                        should_close_opposite=False
                    )
                else:
                    reason = f"Pyramiding limit reached ({len(same_side_positions)}/{self.pyramiding_max_entries})"
                    return PositionOpenResult(can_open=False, reason=reason)

    def check_position_timeout(
        self,
        position: Dict,
        current_timestamp: Any
    ) -> Tuple[bool, str]:
        """
        Pozisyon timeout kontrolü

        Bu method BacktestEngine (line 664-665) ve TradingEngine (line 997-1004)
        içindeki duplicate kodun yerine geçer.

        Args:
            position: Pozisyon dict {'opened_at': datetime/timestamp, ...}
            current_timestamp: Şu anki zaman (datetime, pd.Timestamp, veya ms int)

        Returns:
            (should_close: bool, reason: str)

        Örnek:
            >>> should_close, reason = pm.check_position_timeout(pos, current_time)
            >>> if should_close:
            ...     engine.close_position(pos, reason=reason)
        """
        if not self.position_timeout_enabled:
            return False, ""

        # Pozisyonun açılış zamanını al
        opened_at = position.get('opened_at') or position.get('entry_time') or position.get('open_time')
        if opened_at is None:
            return False, ""

        # Timestamp'leri normalize et
        opened_at_dt = self._normalize_timestamp(opened_at)
        current_dt = self._normalize_timestamp(current_timestamp)

        if opened_at_dt is None or current_dt is None:
            return False, ""

        # Geçen süreyi hesapla (dakika)
        time_elapsed_minutes = (current_dt - opened_at_dt).total_seconds() / 60

        if time_elapsed_minutes >= self.position_timeout:
            reason = f"Position timeout ({self.position_timeout} min)"
            return True, reason

        return False, ""

    def get_positions_for_symbol(
        self,
        symbol: str,
        positions: List[Dict],
        side: Optional[str] = None
    ) -> List[Dict]:
        """
        Symbol (ve opsiyonel side) için pozisyonları getir

        Args:
            symbol: Sembol
            positions: Tüm pozisyonlar
            side: 'LONG' veya 'SHORT' (None = tüm yönler)

        Returns:
            Filtrelenmiş pozisyon listesi
        """
        result = self._get_positions_for_symbol(symbol, positions)

        if side:
            result = [p for p in result if p.get('side') == side]

        return result

    def get_position_count(
        self,
        symbol: str,
        positions: List[Dict],
        side: Optional[str] = None
    ) -> int:
        """
        Symbol için pozisyon sayısını getir

        Args:
            symbol: Sembol
            positions: Tüm pozisyonlar
            side: 'LONG' veya 'SHORT' (None = tüm yönler)

        Returns:
            Pozisyon sayısı
        """
        return len(self.get_positions_for_symbol(symbol, positions, side))

    def can_pyramid(
        self,
        symbol: str,
        side: str,
        positions: List[Dict]
    ) -> Tuple[bool, int, float]:
        """
        Pyramiding yapılabilir mi kontrol et

        Args:
            symbol: Sembol
            side: 'LONG' veya 'SHORT'
            positions: Tüm pozisyonlar

        Returns:
            (can_pyramid: bool, current_entries: int, scale_factor: float)
        """
        if not self.pyramiding_enabled:
            return False, 0, 1.0

        same_side = self.get_positions_for_symbol(symbol, positions, side)
        current_entries = len(same_side)

        if current_entries >= self.pyramiding_max_entries:
            return False, current_entries, 1.0

        # Scale factor: her sonraki entry daha küçük
        scale = self.pyramiding_scale_factor ** current_entries if current_entries > 0 else 1.0

        return True, current_entries, scale

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_positions_for_symbol(self, symbol: str, positions: List[Dict]) -> List[Dict]:
        """Symbol için pozisyonları filtrele"""
        return [p for p in positions if p.get('symbol') == symbol]

    def _normalize_timestamp(self, ts: Any) -> Optional[datetime]:
        """Timestamp'i datetime'a çevir"""
        if ts is None:
            return None

        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()

        if isinstance(ts, (int, float)):
            # Milliseconds timestamp
            try:
                return pd.Timestamp(ts, unit='ms').to_pydatetime()
            except:
                return None

        if isinstance(ts, str):
            try:
                return pd.Timestamp(ts).to_pydatetime()
            except:
                return None

        return None

    # ========================================================================
    # POSITION CREATION & UPDATE
    # ========================================================================

    @staticmethod
    def create_position(
        position_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        order_id: Optional[str] = None,
        entry_time: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Standart pozisyon dict oluştur

        Bu method TradingEngine ve BacktestEngine'deki manuel dict
        construction'ın yerini alır. Tek bir yerden position format'ı
        kontrol edilir.

        Args:
            position_id: Pozisyon ID
            symbol: Trading symbol
            side: 'LONG' veya 'SHORT'
            entry_price: Giriş fiyatı
            quantity: Miktar
            sl_price: Stop loss fiyatı (optional)
            tp_price: Take profit fiyatı (optional)
            order_id: Broker order ID (optional)
            entry_time: Giriş zamanı (optional, default=now)

        Returns:
            Dict: Standart position dict

        Örnek:
            >>> pos = PositionManager.create_position(
            ...     position_id=1,
            ...     symbol='BTCUSDT',
            ...     side='LONG',
            ...     entry_price=95000.0,
            ...     quantity=0.1,
            ...     sl_price=94000.0,
            ...     tp_price=97000.0
            ... )
        """
        from datetime import datetime

        if entry_time is None:
            entry_time = datetime.now()

        return {
            'id': position_id,
            'symbol': symbol,
            'side': side.upper(),
            'entry_time': entry_time,
            'entry_price': entry_price,
            'quantity': quantity,
            'original_quantity': quantity,  # Partial exit tracking için
            'sl_price': sl_price,
            'tp_price': tp_price,
            'stop_loss': sl_price,      # Alias (bazı yerlerde stop_loss kullanılıyor)
            'take_profit': tp_price,    # Alias
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'order_id': order_id,
            'completed_partial_exits': 0,
        }

    @staticmethod
    def update_extreme_prices(
        position: Dict[str, Any],
        current_price: float
    ) -> bool:
        """
        Pozisyon için highest/lowest fiyat güncelle

        Trailing stop ve break-even hesaplamaları için gerekli.

        Args:
            position: Position dict (mutable - yerinde güncellenir)
            current_price: Güncel fiyat

        Returns:
            bool: True = güncelleme yapıldı

        Örnek:
            >>> updated = PositionManager.update_extreme_prices(position, current_price)
            >>> if updated:
            ...     print(f"New high/low: {position['highest_price']}/{position['lowest_price']}")
        """
        updated = False
        side = position.get('side', '').upper()

        if side == 'LONG':
            if current_price > position.get('highest_price', current_price):
                position['highest_price'] = current_price
                updated = True
        elif side == 'SHORT':
            if current_price < position.get('lowest_price', current_price):
                position['lowest_price'] = current_price
                updated = True

        return updated

    # ========================================================================
    # DISPLAY / DEBUG
    # ========================================================================

    def get_config_summary(self) -> Dict:
        """Konfigürasyon özetini getir"""
        return {
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'pyramiding_enabled': self.pyramiding_enabled,
            'pyramiding_max_entries': self.pyramiding_max_entries,
            'pyramiding_scale_factor': self.pyramiding_scale_factor,
            'allow_hedging': self.allow_hedging,
            'position_timeout_enabled': self.position_timeout_enabled,
            'position_timeout': self.position_timeout
        }

    def __repr__(self) -> str:
        return (
            f"PositionManager("
            f"max_per_symbol={self.max_positions_per_symbol}, "
            f"pyramiding={self.pyramiding_enabled}, "
            f"hedging={self.allow_hedging})"
        )


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    """Test Position Manager"""
    from dataclasses import dataclass

    @dataclass
    class MockPositionManagement:
        max_positions_per_symbol: int = 2
        pyramiding_enabled: bool = True
        pyramiding_max_entries: int = 3
        pyramiding_scale_factor: float = 0.5
        allow_hedging: bool = False
        position_timeout_enabled: bool = True
        position_timeout: int = 60  # minutes

    class MockStrategy:
        position_management = MockPositionManagement()

    print("=" * 60)
    print("PositionManager v2.0 - Test")
    print("=" * 60)

    pm = PositionManager(MockStrategy())

    # Test 1: Boş pozisyon listesi
    print("\n1. Empty positions - can open LONG?")
    result = pm.can_open_position('BTCUSDT', 'LONG', [])
    print(f"   can_open: {result.can_open}, reason: {result.reason}, scale: {result.pyramiding_scale}")

    # Test 2: Bir LONG pozisyon var, ikinci LONG açılabilir mi?
    print("\n2. One LONG exists - can open another LONG?")
    positions = [{'symbol': 'BTCUSDT', 'side': 'LONG'}]
    result = pm.can_open_position('BTCUSDT', 'LONG', positions)
    print(f"   can_open: {result.can_open}, reason: {result.reason}, scale: {result.pyramiding_scale:.2f}")

    # Test 3: Bir LONG var, SHORT açılabilir mi? (one-way mode)
    print("\n3. One LONG exists - can open SHORT? (one-way mode)")
    result = pm.can_open_position('BTCUSDT', 'SHORT', positions)
    print(f"   can_open: {result.can_open}, should_close_opposite: {result.should_close_opposite}")
    print(f"   opposite_positions: {len(result.opposite_positions)}")

    # Test 4: Max positions
    print("\n4. Max positions reached?")
    positions = [
        {'symbol': 'BTCUSDT', 'side': 'LONG'},
        {'symbol': 'BTCUSDT', 'side': 'LONG'}
    ]
    result = pm.can_open_position('BTCUSDT', 'LONG', positions)
    print(f"   can_open: {result.can_open}, reason: {result.reason}")

    # Test 5: Pyramiding limit
    print("\n5. Pyramiding limit (max_entries=3)?")
    positions = [
        {'symbol': 'BTCUSDT', 'side': 'LONG'},
        {'symbol': 'BTCUSDT', 'side': 'LONG'},
        {'symbol': 'BTCUSDT', 'side': 'LONG'}
    ]
    # Temporarily set max_positions higher to test pyramiding limit
    pm.max_positions_per_symbol = 5
    result = pm.can_open_position('BTCUSDT', 'LONG', positions)
    print(f"   can_open: {result.can_open}, reason: {result.reason}")

    # Test 6: Timeout check
    print("\n6. Position timeout check")
    import time
    pos = {'opened_at': datetime.now()}
    should_close, reason = pm.check_position_timeout(pos, datetime.now())
    print(f"   should_close: {should_close} (just opened)")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
