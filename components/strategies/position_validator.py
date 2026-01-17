#!/usr/bin/env python3
"""
Position Validator - Pozisyon Kurallarını Kontrol Eder

Strateji parametrelerine göre pozisyon açılıp açılamayacağını validate eder.
Bu dosya strategies/ klasöründe çünkü STRATEJI KURALLARI ile ilgilidir.

Sorumluluklar:
- Pyramiding kurallarını kontrol et
- Hedging kurallarını kontrol et
- Pozisyon limitlerini kontrol et
- Pozisyon timeout kurallarını kontrol et

NOT: Execution (gerçek pozisyon açma/kapama) managers/position_manager.py'de yapılır.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """
    Pozisyon validasyon sonucu
    
    Attributes:
        accepted: Pozisyon kabul edildi mi?
        reason: Red/kabul sebebi (Türkçe açıklama)
        action: Yapılacak aksiyon ('open_new', 'pyramid', 'reject', 'close_opposite')
    """
    accepted: bool
    reason: str
    action: Optional[str] = None


class PositionValidator:
    """
    Pozisyon Validasyon Mantığı (Strateji Kuralları)
    
    Sorumluluklar:
    - Sinyalin pozisyon açıp açamayacağını kontrol et
    - Pyramiding mantığını yönet
    - Hedging mantığını yönet
    - Pozisyon limitlerini zorla
    
    Sorumlu OLMADIKLARI:
    - Gerçek pozisyon açma (PositionManager)
    - Risk hesaplama (RiskManager)
    - Emir çalıştırma (OrderManager)
    
    Mimari:
    - Strateji kuralları burada (strategies/)
    - Execution logic manager'da (managers/)
    - DRY: Canlı trading ve backtest aynı kodu kullanır
    - SOLID: Single Responsibility Principle
    - Testable: Unit test yazılabilir
    """
    
    def __init__(self, position_management_config: Any, logger: Optional[Any] = None):
        """
        Validator'ı stratejinin position_management config'i ile başlat
        
        Args:
            position_management_config: Stratejinin PositionManagement objesi
            logger: Opsiyonel logger
        """
        self.config = position_management_config
        self.logger = logger
        
        # Ayarları çıkar
        self.max_positions_per_symbol = getattr(position_management_config, 'max_positions_per_symbol', 1)
        self.max_total_positions = getattr(position_management_config, 'max_total_positions', 2)
        self.allow_hedging = getattr(position_management_config, 'allow_hedging', False)
        
        # Pyramiding ayarları
        self.pyramiding_enabled = getattr(position_management_config, 'pyramiding_enabled', False)
        self.pyramiding_max_entries = getattr(position_management_config, 'pyramiding_max_entries', 1)
        self.pyramiding_scale_factor = getattr(position_management_config, 'pyramiding_scale_factor', 1.0)
        
        # Timeout ayarları
        self.timeout_enabled = getattr(position_management_config, 'position_timeout_enabled', False)
        self.timeout_minutes = getattr(position_management_config, 'position_timeout', 1800)
        
        if self.logger:
            self.logger.debug(f"PositionValidator başlatıldı:")
            self.logger.debug(f"  - Max pozisyon/sembol: {self.max_positions_per_symbol}")
            self.logger.debug(f"  - Max toplam pozisyon: {self.max_total_positions}")
            self.logger.debug(f"  - Hedging izni: {self.allow_hedging}")
            self.logger.debug(f"  - Pyramiding: {self.pyramiding_enabled} (max: {self.pyramiding_max_entries})")
            self.logger.debug(f"  - Timeout: {self.timeout_enabled} ({self.timeout_minutes} dakika)")
    
    def can_open_position(
        self,
        signal_side: str,
        open_positions: Dict[str, Any],
        total_positions_count: Optional[int] = None
    ) -> ValidationResult:
        """
        Sinyalin pozisyon açıp açamayacağını kontrol et
        
        Args:
            signal_side: 'long' veya 'short'
            open_positions: Açık pozisyonlar dict'i {side -> Position veya List[Position]}
            total_positions_count: Tüm sembollerdeki toplam açık pozisyon (opsiyonel)
        
        Returns:
            ValidationResult (kabul/red kararı)
        """
        # Adım 1: Toplam pozisyon limitini kontrol et
        if total_positions_count is None:
            total_positions_count = self._count_total_positions(open_positions)
        
        if total_positions_count >= self.max_total_positions:
            return ValidationResult(
                accepted=False,
                reason=f"Maksimum toplam pozisyon limitine ulaşıldı ({total_positions_count}/{self.max_total_positions})"
            )
        
        # Adım 2: Bu tarafta pozisyon var mı kontrol et
        same_side_positions = self._get_positions_by_side(open_positions, signal_side)
        opposite_side = 'short' if signal_side == 'long' else 'long'
        opposite_side_positions = self._get_positions_by_side(open_positions, opposite_side)
        
        # Adım 2.5: max_positions_per_symbol kontrolü (multi-symbol desteği için)
        # Bu sembol için tüm pozisyonları say (her iki taraf)
        symbol_position_count = 0
        if same_side_positions:
            symbol_position_count += len(same_side_positions) if isinstance(same_side_positions, list) else 1
        if opposite_side_positions:
            symbol_position_count += len(opposite_side_positions) if isinstance(opposite_side_positions, list) else 1
        
        if symbol_position_count >= self.max_positions_per_symbol:
            return ValidationResult(
                accepted=False,
                reason=f"Sembol başına maksimum pozisyon limitine ulaşıldı ({symbol_position_count}/{self.max_positions_per_symbol})"
            )
        
        # Adım 3: PYRAMIDING mantığı
        if same_side_positions:
            if self.pyramiding_enabled:
                # Başka bir entry eklenebilir mi?
                num_entries = len(same_side_positions) if isinstance(same_side_positions, list) else 1
                
                if num_entries < self.pyramiding_max_entries:
                    return ValidationResult(
                        accepted=True,
                        reason=f"Pyramiding: Entry {num_entries + 1}/{self.pyramiding_max_entries} ekleniyor",
                        action='pyramid'
                    )
                else:
                    return ValidationResult(
                        accepted=False,
                        reason=f"Pyramiding limitine ulaşıldı ({num_entries}/{self.pyramiding_max_entries})"
                    )
            else:
                # Pyramiding kapalı - aynı tarafta açılamaz
                return ValidationResult(
                    accepted=False,
                    reason=f"Zaten {signal_side} pozisyonu var (pyramiding devre dışı)"
                )
        
        # Adım 4: HEDGING mantığı
        if opposite_side_positions:
            if self.allow_hedging:
                # Ters taraf açılabilir (hedging)
                return ValidationResult(
                    accepted=True,
                    reason=f"Hedging: {opposite_side} pozisyonu varken {signal_side} açılıyor",
                    action='open_new'
                )
            else:
                # Hedging kapalı - ters taraf açılamaz
                return ValidationResult(
                    accepted=False,
                    reason=f"Zaten {opposite_side} pozisyonu var (hedging devre dışı)"
                )
        
        # Adım 5: Normal giriş (çakışma yok)
        return ValidationResult(
            accepted=True,
            reason=f"Yeni {signal_side} pozisyonu açılıyor",
            action='open_new'
        )
    
    def _count_total_positions(self, open_positions: Dict[str, Any]) -> int:
        """
        Toplam açık pozisyon sayısını hesapla
        
        Args:
            open_positions: Açık pozisyonlar dict'i
        
        Returns:
            Toplam sayı
        """
        count = 0
        for side, pos in open_positions.items():
            if isinstance(pos, list):
                count += len(pos)
            elif pos is not None:
                count += 1
        return count
    
    def _get_positions_by_side(self, open_positions: Dict[str, Any], side: str) -> Any:
        """
        Belirli bir taraf için pozisyonları getir
        
        Args:
            open_positions: Açık pozisyonlar dict'i
            side: 'long' veya 'short'
        
        Returns:
            Position, List[Position], veya None
        """
        return open_positions.get(side)
    
    def calculate_pyramid_size(self, base_size: float, entry_number: int) -> float:
        """
        Pyramiding entry için pozisyon boyutunu hesapla
        
        Args:
            base_size: Baz pozisyon boyutu
            entry_number: Entry numarası (1, 2, 3, ...)
        
        Returns:
            Ölçeklendirilmiş pozisyon boyutu
        
        Örnek:
            base_size = 1.0, scale_factor = 0.5
            Entry 1: 1.0 (100%)
            Entry 2: 0.5 (50%)
            Entry 3: 0.25 (25%)
        """
        if entry_number == 1:
            return base_size
        
        # Her entry için pyramiding_scale_factor ile ölçeklendir
        scaled_size = base_size * (self.pyramiding_scale_factor ** (entry_number - 1))
        return scaled_size
    
    def should_close_opposite_for_entry(
        self,
        signal_side: str,
        open_positions: Dict[str, Any]
    ) -> bool:
        """
        Giriş öncesi ters pozisyon kapatılmalı mı kontrol et
        (Hedging olmayan stratejiler için)
        
        Args:
            signal_side: 'long' veya 'short'
            open_positions: Açık pozisyonlar dict'i
        
        Returns:
            True ise ters pozisyon kapatılmalı
        """
        if self.allow_hedging:
            return False  # Hedging izinli, kapatmaya gerek yok
        
        opposite_side = 'short' if signal_side == 'long' else 'long'
        opposite_pos = self._get_positions_by_side(open_positions, opposite_side)
        
        return opposite_pos is not None

