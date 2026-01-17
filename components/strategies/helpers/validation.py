#!/usr/bin/env python3
"""
Strategy Validation - Wrapper Module (Geriye Uyumluluk İçin)

Bu modül artık sadece bir WRAPPER'dır.
Gerçek validation logic'i components/strategies/strategy_validator.py'de bulunur.

Bu dosya sadece geriye uyumluluk için tutulmaktadır.

Yeni kodlarda doğrudan strategy_validator.py kullanılmalıdır:
    from components.strategies.strategy_validator import validate_strategy
"""

from typing import Any, Optional

# Import edilebilir tüm öğeler
from components.strategies.strategy_validator import (
    StrategyValidator,
    ValidationError,
    StrategyValidationResult,
    ValidationException,
    validate_strategy,
    validate_strategy_strict,
)


# ==================== LEGACY WRAPPER FUNCTIONS ====================
# (Eski kodlar için geriye uyumluluk - yeni validation'ı çağırır)


def validate_condition(condition: Any, context: str = "") -> None:
    """
    Tek bir koşulu validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    Geriye uyumluluk için tutulmuştur.
    
    Args:
        condition: Koşul (list/tuple)
        context: Hata mesajı context'i
    
    Raises:
        ValidationException: Geçersiz koşul varsa
    """
    # Bu fonksiyon artık deprecated, ama geriye uyumluluk için
    # minimal validation yapalım
    if not isinstance(condition, (list, tuple)):
        raise ValidationException(f"{context}: Koşul list veya tuple olmalı")
    
    if len(condition) < 3:
        raise ValidationException(f"{context}: Koşul en az 3 eleman içermeli")


def validate_entry_conditions(entry_conditions: Any) -> None:
    """
    Entry conditions'ı validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    
    Args:
        entry_conditions: Entry conditions dict
    
    Raises:
        ValidationException: Geçersiz koşul varsa
    """
    if not isinstance(entry_conditions, dict):
        raise ValidationException("entry_conditions dict olmalı")
    
    if 'long' not in entry_conditions and 'short' not in entry_conditions:
        raise ValidationException("En az 'long' veya 'short' koşulu gerekli")
    
    for side, conditions in entry_conditions.items():
        if not isinstance(conditions, list):
            raise ValidationException(f"{side} koşulları list olmalı")
        
        for idx, condition in enumerate(conditions):
            validate_condition(condition, f"entry_conditions.{side}[{idx}]")


def validate_exit_conditions(exit_conditions: Any) -> None:
    """
    Exit conditions'ı validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    
    Args:
        exit_conditions: Exit conditions dict
    
    Raises:
        ValidationException: Geçersiz koşul varsa
    """
    if not isinstance(exit_conditions, dict):
        raise ValidationException("exit_conditions dict olmalı")
    
    for side, conditions in exit_conditions.items():
        if not isinstance(conditions, list):
            raise ValidationException(f"{side} koşulları list olmalı")
        
        for idx, condition in enumerate(conditions):
            validate_condition(condition, f"exit_conditions.{side}[{idx}]")


def validate_risk_management(risk_management: Any) -> None:
    """
    Risk management config'ini validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    
    Args:
        risk_management: RiskManagement dataclass
    
    Raises:
        ValidationException: Geçersiz config varsa
    """
    if not hasattr(risk_management, 'size_value'):
        raise ValidationException("Risk management'da size_value gerekli")
    
    if risk_management.size_value <= 0:
        raise ValidationException("size_value pozitif olmalı")


def validate_exit_strategy(exit_strategy: Any) -> None:
    """
    Exit strategy config'ini validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    
    Args:
        exit_strategy: ExitStrategy dataclass
    
    Raises:
        ValidationException: Geçersiz config varsa
    """
    if hasattr(exit_strategy, 'take_profit_value'):
        if exit_strategy.take_profit_value <= 0:
            raise ValidationException("take_profit_value pozitif olmalı")
    
    if hasattr(exit_strategy, 'stop_loss_value'):
        if exit_strategy.stop_loss_value <= 0:
            raise ValidationException("stop_loss_value pozitif olmalı")


def validate_position_management(position_management: Any) -> None:
    """
    Position management config'ini validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    
    Args:
        position_management: PositionManagement dataclass
    
    Raises:
        ValidationException: Geçersiz config varsa
    """
    if hasattr(position_management, 'max_total_positions'):
        if position_management.max_total_positions <= 0:
            raise ValidationException("max_total_positions pozitif olmalı")


def validate_complete_strategy(strategy: Any) -> None:
    """
    Tüm stratejiyi validate et (LEGACY)
    
    Artık bu fonksiyon StrategyValidator içindedir.
    Bu wrapper yeni validator'ı çağırır.
    
    Args:
        strategy: Strateji objesi
    
    Raises:
        ValidationException: Geçersiz config varsa
    """
    # Yeni validator'ı kullan
    validate_strategy_strict(strategy)


# ==================== EXPORT ====================

__all__ = [
    # Ana validator (yeni)
    'StrategyValidator',
    'ValidationError',
    'StrategyValidationResult',
    'ValidationException',
    'validate_strategy',
    'validate_strategy_strict',
    
    # Legacy functions (geriye uyumluluk)
    'validate_condition',
    'validate_entry_conditions',
    'validate_exit_conditions',
    'validate_risk_management',
    'validate_exit_strategy',
    'validate_position_management',
    'validate_complete_strategy',
]
