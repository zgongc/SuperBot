#!/usr/bin/env python3
"""
components/strategies/portfolio_coordinator.py
SuperBot - Portfolio Coordinator

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Multi-symbol koordinasyonu:
    - Symbol başına pozisyon limitleri
    - Toplam pozisyon limitleri
    - Available slot kontrolü
    - Korelasyon kontrolü (optional)
    - Portfolio-level risk yönetimi

Kullanım:
    from components.strategies.portfolio_coordinator import PortfolioCoordinator

    coordinator = PortfolioCoordinator(strategy, position_manager)
    can_open = coordinator.can_open_position('BTCUSDT')
    coordinator.register_position('BTCUSDT', 'LONG')
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict

from components.strategies.base_strategy import BaseStrategy


class PortfolioCoordinator:
    """
    Portfolio yönetimi coordinator'ı

    Multi-symbol pozisyon koordinasyonu ve limitleri yönetir
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        position_manager: Optional[Any] = None,
        logger: Any = None
    ):
        """
        Initialize PortfolioCoordinator

        Args:
            strategy: BaseStrategy instance
            position_manager: PositionManager instance (optional)
            logger: Logger instance (optional)
        """
        self.strategy = strategy
        self.position_manager = position_manager
        self.logger = logger
        
        # Position limits from strategy
        self.max_positions_per_symbol = strategy.position_management.max_positions_per_symbol
        self.max_total_positions = strategy.position_management.max_total_positions
        
        # Track active positions per symbol
        # Format: {symbol: [{'id': ..., 'side': 'LONG/SHORT', ...}, ...]}
        self.active_positions: Dict[str, List[Dict]] = defaultdict(list)
        
        # Correlation configuration (if exists)
        self.correlation_enabled = self._parse_correlation_config()
    
    # ========================================================================
    # POSITION TRACKING
    # ========================================================================
    
    def register_position(
        self,
        symbol: str,
        position_data: Dict[str, Any]
    ) -> None:
        """
        Yeni pozisyon kaydet
        
        Args:
            symbol: Trading symbol
            position_data: Position bilgisi
                {
                    'id': str,
                    'side': 'LONG' | 'SHORT',
                    'entry_price': float,
                    'quantity': float,
                    ...
                }
        """
        self.active_positions[symbol].append(position_data)
        
        if self.logger:
            self.logger.debug(
                f"Portfolio: Registered {symbol} {position_data.get('side')} position. "
                f"Total: {self.get_total_position_count()}"
            )
    
    def unregister_position(
        self,
        symbol: str,
        position_id: str
    ) -> None:
        """
        Pozisyonu kaldır
        
        Args:
            symbol: Trading symbol
            position_id: Position ID
        """
        if symbol in self.active_positions:
            self.active_positions[symbol] = [
                p for p in self.active_positions[symbol]
                if p.get('id') != position_id
            ]
            
            # Remove empty symbol entries
            if not self.active_positions[symbol]:
                del self.active_positions[symbol]
            
            if self.logger:
                self.logger.debug(
                    f"Portfolio: Unregistered {symbol} position {position_id}. "
                    f"Total: {self.get_total_position_count()}"
                )
    
    def update_positions_from_position_manager(self) -> None:
        """
        PositionManager'dan pozisyonları sync et
        
        Note:
            Bu method position_manager varsa çağrılmalı
        """
        if self.position_manager is None:
            return
        
        # Clear current tracking
        self.active_positions.clear()
        
        # Get active positions from position_manager
        if hasattr(self.position_manager, 'get_all_active_positions'):
            active_positions = self.position_manager.get_all_active_positions()
            
            for position in active_positions:
                symbol = position.get('symbol')
                if symbol:
                    self.register_position(symbol, position)
    
    # ========================================================================
    # POSITION LIMITS
    # ========================================================================
    
    def can_open_position(
        self,
        symbol: str,
        side: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Yeni pozisyon açılabilir mi?
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT' (optional)
        
        Returns:
            (can_open: bool, reason: Optional[str])
                - (True, None): OK
                - (False, reason): Cannot open
        """
        # Check total position limit
        total_count = self.get_total_position_count()
        if total_count >= self.max_total_positions:
            return False, f"Max total positions reached ({self.max_total_positions})"
        
        # Check per-symbol limit
        symbol_count = self.get_symbol_position_count(symbol)
        if symbol_count >= self.max_positions_per_symbol:
            return False, f"Max positions for {symbol} reached ({self.max_positions_per_symbol})"
        
        # Check correlation (if enabled)
        if self.correlation_enabled:
            correlated = self._check_correlation(symbol)
            if correlated:
                return False, f"Symbol {symbol} correlated with existing positions"
        
        return True, None
    
    def get_available_slots(self) -> int:
        """
        Kaç slot daha açık?
        
        Returns:
            int: Kalan pozisyon slot sayısı
        """
        total = self.get_total_position_count()
        return max(0, self.max_total_positions - total)
    
    def get_symbol_available_slots(self, symbol: str) -> int:
        """
        Symbol için kaç slot daha açık?
        
        Args:
            symbol: Trading symbol
        
        Returns:
            int: Symbol için kalan slot sayısı
        """
        count = self.get_symbol_position_count(symbol)
        return max(0, self.max_positions_per_symbol - count)
    
    # ========================================================================
    # POSITION QUERIES
    # ========================================================================
    
    def get_total_position_count(self) -> int:
        """Toplam aktif pozisyon sayısı"""
        return sum(len(positions) for positions in self.active_positions.values())
    
    def get_symbol_position_count(self, symbol: str) -> int:
        """Symbol için aktif pozisyon sayısı"""
        return len(self.active_positions.get(symbol, []))
    
    def get_active_symbols(self) -> List[str]:
        """Aktif pozisyonu olan symbol'ler"""
        return list(self.active_positions.keys())
    
    def has_position(self, symbol: str, side: Optional[str] = None) -> bool:
        """
        Symbol için pozisyon var mı?
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT' (None = any side)
        
        Returns:
            bool: True if position exists
        """
        if symbol not in self.active_positions:
            return False
        
        if side is None:
            return len(self.active_positions[symbol]) > 0
        
        # Check specific side
        for pos in self.active_positions[symbol]:
            if pos.get('side', '').upper() == side.upper():
                return True
        
        return False
    
    def get_symbol_positions(self, symbol: str) -> List[Dict]:
        """
        Symbol için tüm pozisyonları dön
        
        Args:
            symbol: Trading symbol
        
        Returns:
            List[Dict]: Position data list
        """
        return self.active_positions.get(symbol, [])
    
    def get_all_positions(self) -> Dict[str, List[Dict]]:
        """Tüm pozisyonları dön"""
        return dict(self.active_positions)
    
    # ========================================================================
    # CORRELATION CONTROL
    # ========================================================================
    
    def _check_correlation(self, symbol: str) -> bool:
        """
        Symbol mevcut pozisyonlarla korele mi?
        
        Args:
            symbol: Kontrol edilecek symbol
        
        Returns:
            bool: True if correlated (should block)
        
        Note:
            Bu basitleştirilmiş bir implementasyon.
            Production'da gerçek korelasyon analizi yapılmalı.
        """
        if not self.correlation_enabled:
            return False
        
        active_symbols = self.get_active_symbols()
        
        if not active_symbols:
            return False
        
        # Basit heuristic: aynı base currency
        symbol_base = symbol.replace('USDT', '').replace('USD', '').replace('BUSD', '')
        
        for active_symbol in active_symbols:
            active_base = active_symbol.replace('USDT', '').replace('USD', '').replace('BUSD', '')
            
            if symbol_base == active_base:
                # Aynı coin, farklı quote
                return True
        
        # TODO: Gerçek korelasyon hesaplama (price correlation)
        
        return False
    
    def _parse_correlation_config(self) -> bool:
        """Parse correlation config from strategy"""
        if hasattr(self.strategy, 'custom_parameters'):
            custom_params = self.strategy.custom_parameters
            if custom_params and 'correlation_control' in custom_params:
                return custom_params['correlation_control'].get('enabled', False)
        
        return False
    
    # ========================================================================
    # PORTFOLIO RISK
    # ========================================================================
    
    def get_portfolio_risk_percent(self) -> float:
        """
        Portfolio risk yüzdesini hesapla
        
        Returns:
            float: Total risk percent (sum of all position risks)
        
        Note:
            Bu basitleştirilmiş bir implementasyon.
            Production'da gerçek PnL/risk hesabı yapılmalı.
        """
        if self.position_manager is None:
            # Approximate risk
            position_count = self.get_total_position_count()
            avg_risk_per_trade = self.strategy.risk_management.max_risk_per_trade
            return position_count * avg_risk_per_trade
        
        # TODO: Get actual risk from position_manager
        return 0.0
    
    def is_portfolio_risk_exceeded(self) -> bool:
        """
        Portfolio risk limiti aşıldı mı?
        
        Returns:
            bool: True if exceeded
        """
        current_risk = self.get_portfolio_risk_percent()
        max_risk = self.strategy.max_portfolio_risk  # Auto-calculated from leverage

        return current_risk >= max_risk
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def clear(self) -> None:
        """Tüm pozisyon tracking'i temizle"""
        self.active_positions.clear()
        
        if self.logger:
            self.logger.debug("Portfolio: Cleared all position tracking")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Portfolio özeti
        
        Returns:
            Dict: Portfolio summary
        """
        return {
            'total_positions': self.get_total_position_count(),
            'max_total_positions': self.max_total_positions,
            'available_slots': self.get_available_slots(),
            'active_symbols': self.get_active_symbols(),
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'portfolio_risk_percent': self.get_portfolio_risk_percent(),
            'max_portfolio_risk': self.strategy.max_portfolio_risk,  # Auto-calculated
        }
    
    def __repr__(self) -> str:
        return (
            f"<PortfolioCoordinator "
            f"positions={self.get_total_position_count()}/{self.max_total_positions} "
            f"symbols={len(self.get_active_symbols())}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PortfolioCoordinator',
]

