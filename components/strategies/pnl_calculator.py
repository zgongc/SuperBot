#!/usr/bin/env python3
"""
components/strategies/pnl_calculator.py
SuperBot - PnL Calculator
Author: SuperBot Team
Date: 2025-12-07
Version: 1.0.0

PnL (Profit and Loss) calculation utility class

Purpose: To remove duplicate PnL calculation code in TradingEngine and BacktestEngine.
move to a central location. DRY principle.

Usage:
    from components.strategies.pnl_calculator import PnLCalculator

    # Single calculation
    gross, net, pct = PnLCalculator.calculate(
        entry_price=95000,
        exit_price=96000,
        quantity=0.1,
        side='LONG',
        fee=10.0
    )

    # Detailed result
    result = PnLCalculator.calculate_detailed(
        entry_price=95000,
        exit_price=96000,
        quantity=0.1,
        side='LONG',
        entry_fee=5.0,
        exit_fee=5.0
    )
"""

from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PnLResult:
    """P&L calculation result"""
    gross_pnl: float       # PnL before fees
    net_pnl: float         # PnL after fees
    net_pnl_pct: float     # Net PnL percentage
    position_value: float  # Entry value (entry_price * quantity)
    total_fee: float       # Total fee
    is_profitable: bool    # Net PnL >= 0


class PnLCalculator:
    """
    PnL (Profit and Loss) Calculator

    Static utility class - no need to create an instance.

    LONG pozisyon:  PnL = (exit_price - entry_price) * quantity
    SHORT pozisyon: PnL = (entry_price - exit_price) * quantity
    """

    @staticmethod
    def calculate(
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        fee: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        PnL hesapla (basit)

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Quantity
            side: 'LONG' or 'SHORT'
            fee: Total fee (entry + exit)

        Returns:
            Tuple: (gross_pnl, net_pnl, net_pnl_pct)

        Example:
            >>> gross, net, pct = PnLCalculator.calculate(95000, 96000, 0.1, 'LONG', 10)
            >>> print(f"PnL: ${net:.2f} ({pct:.2f}%)")
            PnL: $90.00 (0.95%)
        """
        # Gross PnL
        if side.upper() == 'LONG':
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Net PnL
        net_pnl = gross_pnl - fee

        # Position value & percentage
        position_value = entry_price * quantity
        net_pnl_pct = (net_pnl / position_value) * 100 if position_value > 0 else 0.0

        return gross_pnl, net_pnl, net_pnl_pct

    @staticmethod
    def calculate_detailed(
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,
        entry_fee: float = 0.0,
        exit_fee: float = 0.0
    ) -> PnLResult:
        """
        Calculate PnL (detailed)

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Quantity
            side: 'LONG' or 'SHORT'
            entry_fee: Entry fee
            exit_fee: Exit fee

        Returns:
            PnLResult: Detailed result dataclass
        """
        total_fee = entry_fee + exit_fee
        gross_pnl, net_pnl, net_pnl_pct = PnLCalculator.calculate(
            entry_price, exit_price, quantity, side, total_fee
        )

        position_value = entry_price * quantity

        return PnLResult(
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            position_value=position_value,
            total_fee=total_fee,
            is_profitable=net_pnl >= 0
        )

    @staticmethod
    def calculate_partial(
        entry_price: float,
        exit_price: float,
        close_quantity: float,
        side: str,
        fee: float = 0.0
    ) -> Tuple[float, float]:
        """
        Partial close PnL hesapla

        Args:
            entry_price: Original entry price
            exit_price: Partial exit price
            close_quantity: Quantity closed
            side: 'LONG' or 'SHORT'
            fee: Fee for this partial close

        Returns:
            Tuple: (gross_pnl, net_pnl)
        """
        if side.upper() == 'LONG':
            gross_pnl = (exit_price - entry_price) * close_quantity
        else:
            gross_pnl = (entry_price - exit_price) * close_quantity

        net_pnl = gross_pnl - fee

        return gross_pnl, net_pnl


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PnLCalculator Test")
    print("=" * 60)

    # Test 1: LONG profitable
    print("\n1. LONG profitable:")
    gross, net, pct = PnLCalculator.calculate(
        entry_price=95000,
        exit_price=96000,
        quantity=0.1,
        side='LONG',
        fee=10.0
    )
    print(f"   Entry: $95,000 -> Exit: $96,000 (0.1 BTC)")
    print(f"   Gross: ${gross:,.2f}, Net: ${net:,.2f} ({pct:+.2f}%)")

    # Test 2: LONG loss
    print("\n2. LONG loss:")
    gross, net, pct = PnLCalculator.calculate(
        entry_price=95000,
        exit_price=94000,
        quantity=0.1,
        side='LONG',
        fee=10.0
    )
    print(f"   Entry: $95,000 -> Exit: $94,000 (0.1 BTC)")
    print(f"   Gross: ${gross:,.2f}, Net: ${net:,.2f} ({pct:+.2f}%)")

    # Test 3: SHORT profitable
    print("\n3. SHORT profitable:")
    gross, net, pct = PnLCalculator.calculate(
        entry_price=95000,
        exit_price=94000,
        quantity=0.1,
        side='SHORT',
        fee=10.0
    )
    print(f"   Entry: $95,000 -> Exit: $94,000 (0.1 BTC)")
    print(f"   Gross: ${gross:,.2f}, Net: ${net:,.2f} ({pct:+.2f}%)")

    # Test 4: Detailed result
    print("\n4. Detailed result:")
    result = PnLCalculator.calculate_detailed(
        entry_price=95000,
        exit_price=96000,
        quantity=0.1,
        side='LONG',
        entry_fee=5.0,
        exit_fee=5.0
    )
    print(f"   Position Value: ${result.position_value:,.2f}")
    print(f"   Gross PnL: ${result.gross_pnl:,.2f}")
    print(f"   Net PnL: ${result.net_pnl:,.2f} ({result.net_pnl_pct:+.2f}%)")
    print(f"   Total Fee: ${result.total_fee:.2f}")
    print(f"   Profitable: {result.is_profitable}")

    # Test 5: Partial close
    print("\n5. Partial close (50%):")
    gross, net = PnLCalculator.calculate_partial(
        entry_price=95000,
        exit_price=96000,
        close_quantity=0.05,  # 50% of 0.1
        side='LONG',
        fee=5.0
    )
    print(f"   Closing 0.05 BTC @ $96,000")
    print(f"   Gross: ${gross:,.2f}, Net: ${net:,.2f}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
