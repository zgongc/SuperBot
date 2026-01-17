#!/usr/bin/env python3
"""
components/strategies/pnl_calculator.py
SuperBot - PnL Calculator
Yazar: SuperBot Team
Tarih: 2025-12-07
Versiyon: 1.0.0

PnL (Profit and Loss) hesaplama utility class

Amaç: TradingEngine ve BacktestEngine'deki duplicate PnL hesaplama kodunu
merkezi bir yere taşımak. DRY prensibi.

Kullanım:
    from components.strategies.pnl_calculator import PnLCalculator

    # Tek hesaplama
    gross, net, pct = PnLCalculator.calculate(
        entry_price=95000,
        exit_price=96000,
        quantity=0.1,
        side='LONG',
        fee=10.0
    )

    # Detaylı sonuç
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
    """PnL hesaplama sonucu"""
    gross_pnl: float       # Fee öncesi PnL
    net_pnl: float         # Fee sonrası PnL
    net_pnl_pct: float     # Net PnL yüzdesi
    position_value: float  # Entry değeri (entry_price * quantity)
    total_fee: float       # Toplam fee
    is_profitable: bool    # Net PnL >= 0


class PnLCalculator:
    """
    PnL (Profit and Loss) Calculator

    Static utility class - instance oluşturmaya gerek yok.

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
            entry_price: Giriş fiyatı
            exit_price: Çıkış fiyatı
            quantity: Miktar
            side: 'LONG' veya 'SHORT'
            fee: Toplam fee (entry + exit)

        Returns:
            Tuple: (gross_pnl, net_pnl, net_pnl_pct)

        Örnek:
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
        PnL hesapla (detaylı)

        Args:
            entry_price: Giriş fiyatı
            exit_price: Çıkış fiyatı
            quantity: Miktar
            side: 'LONG' veya 'SHORT'
            entry_fee: Giriş fee
            exit_fee: Çıkış fee

        Returns:
            PnLResult: Detaylı sonuç dataclass
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
            entry_price: Orijinal giriş fiyatı
            exit_price: Kısmi çıkış fiyatı
            close_quantity: Kapatılan miktar
            side: 'LONG' veya 'SHORT'
            fee: Bu partial close için fee

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
