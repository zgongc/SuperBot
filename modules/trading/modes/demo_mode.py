#!/usr/bin/env python3
"""
modules/trading/modes/demo_mode.py
SuperBot - Demo Trading Mode (Testnet)
Author: SuperBot Team
Date: 2025-11-28
Versiyon: 2.0.0

Demo Trading Mode - Testnet Execution

DEMO MODE:
- Testnet data (WebSocket + REST)
- Testnet balance
- REAL order execution (Binance Futures Testnet)

PaperMode'dan FARKI:
- is_real_execution = True (order actually goes to the testnet)
- execute_order() sends to the API

Usage:
    mode = DemoMode(config, logger)
    await mode.initialize()
    result = await mode.execute_order(order)  # Testnet'e gider!

Dependencies:
    - python>=3.12
    - PaperMode (extends)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path for direct execution
import sys
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from modules.trading.modes.paper_mode import PaperMode
from modules.trading.modes.base_mode import (
    ModeType,
    Order,
    OrderResult,
    OrderStatus,
    OrderSide
)


class DemoMode(PaperMode):
    """
    Demo Trading Mode - Testnet Execution
    
    Extends PaperMode, only execute_order() is different.
    Orders are sent to the Binance Futures Testnet.
    """
    
    @property
    def mode_type(self) -> ModeType:
        return ModeType.DEMO
    
    @property
    def is_real_execution(self) -> bool:
        return True  # Order really goes to the testnet
    
    async def initialize(self) -> None:
        """Initialize demo mode"""
        self.log("🚀 DemoMode is starting (Testnet)...")
        
        # Parent initialization (creates the connector)
        await super().initialize()
        
        # Testnet balance kontrol
        if self._connector:
            try:
                balance = await self._connector.get_account_balance()
                if balance:
                    self._balance = float(balance.get("totalWalletBalance", 0))
                    self._available_balance = float(balance.get("availableBalance", 0))
                    self.log(f"   💰 Testnet Balance: ${self._balance:,.2f}")
            except Exception as e:
                self.log(f"   ⚠️ Could not retrieve testnet balance: {e}", "warning")
        
        self.log("✅ DemoMode ready (Testnet)")
    
    async def execute_order(self, order: Order) -> OrderResult:
        """
        Testnet order execution
        
        The order is sent to the Binance Futures Testnet.
        
        Args:
            order: Order request
            
        Returns:
            OrderResult: Execution result
        """
        if not self._connector:
            self.log("❌ No connector found, falling back to paper mode", "error")
            return await super().execute_order(order)
        
        try:
            # Order parameters
            side = order.side.value  # "BUY" or "SELL"
            order_type = order.order_type.value if hasattr(order, 'order_type') else "MARKET"
            
            self.log(
                f"📤 {order.symbol}: Sending {side} order (Testnet)..."
            )
            
            # Send to the API
            result = await self._connector.place_order(
                symbol=order.symbol,
                side=side,
                order_type=order_type,
                quantity=order.quantity,
                price=order.price if order_type == "LIMIT" else None
            )
            
            if result and result.get("orderId"):
                # Successful
                fill_price = float(result.get("avgPrice", 0)) or float(result.get("price", 0))
                filled_qty = float(result.get("executedQty", 0))
                
                order_result = OrderResult(
                    order_id=str(result.get("orderId")),
                    symbol=order.symbol,
                    side=order.side,
                    status=OrderStatus.FILLED if filled_qty > 0 else OrderStatus.PENDING,
                    quantity=order.quantity,
                    filled_quantity=filled_qty,
                    price=fill_price,
                    fee=0.0,  # Calculated from the API
                    timestamp=datetime.now(),
                    raw=result
                )
                
                self.log(
                    f"✅ {order.symbol}: {side} @ ${fill_price:,.2f} "
                    f"(qty: {filled_qty:.6f}) [Testnet]"
                )
                
                return order_result
            else:
                # API error
                error_msg = result.get("msg", "Unknown error") if result else "No response"
                self.log(f"❌ Order error: {error_msg}", "error")
                
                return OrderResult(
                    order_id="FAILED",
                    symbol=order.symbol,
                    side=order.side,
                    status=OrderStatus.REJECTED,
                    quantity=order.quantity,
                    filled_quantity=0.0,
                    price=0.0,
                    fee=0.0,
                    timestamp=datetime.now(),
                    raw={"error": error_msg}
                )
                
        except Exception as e:
            self.log(f"❌ Testnet order exception: {e}", "error")
            
            # Fallback to paper mode simulation
            self.log("⚠️ Falling back to paper mode...")
            return await super().execute_order(order)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Trading istatistikleri"""
        stats = super().get_statistics()
        stats["mode"] = "demo"
        stats["execution"] = "testnet"
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DemoMode Test")
    print("=" * 60)
    
    async def test():
        config = {
            "initial_balance": 10000,
            "leverage": 10
        }
        
        print("\n1️⃣ Creating DemoMode:")
        mode = DemoMode(config)
        
        print(f"   Mode type: {mode.mode_type}")
        print(f"   Real execution: {mode.is_real_execution}")
        print(f"   Testnet: {mode.is_testnet}")
        
        # Initialize (requires testnet connection)
        # await mode.initialize()
        
        print("\n2️⃣ Statistics:")
        stats = mode.get_statistics()
        print(f"   Mode: {stats['mode']}")
        
        print("\n✅ Test completed!")
    
    import asyncio
    asyncio.run(test())
    
    print("=" * 60)