#!/usr/bin/env python3
"""
modules/trading/modes/paper_mode.py
SuperBot - Paper Trading Mode
Author: SuperBot Team
Date: 2025-11-28
Versiyon: 2.0.0

Paper Trading Mode - Simulated Execution

PAPER MODE:
- Production data (WebSocket + REST)
- Sanal para (virtual balance)
- Simulated order execution (exchange'e gitmez)

FEATURES:
✅ Get config from Strategy (NO hardcoding)
✅ CacheManager'dan price al
✅ Virtual balance tracking
✅ Slippage/fee simulation

Usage:
    mode = PaperMode(config, logger)
    await mode.initialize()
    result = await mode.execute_order(order)

Dependencies:
    - python>=3.12
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

# Add project root to path for direct execution
import sys
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from modules.trading.modes.base_mode import (
    BaseMode,
    ModeType,
    OrderSide,
    OrderType,
    OrderStatus,
    Candle,
    Order,
    OrderResult,
    Position,
    Balance
)


class PaperMode(BaseMode):
    """
    Paper Trading Mode - Simulated Execution
    
    Production WebSocket data for virtual trading.
    Orders are simulated and do not go to the exchange.
    
    CONFIG SOURCE:
    - fee_rate: strategy.backtest_parameters.commission
    - slippage_rate: strategy.backtest_parameters.max_slippage
    - initial_balance: strategy.initial_balance
    - leverage: strategy.leverage
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any = None):
        """
        Args:
            config: Mode configuration (includes strategy, cache_manager, event_bus)
            logger: Logger instance
        """
        super().__init__(config, logger)
        
        # Strategy'den config al
        strategy = config.get("strategy")
        
        if strategy:
            backtest_params = getattr(strategy, 'backtest_parameters', {})
            # backtest_parameters uses percentage format (0.02 = 0.02%)
            # paper_mode uses decimal format (0.0002 = 0.02%)
            raw_commission = backtest_params.get("commission", 0.04)
            raw_slippage = backtest_params.get("max_slippage", 0.05)
            self.fee_rate = raw_commission / 100 if raw_commission > 0.01 else raw_commission
            self.slippage_rate = raw_slippage / 100 if raw_slippage > 0.01 else raw_slippage
            self.leverage = getattr(strategy, 'leverage', 1)
            self._initial_balance = getattr(strategy, 'initial_balance', 10000.0)
        else:
            # Fallback defaults
            self.fee_rate = config.get("fee_rate", 0.0004)
            self.slippage_rate = config.get("slippage_rate", 0.0005)
            self.leverage = config.get("leverage", 1)
            self._initial_balance = config.get("initial_balance", 10000.0)
        
        # Cache manager (for price)
        self.cache = config.get("cache_manager")

        # Connector (comes from TradingEngine)
        self._connector = config.get("connector")

        # Virtual state
        self._balance = self._initial_balance
        self._available_balance = self._initial_balance
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, OrderResult] = {}
        self._trade_history: List[Dict] = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def mode_type(self) -> ModeType:
        return ModeType.PAPER
    
    @property
    def is_live_data(self) -> bool:
        return True  # From WebSocket, actual data
    
    @property
    def is_real_execution(self) -> bool:
        return False  # Simulate execution
    
    @property
    def is_testnet(self) -> bool:
        return False  # Use production data (paper trade is simulated)

    # ═══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """Initialize paper mode"""
        self.log("🚀 PaperMode is starting...")
        self.log(f"   Balance: ${self._balance:,.2f}")
        self.log(f"   Leverage: {self.leverage}x")
        self.log(f"   Fee Rate: {self.fee_rate*100:.3f}%")
        self.log(f"   Slippage Rate: {self.slippage_rate*100:.3f}%")

        # Comes from Connector TradingEngine
        if self._connector:
            self.log("   ✅ Connector ready (from TradingEngine)")
        else:
            self.log("   ⚠️ No connector found", "warning")

        self._initialized = True
        self._running = True
        self.log("✅ PaperMode ready")
    
    async def shutdown(self) -> None:
        """Disable paper mode"""
        self.log("🛑 PaperMode is being disabled...")
        self._running = False
        
        # Final stats
        pnl = self._balance - self._initial_balance
        pnl_pct = (pnl / self._initial_balance) * 100
        
        self.log(f"   Final Balance: ${self._balance:,.2f}")
        self.log(f"   PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        self.log(f"   Total Trades: {len(self._trade_history)}")
        self.log("✅ PaperMode disabled")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════════════════
    
    async def execute_order(self, order: Order) -> OrderResult:
        """
        Simulated order execution
        
        Args:
            order: Order request
            
        Returns:
            OrderResult: Execution result
        """
        # Get the current price (from cache)
        current_price = self._get_current_price(order.symbol)
        if not current_price:
            return self._failed_order(order, "Price not found")
        
        # Slippage uygula
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage_rate)
        else:
            fill_price = current_price * (1 - self.slippage_rate)
        
        # Notional value
        notional = order.quantity * fill_price
        
        # Fee hesapla
        fee = notional * self.fee_rate
        
        # Balance check
        required_margin = notional / self.leverage
        if required_margin + fee > self._available_balance:
            return self._failed_order(order, "Insufficient balance")
        
        # Create order ID
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        
        # Update balance
        self._available_balance -= (required_margin + fee)
        
        # Update position
        self._update_position(order, fill_price)
        
        # Add to trade history
        self._trade_history.append({
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": fill_price,
            "fee": fee,
            "timestamp": datetime.now().isoformat()
        })
        
        result = OrderResult(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.FILLED,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.now(),
            raw=None
        )
        
        self._orders[order_id] = result
        
        self.log(
            f"✅ {order.symbol}: {order.side.value} @ ${fill_price:,.2f} "
            f"(qty: {order.quantity:.6f}, fee: ${fee:.4f})"
        )
        
        return result
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel in paper mode is always successful"""
        if order_id in self._orders:
            self.log(f"🗑️ Order cancelled: {order_id}")
            return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACCOUNT INFO
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_balance(self) -> Balance:
        """Virtual balance al"""
        # Unrealized PnL hesapla
        unrealized_pnl = 0.0
        for pos in self._positions.values():
            unrealized_pnl += pos.unrealized_pnl
        
        in_position = self._balance - self._available_balance
        
        return Balance(
            total=self._balance + unrealized_pnl,
            available=self._available_balance,
            in_position=in_position,
            unrealized_pnl=unrealized_pnl,
            currency="USDT"
        )
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position information"""
        return self._positions.get(symbol)
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self._positions.values())
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price"""
        return self._get_current_price(symbol) or 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA ACCESS
    # ═══════════════════════════════════════════════════════════════════════
    
    async def subscribe(self, symbols: List[str], timeframe: str = "5m") -> None:
        """Subscribe (done by DataManager)"""
        pass
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe"""
        pass
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> List[Candle]:
        """Historical candles (via connector)"""
        if not self._connector:
            return []
        
        try:
            klines = await self._connector.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            return [self._kline_to_candle(k, symbol, timeframe) for k in klines]
        except Exception as e:
            self.log(f"⚠️ Candle fetch error: {e}", "warning")
            return []
    
    def on_candle(self, callback) -> None:
        """Candle callback (managed by DataManager)"""
        pass
    
    def get_connector(self) -> Any:
        """Returns the connector (for DataManager)"""
        return self._connector
    
    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current price from the cache"""
        if self.cache:
            return self.cache.get(f"price:{symbol}")
        return None
    
    def _update_position(self, order: Order, fill_price: float):
        """Update position"""
        symbol = order.symbol

        if symbol in self._positions:
            # There is an existing position
            pos = self._positions[symbol]

            if (order.side == OrderSide.BUY and pos.side == "LONG") or \
               (order.side == OrderSide.SELL and pos.side == "SHORT"):
                # Same direction - increase position (pyramiding)
                total_qty = pos.quantity + order.quantity
                avg_price = (pos.entry_price * pos.quantity + fill_price * order.quantity) / total_qty
                pos.quantity = total_qty
                pos.entry_price = avg_price
            else:
                # Reverse direction - close/reduce position
                if order.quantity >= pos.quantity:
                    # Completely close - Calculate PnL and update balance
                    if pos.side == "LONG":
                        pnl = (fill_price - pos.entry_price) * pos.quantity
                    else:
                        pnl = (pos.entry_price - fill_price) * pos.quantity

                    # Add margin back + add PnL
                    self._available_balance += pos.margin + pnl
                    self._balance += pnl

                    self.log(f"💰 {symbol}: Position closed, PnL: ${pnl:+,.2f}, Balance: ${self._balance:,.2f}")

                    del self._positions[symbol]
                else:
                    # Partial close - calculate proportional PnL
                    close_ratio = order.quantity / pos.quantity
                    if pos.side == "LONG":
                        pnl = (fill_price - pos.entry_price) * order.quantity
                    else:
                        pnl = (pos.entry_price - fill_price) * order.quantity

                    # Add proportional margin back + add PnL
                    margin_released = pos.margin * close_ratio
                    self._available_balance += margin_released + pnl
                    self._balance += pnl
                    pos.margin -= margin_released
                    pos.quantity -= order.quantity

                    self.log(f"💰 {symbol}: Partial close, PnL: ${pnl:+,.2f}, Balance: ${self._balance:,.2f}")
        else:
            # New position
            side = "LONG" if order.side == OrderSide.BUY else "SHORT"
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                leverage=self.leverage,
                margin=order.quantity * fill_price / self.leverage
            )
    
    def _failed_order(self, order: Order, reason: str) -> OrderResult:
        """Create a failed order result"""
        self.log(f"❌ Order failed: {reason}", "error")
        
        return OrderResult(
            order_id=f"FAILED-{uuid.uuid4().hex[:8]}",
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.REJECTED,
            quantity=order.quantity,
            filled_quantity=0.0,
            price=0.0,
            fee=0.0,
            timestamp=datetime.now(),
            raw={"error": reason}
        )
    
    def _kline_to_candle(self, kline: Dict, symbol: str, timeframe: str) -> Candle:
        """Convert Kline dictionary to Candle."""
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=kline.get("timestamp", 0),
            open=float(kline.get("open", 0)),
            high=float(kline.get("high", 0)),
            low=float(kline.get("low", 0)),
            close=float(kline.get("close", 0)),
            volume=float(kline.get("volume", 0)),
            is_closed=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Trading istatistikleri"""
        pnl = self._balance - self._initial_balance
        
        return {
            "mode": "paper",
            "initial_balance": self._initial_balance,
            "current_balance": self._balance,
            "available_balance": self._available_balance,
            "pnl": pnl,
            "pnl_pct": (pnl / self._initial_balance) * 100,
            "total_trades": len(self._trade_history),
            "open_positions": len(self._positions),
            "leverage": self.leverage,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate
        }


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 PaperMode Test")
    print("=" * 60)
    
    async def test():
        # Mock config
        config = {
            "initial_balance": 10000,
            "leverage": 10,
            "fee_rate": 0.0004,
            "slippage_rate": 0.0005
        }
        
        print("\n1️⃣ Creating PaperMode:")
        mode = PaperMode(config)
        await mode.initialize()
        
        print("\n2️⃣ Balance kontrol:")
        balance = await mode.get_balance()
        print(f"   Total: ${balance.total:,.2f}")
        print(f"   Available: ${balance.available:,.2f}")
        
        print("\n3️⃣ Statistics:")
        stats = mode.get_statistics()
        print(f"   Mode: {stats['mode']}")
        print(f"   Leverage: {stats['leverage']}x")
        
        print("\n4️⃣ Shutdown:")
        await mode.shutdown()
        
        print("\n✅ Test completed!")
    
    import asyncio
    asyncio.run(test())
    
    print("=" * 60)