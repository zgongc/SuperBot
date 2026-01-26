#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/trading/modes/live_mode.py

SuperBot - Live Trading Mode (Production)
Date: 2025-11-27
Versiyon: 1.1.0

LIVE MODE:
    - Real data (WebSocket) - Production
    - Gercek para (production balance)
    - Production order execution (Binance Futures)

!!! DIKKAT !!!
    This mode operates with REAL MONEY!
    Dikkatli kullanin!

Data Sources:
    - BinanceAPI (testnet=False) -> Production REST API
    - WebSocketEngine (testnet=False) -> Production WebSocket

Production URL:
    API: https://fapi.binance.com
    WS:  wss://fstream.binance.com

Usage:
    mode = LiveMode(config, logger)
    await mode.initialize()

    # GERCEK ORDER!
    result = await mode.execute_order(order)
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from collections import defaultdict

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

# Centralized engines
from components.exchanges.binance_api import BinanceAPI
from components.managers.websocket_engine import WebSocketEngine


class LiveMode(BaseMode):
    """
    Live Trading Mode (Production)

    !!! TRANSACTIONS ARE MADE WITH REAL MONEY !!!

    Binance Futures production API kullanir.
    Uses the central BinanceAPI and WebSocketEngine (testnet=False).
    """

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)

        # Production API credentials
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")

        if not self.api_key or not self.api_secret:
            raise ValueError("LIVE MODE: API key/secret IS REQUIRED!")

        # Safety checks
        self._safety_enabled = config.get("safety_enabled", True)
        self._max_position_size = config.get("max_position_size", 1000)  # USD
        self._max_daily_loss = config.get("max_daily_loss", 500)  # USD
        self._daily_pnl = 0.0

        # Settings
        self.default_leverage = config.get("default_leverage", 10)

        # Centralized engines (will be initialized in initialize())
        self._api: Optional[BinanceAPI] = None
        self._websocket: Optional[WebSocketEngine] = None

        # Callbacks & data
        self._candle_callbacks: List[Callable] = []
        self._subscribed_symbols: List[str] = []
        self._current_prices: Dict[str, float] = {}
        self._candle_buffer: Dict[str, List[Candle]] = defaultdict(list)

        # Event bus reference
        self._event_bus = None

    @property
    def mode_type(self) -> ModeType:
        return ModeType.LIVE

    @property
    def is_live_data(self) -> bool:
        return True

    @property
    def is_real_execution(self) -> bool:
        return True  # GERCEK PARA!

    # ═══════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """Initialize live mode with production connection"""
        self.log("=" * 50)
        self.log("!!! LIVE MODE BASLATILIYOR !!!")
        self.log("!!! TRANSACTION WILL BE MADE WITH REAL MONEY !!!")
        self.log("=" * 50)

        # Event bus'i al
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
            self.log("   EventBus baglandi")
        except Exception:
            self._event_bus = None
            self.log("   EventBus not found", "warning")

        # BinanceAPI olustur (production - testnet=False)
        api_config = {
            **self.config.get("binance", {}),
            "testnet": False,  # PRODUCTION!
            "enabled": True,
            "api_key": self.api_key,
            "api_secret": self.api_secret
        }
        self._api = BinanceAPI(config=api_config)
        self.log("   BinanceAPI (PRODUCTION) olusturuldu")

        # Test the production connection
        try:
            account = await self._api.get_account()
            balance = float(account.get("totalWalletBalance", 0))
            self.log(f"   Production baglantisi OK")
            self.log(f"   REAL BALANCE: ${balance:,.2f}")
        except Exception as e:
            self.log(f"   Production baglanti HATASI: {e}", "error")
            raise

        # WebSocket engine'i olustur (production - testnet=False)
        ws_config = {
            **self.config.get("websocket", {}),
            "testnet": False  # PRODUCTION WS!
        }
        self._websocket = WebSocketEngine(config=ws_config, event_bus=self._event_bus)
        await self._websocket.start()
        self.log("   WebSocket engine (PRODUCTION) hazir")

        self._initialized = True
        self._running = True

        self.log("LIVE MODE HAZIR - DIKKATLI KULLANIN!")

    async def shutdown(self) -> None:
        """Shutdown live mode"""
        self.log("Live Mode kapatiliyor...")
        self._running = False

        # Show open positions with a WARNING
        try:
            positions = await self.get_positions()
            if positions:
                self.log("=" * 50)
                self.log("!!! THERE ARE OPEN POSITIONS !!!")
                for pos in positions:
                    self.log(f"   {pos.symbol}: {pos.side} {pos.quantity} @ ${pos.entry_price:,.2f}")
                    self.log(f"   PnL: ${pos.unrealized_pnl:+,.2f}")
                self.log("=" * 50)
        except Exception:
            pass

        # Close WebSocket
        if self._websocket:
            await self._websocket.stop()
            self._websocket = None

        # BinanceAPI cleanup (if needed)
        self._api = None

        self._initialized = False
        self.log("Live Mode kapatildi")

    # ═══════════════════════════════════════════════════════════════════
    # SAFETY CHECKS
    # ═══════════════════════════════════════════════════════════════════

    def _check_safety(self, order: Order) -> None:
        """Safety checks before order execution"""
        if not self._safety_enabled:
            return

        # Daily loss check
        if self._daily_pnl < -self._max_daily_loss:
            raise ValueError(
                f"SECURITY: Daily loss limit exceeded!"
                f"(${self._daily_pnl:,.2f} < -${self._max_daily_loss:,.2f})"
            )

        # Position size check
        price = self._current_prices.get(order.symbol, 0)
        if price > 0:
            position_value = order.quantity * price
            if position_value > self._max_position_size:
                raise ValueError(
                    f"SECURITY: Position size is too large!"
                    f"(${position_value:,.2f} > ${self._max_position_size:,.2f})"
                )

    # ═══════════════════════════════════════════════════════════════════
    # DATA FEED
    # ═══════════════════════════════════════════════════════════════════

    async def subscribe(self, symbols: List[str], timeframe: str) -> None:
        """Subscribe to symbols via WebSocket"""
        if not self._websocket:
            self.log("WebSocket not found, subscription failed", "error")
            return

        await self._websocket.subscribe(
            symbols=symbols,
            channels=[f"kline_{timeframe}"]
        )

        self._subscribed_symbols.extend(symbols)
        self.log(f"Subscribed: {len(symbols)} symbols @ {timeframe}")

        # Internal candle handler'i bagla
        self._websocket.on_candle = self._handle_websocket_candle

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        if not self._websocket:
            return

        await self._websocket.unsubscribe(symbols)

        for s in symbols:
            if s in self._subscribed_symbols:
                self._subscribed_symbols.remove(s)

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> List[Candle]:
        """Get historical candles from production API"""
        # Try from the buffer once
        buffer_key = f"{symbol}_{timeframe}"
        if buffer_key in self._candle_buffer:
            cached = self._candle_buffer[buffer_key]
            if len(cached) >= limit:
                return cached[-limit:]

        # BinanceAPI'den cek (production)
        if not self._api:
            self.log("BinanceAPI not initialized", "error")
            return []

        try:
            klines = await self._api.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )

            candles = []
            for k in klines:
                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    is_closed=True
                )
                candles.append(candle)

            self._candle_buffer[buffer_key] = candles
            return candles

        except Exception as e:
            self.log(f"Candle fetch hatasi: {e}", "error")
            return []

    def on_candle(self, callback: Callable) -> None:
        """Register candle callback"""
        self._candle_callbacks.append(callback)

    async def _handle_websocket_candle(self, data: Dict) -> None:
        """Internal WebSocket candle handler"""
        try:
            symbol = data.get("symbol", data.get("s", ""))
            is_closed = data.get("is_closed", data.get("x", False))

            candle = Candle(
                symbol=symbol,
                timeframe=data.get("timeframe", data.get("i", "5m")),
                timestamp=data.get("timestamp", data.get("t", 0)),
                open=float(data.get("open", data.get("o", 0))),
                high=float(data.get("high", data.get("h", 0))),
                low=float(data.get("low", data.get("l", 0))),
                close=float(data.get("close", data.get("c", 0))),
                volume=float(data.get("volume", data.get("v", 0))),
                is_closed=is_closed
            )

            # Update the price
            self._current_prices[symbol] = candle.close

            # Add to buffer
            if is_closed:
                buffer_key = f"{symbol}_{candle.timeframe}"
                self._candle_buffer[buffer_key].append(candle)
                if len(self._candle_buffer[buffer_key]) > 1000:
                    self._candle_buffer[buffer_key] = self._candle_buffer[buffer_key][-1000:]

            # Callback'leri cagir
            for callback in self._candle_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(candle)
                else:
                    callback(candle)

            # Publish to the event bus
            if self._event_bus and is_closed:
                await self._event_bus.publish_async(
                    topic=f"candle.{symbol}.{candle.timeframe}",
                    data={"candle": candle.__dict__},
                    source="live_mode"
                )

        except Exception as e:
            self.log(f"Candle handler hatasi: {e}", "error")

    # ═══════════════════════════════════════════════════════════════════
    # ORDER EXECUTION (PRODUCTION - GERCEK PARA!)
    # ═══════════════════════════════════════════════════════════════════

    async def execute_order(self, order: Order) -> OrderResult:
        """
        Execute order on Binance Futures Production.

        !!! TRANSACTIONS WITH REAL MONEY !!!
        """
        if not self._api:
            raise RuntimeError("BinanceAPI not initialized")

        # Safety check
        self._check_safety(order)

        self.log(f"!!! GERCEK ORDER: {order.side.value} {order.quantity} {order.symbol} !!!")

        try:
            # Production'a order gonder via BinanceAPI
            response = await self._api.create_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price
            )

            # Parse the response
            order_id = str(response.get("orderId", ""))
            status_str = response.get("status", "NEW")
            fill_price = float(response.get("avgPrice", 0) or response.get("price", 0))
            filled_qty = float(response.get("executedQty", 0))

            # Status mapping
            status_map = {
                "NEW": OrderStatus.PENDING,
                "FILLED": OrderStatus.FILLED,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "CANCELED": OrderStatus.CANCELED,
                "REJECTED": OrderStatus.REJECTED
            }
            status = status_map.get(status_str, OrderStatus.PENDING)

            # Fee hesapla
            fee = filled_qty * fill_price * 0.0004 if fill_price > 0 else 0

            result = OrderResult(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                status=status,
                quantity=order.quantity,
                filled_quantity=filled_qty,
                price=fill_price,
                fee=fee,
                timestamp=datetime.now(),
                raw=response
            )

            self.log(
                f"GERCEK ORDER DOLDU: {order.side.value} {filled_qty} {order.symbol} "
                f"@ ${fill_price:,.2f} (OrderID: {order_id})"
            )

            # Publish to the event bus
            if self._event_bus:
                await self._event_bus.publish_async(
                    topic="order.filled",
                    data={
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "price": fill_price,
                        "quantity": filled_qty,
                        "fee": fee,
                        "mode": "live"
                    },
                    source="live_mode"
                )

            return result

        except Exception as e:
            self.log(f"!!! GERCEK ORDER HATASI: {e} !!!", "error")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order on production"""
        if not self._api:
            return False

        try:
            await self._api.cancel_order(symbol=symbol, order_id=order_id)
            self.log(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.log(f"Cancel hatasi: {e}", "error")
            return False

    # ═══════════════════════════════════════════════════════════════════
    # ACCOUNT INFO (PRODUCTION)
    # ═══════════════════════════════════════════════════════════════════

    async def get_balance(self) -> Balance:
        """Get production account balance"""
        if not self._api:
            return Balance(total=0, available=0, in_position=0, unrealized_pnl=0)

        try:
            account = await self._api.get_account()

            total = float(account.get("totalWalletBalance", 0))
            available = float(account.get("availableBalance", 0))
            unrealized = float(account.get("totalUnrealizedProfit", 0))
            in_position = float(account.get("totalMarginBalance", 0)) - available

            return Balance(
                total=total,
                available=available,
                in_position=in_position,
                unrealized_pnl=unrealized,
                currency="USDT"
            )

        except Exception as e:
            self.log(f"Balance hatasi: {e}", "error")
            return Balance(total=0, available=0, in_position=0, unrealized_pnl=0)

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        if not self._api:
            return None

        try:
            positions = await self._api.get_position_risk(symbol=symbol)

            for pos in positions:
                qty = float(pos.get("positionAmt", 0))
                if qty == 0:
                    continue

                entry = float(pos.get("entryPrice", 0))
                current = float(pos.get("markPrice", 0))
                leverage = int(pos.get("leverage", 1))
                unrealized = float(pos.get("unRealizedProfit", 0))
                margin = float(pos.get("isolatedMargin", 0)) or (abs(qty) * entry / leverage)
                liquidation = float(pos.get("liquidationPrice", 0))

                side = "LONG" if qty > 0 else "SHORT"
                pnl_pct = (unrealized / (entry * abs(qty))) * 100 if entry > 0 else 0

                return Position(
                    symbol=symbol,
                    side=side,
                    quantity=abs(qty),
                    entry_price=entry,
                    current_price=current,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=pnl_pct,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation if liquidation > 0 else None
                )

            return None

        except Exception as e:
            self.log(f"Position hatasi: {e}", "error")
            return None

    async def get_positions(self) -> List[Position]:
        """Get all open positions"""
        if not self._api:
            return []

        try:
            positions = await self._api.get_position_risk()

            result = []
            for pos in positions:
                qty = float(pos.get("positionAmt", 0))
                if qty == 0:
                    continue

                entry = float(pos.get("entryPrice", 0))
                current = float(pos.get("markPrice", 0))
                leverage = int(pos.get("leverage", 1))
                unrealized = float(pos.get("unRealizedProfit", 0))
                margin = float(pos.get("isolatedMargin", 0)) or (abs(qty) * entry / leverage)
                liquidation = float(pos.get("liquidationPrice", 0))

                side = "LONG" if qty > 0 else "SHORT"
                pnl_pct = (unrealized / (entry * abs(qty))) * 100 if entry > 0 else 0

                result.append(Position(
                    symbol=pos.get("symbol", ""),
                    side=side,
                    quantity=abs(qty),
                    entry_price=entry,
                    current_price=current,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=pnl_pct,
                    leverage=leverage,
                    margin=margin,
                    liquidation_price=liquidation if liquidation > 0 else None
                ))

            return result

        except Exception as e:
            self.log(f"Positions hatasi: {e}", "error")
            return []

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from production"""
        if symbol in self._current_prices:
            return self._current_prices[symbol]

        if not self._api:
            return 0.0

        try:
            ticker = await self._api.get_ticker(symbol)
            price = float(ticker.get("lastPrice", 0))
            self._current_prices[symbol] = price
            return price
        except Exception as e:
            self.log(f"Could not retrieve price for {symbol}: {e}", "error")
            return 0.0

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily PnL for safety tracking"""
        self._daily_pnl += pnl
        self.log(f"Gunluk PnL: ${self._daily_pnl:+,.2f}")

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL (midnight'ta cagirilmali)"""
        self._daily_pnl = 0.0
        self.log("Gunluk PnL sifirlandi")


# ═══════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("LiveMode Test (PRODUCTION)")
    print("=" * 60)
    print()
    print("!!! WARNING: This module processes transactions with REAL MONEY!!!")
    print("!!! Use DEMO or PAPER mode for testing !!!")
    print()

    async def test():
        # Show only the mode type without an API key
        try:
            config = {
                "api_key": "",
                "api_secret": "",
                "safety_enabled": True,
                "max_position_size": 100,
                "max_daily_loss": 50
            }

            mode = LiveMode(config)
            print("LiveMode could not be created - API key required (expected)")

        except ValueError as e:
            print(f"Expected error: {e}")
            print("\nLive mode API key olmadan calistirilamaz.")
            print("Bu bir guvenlik onlemidir.")

    asyncio.run(test())

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
