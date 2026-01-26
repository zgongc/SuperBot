#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/trading/modes/base_mode.py

SuperBot - Base Trading Mode (Abstract Interface)
Date: 2025-11-27
Versiyon: 1.0.0

Abstract base class for all trading modes.

MODES:
    PAPER  - Real data, virtual money, simulated order
    DEMO   - Real data, virtual money, testnet order
    LIVE   - Real data, real money, production order
    REPLAY - Past data (parquet), virtual money, simulated order

INTERFACE:
    get_candles()     - Data source (WebSocket/Parquet)
    execute_order()   - Order execution
    get_balance()     - Query balance
    get_position()    - Query position
    subscribe()       - Symbol subscribe (WebSocket)
    unsubscribe()     - Symbol unsubscribe

Usage:
    from modules.trading.modes import select_mode

    mode = select_mode("paper")
    await mode.initialize()

    async for candle in mode.get_candles(symbols):
        # process candle
        pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime


class ModeType(Enum):
    """Trading mode types"""
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"
    REPLAY = "replay"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


@dataclass
class Candle:
    """Candle data"""
    symbol: str
    timeframe: str
    timestamp: int          # Unix timestamp (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True  # Mum kapandi mi?


@dataclass
class Order:
    """Order request"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None       # For LIMIT
    stop_price: Optional[float] = None  # For STOP
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    quantity: float
    filled_quantity: float
    price: float              # Ortalama fill price
    fee: float
    timestamp: datetime
    raw: Optional[Dict] = None  # Exchange'den gelen raw response


@dataclass
class Position:
    """Position info"""
    symbol: str
    side: str                 # LONG/SHORT
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int
    margin: float
    liquidation_price: Optional[float] = None


@dataclass
class Balance:
    """Account balance"""
    total: float              # Total balance
    available: float          # Kullanilabilir
    in_position: float        # Pozisyonlarda kilitli
    unrealized_pnl: float     # Unrealized position PnL
    currency: str = "USDT"


class BaseMode(ABC):
    """
    Abstract base class for all trading modes.

    Her mode bu interface'i implement etmeli:
    - PAPER: WebSocket data, simulated execution
    - DEMO: WebSocket data, testnet execution
    - LIVE: WebSocket data, production execution
    - REPLAY: Parquet data, simulated execution
    """

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        """
        Args:
            config: Mode configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self._initialized = False
        self._running = False

    @property
    @abstractmethod
    def mode_type(self) -> ModeType:
        """Mode type"""
        pass

    @property
    @abstractmethod
    def is_live_data(self) -> bool:
        """Is it real-time data? (WebSocket)"""
        pass

    @property
    @abstractmethod
    def is_real_execution(self) -> bool:
        """Gercek order execution mi? (Exchange'e gidiyor mu)"""
        pass

    # ═══════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the mode.
        - Baglantilari kur
        - Load initial data
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Turn off the mode.
        - Close connections
        - Clean up resources
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # DATA FEED
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def subscribe(self, symbols: List[str], timeframe: str) -> None:
        """
        Symbol'lere subscribe ol.

        Args:
            symbols: List of symbols (["BTCUSDT", "ETHUSDT"])
            timeframe: Timeframe ("5m", "15m", etc.)
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """
        Symbol'lerden unsubscribe ol.

        Args:
            symbols: List of symbols
        """
        pass

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> List[Candle]:
        """
        Gecmis candle'lari al.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            limit: Maximum number of candles.

        Returns:
            List[Candle]: List of candles (ordered from old to new)
        """
        pass

    @abstractmethod
    def on_candle(self, callback) -> None:
        """
        Register the candle callback.

        Args:
            callback: async def callback(candle: Candle)
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def execute_order(self, order: Order) -> OrderResult:
        """
        Order calistir.

        Args:
            order: Order request

        Returns:
            OrderResult: Execution result
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel the order.

        Args:
            symbol: Symbol
            order_id: Order ID

        Returns:
            bool: Is it successful?
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # ACCOUNT INFO
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def get_balance(self) -> Balance:
        """
        Hesap bakiyesini al.

        Returns:
            Balance: Balance information
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position information.

        Args:
            symbol: Symbol

        Returns:
            Position or None
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Take all open positions.

        Returns:
            List[Position]: List of positions
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """
        Guncel fiyati al.

        Args:
            symbol: Symbol

        Returns:
            float: Current price
        """
        pass

    def log(self, msg: str, level: str = "info") -> None:
        """Log helper"""
        if self.logger:
            getattr(self.logger, level)(msg)


# ═══════════════════════════════════════════════════════════════════════
# MODE SELECTOR (Factory)
# ═══════════════════════════════════════════════════════════════════════

def select_mode(
    mode: str,
    config: Dict[str, Any] = None,
    logger: Any = None
) -> BaseMode:
    """
    Mode selector (factory function).

    Args:
        mode: Mode name ("paper", "demo", "live", "replay")
        config: Mode configuration
        logger: Logger instance

    Returns:
        BaseMode: Selected mode instance

    Raises:
        ValueError: Unknown mode
    """
    config = config or {}

    mode_lower = mode.lower()

    if mode_lower == "paper":
        from modules.trading.modes.paper_mode import PaperMode
        return PaperMode(config, logger)

    elif mode_lower == "demo":
        from modules.trading.modes.demo_mode import DemoMode
        return DemoMode(config, logger)

    elif mode_lower == "live":
        from modules.trading.modes.live_mode import LiveMode
        return LiveMode(config, logger)

    elif mode_lower == "replay":
        from modules.trading.modes.replay_mode import ReplayMode
        return ReplayMode(config, logger)

    else:
        raise ValueError(f"Unknown mode: {mode}. Valid: paper, demo, live, replay")


# ═══════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("BaseMode - Abstract Interface Test")
    print("=" * 60)

    # Dataclass test
    print("\n1. Candle dataclass:")
    candle = Candle(
        symbol="BTCUSDT",
        timeframe="5m",
        timestamp=1700000000000,
        open=95000.0,
        high=95500.0,
        low=94800.0,
        close=95200.0,
        volume=100.5,
        is_closed=True
    )
    print(f"   {candle}")

    print("\n2. Order dataclass:")
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.01
    )
    print(f"   {order}")

    print("\n3. Position dataclass:")
    position = Position(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.01,
        entry_price=95000.0,
        current_price=95500.0,
        unrealized_pnl=5.0,
        unrealized_pnl_pct=0.53,
        leverage=10,
        margin=95.0
    )
    print(f"   {position}")

    print("\n4. Balance dataclass:")
    balance = Balance(
        total=10000.0,
        available=9000.0,
        in_position=1000.0,
        unrealized_pnl=50.0
    )
    print(f"   {balance}")

    print("\n5. ModeType enum:")
    for mode in ModeType:
        print(f"   {mode.name} = {mode.value}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
