#!/usr/bin/env python3
"""
components/exchanges/base_api.py
SuperBot - Base Exchange API
Author: SuperBot Team
Date: 2025-11-15
Versiyon: 1.0.0

Abstract base class for all exchange APIs.

Features:
- Tek interface (BaseExchangeAPI)
- Market data methods (ticker, orderbook, klines)
- Trading methods (create_order, cancel_order)
- Account methods (balance, open orders)
- Utility methods (server_time, health_check, stats)

Usage:
    from components.exchanges.base_api import BaseExchangeAPI

    class BinanceAPI(BaseExchangeAPI):
        def __init__(self, config):
            super().__init__(config)

        async def get_ticker(self, symbol: str):
            # Implementation
            pass

Dependencies:
    - python>=3.10
"""

from __future__ import annotations

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))


# ============================================================================
# BASE EXCHANGE API
# ============================================================================

class BaseExchangeAPI(ABC):
    """
    Abstract base class for all exchange APIs.

    Each exchange API implements this interface and provides common methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base API.

        Args:
            config: Exchange-specific section from the configuration file
                    (e.g., config/connectors.yaml -> binance section)

        Example config:
            {
                "enabled": true,
                "testnet": true,
                "endpoints": {
                    "testnet": {"api_key": "...", "secret_key": "..."},
                    "production": {"api_key": "...", "secret_key": "..."}
                },
                "rate_limit": {...},
                "retry": {...}
            }
        """
        self.config = config
        self.testnet = config.get('testnet', True)
        self.enabled = config.get('enabled', False)

    # ========================================================================
    # MARKET DATA - Abstract Methods
    # ========================================================================

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker price information.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Dict: Ticker verisi
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "45000.00",
                    "volume": "123456.78",
                    ...
                }

        Raises:
            Exception: In case of an API error.
        """
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Order book al

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Depth (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Dict: Order book verisi
                {
                    "bids": [[price, quantity], ...],
                    "asks": [[price, quantity], ...],
                    "lastUpdateId": 123456
                }

        Raises:
            Exception: In case of an API error.
        """
        pass

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List:
        """
        Kline/Candlestick data al

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of klines (max 1000)
            start_time: Start time (timestamp ms)
            end_time: End time (timestamp ms)

        Returns:
            List: Kline verisi
                [
                    [open_time, open, high, low, close, volume, close_time, ...],
                    ...
                ]

        Raises:
            Exception: In case of an API error.
        """
        pass

    # ========================================================================
    # ACCOUNT - Abstract Methods
    # ========================================================================

    @abstractmethod
    async def get_balance(self) -> Dict[str, Any]:
        """
        Hesap bakiyesi al

        Returns:
            Dict: Balance bilgisi
                {
                    "balances": [
                        {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                        {"asset": "USDT", "free": "10000", "locked": "2000"},
                        ...
                    ],
                    "totalWalletBalance": "15000.00"
                }

        Raises:
            Exception: In case of an API error.
        """
        pass

    # ========================================================================
    # TRADING - Abstract Methods
    # ========================================================================

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an order.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: BUY or SELL
            order_type: LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, etc.
            quantity: Quantity
            price: Price (required for LIMIT order)
            **kwargs: Exchange-specific parameters

        Returns:
            Dict: Order result
                {
                    "orderId": 123456,
                    "symbol": "BTCUSDT",
                    "status": "FILLED",
                    "executedQty": "0.001",
                    "price": "45000.00",
                    ...
                }

        Raises:
            Exception: In case of an API error.
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Order iptal et

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            order_id: Order ID
            **kwargs: Exchange-specific parameters

        Returns:
            Dict: Cancel result
                {
                    "orderId": 123456,
                    "symbol": "BTCUSDT",
                    "status": "CANCELED",
                    ...
                }

        Raises:
            Exception: In case of an API error.
        """
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get open orders.

        Args:
            symbol: Trading pair (None if all symbols)

        Returns:
            List[Dict]: List of open orders.
                [
                    {
                        "orderId": 123456,
                        "symbol": "BTCUSDT",
                        "status": "NEW",
                        "side": "BUY",
                        "price": "45000.00",
                        "quantity": "0.001",
                        ...
                    },
                    ...
                ]

        Raises:
            Exception: In case of an API error.
        """
        pass

    # ========================================================================
    # UTILITY - Abstract Methods
    # ========================================================================

    @abstractmethod
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get the time from the exchange server.

        Returns:
            Dict: Server time
                {
                    "serverTime": 1234567890000
                }

        Raises:
            Exception: In case of an API error.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        API health check

        Returns:
            bool: True if the API is working, False if there is an error.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        API istatistikleri al

        Returns:
            Dict: Statistics
                {
                    "total_requests": 1234,
                    "total_errors": 5,
                    "cache_hits": 100,
                    "cache_misses": 50,
                    "testnet": true,
                    ...
                }
        """
        pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BaseExchangeAPI',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 BaseExchangeAPI Test")
    print("=" * 60)

    print("\n1️⃣  Abstract class test:")
    try:
        # You cannot directly instantiate the abstract class.
        api = BaseExchangeAPI(config={})
        print("   ❌ Error: Abstract class was instantiated!")
    except TypeError as e:
        print(f"   ✅ Expected error: {e}")

    print("\n2️⃣  Concrete implementation test:")

    class TestExchangeAPI(BaseExchangeAPI):
        """Concrete implementation for testing"""

        async def get_ticker(self, symbol: str):
            return {"symbol": symbol, "price": "45000"}

        async def get_orderbook(self, symbol: str, limit: int = 100):
            return {"bids": [], "asks": []}

        async def get_klines(self, symbol: str, interval: str, limit: int = 100,
                           start_time=None, end_time=None):
            return []

        async def get_balance(self):
            return {"balances": []}

        async def create_order(self, symbol: str, side: str, order_type: str,
                             quantity: float, price=None, **kwargs):
            return {"orderId": 123456}

        async def cancel_order(self, symbol: str, order_id: str, **kwargs):
            return {"orderId": order_id, "status": "CANCELED"}

        async def get_open_orders(self, symbol=None):
            return []

        async def get_server_time(self):
            return {"serverTime": 1234567890000}

        def health_check(self):
            return True

        def get_stats(self):
            return {"total_requests": 0}

    test_config = {
        "enabled": True,
        "testnet": True,
        "endpoints": {
            "testnet": {"api_key": "test_key", "secret_key": "test_secret"}
        }
    }

    test_api = TestExchangeAPI(config=test_config)
    print(f"   ✅ Test API created")
    print(f"   - testnet: {test_api.testnet}")
    print(f"   - enabled: {test_api.enabled}")
    print(f"   - health: {test_api.health_check()}")

    print("\n✅ All tests completed!")
    print("=" * 60)
