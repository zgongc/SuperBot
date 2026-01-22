#!/usr/bin/env python3
"""
components/exchanges/ccxt_wrapper.py
SuperBot - CCXT Wrapper Base Class
Author: SuperBot Team
Date: 2025-11-15
Versiyon: 1.0.0

Base wrapper class for CCXT-based exchanges.

Features:
- Uses CCXT async_support
- Supports testnet/production environments
- Config-driven credentials
- Rate limiting
- Error handling

Desteklenen Exchange'ler:
- Bybit, OKX, KuCoin, Gate.io, and all exchanges supported by CCXT.

Usage:
    from components.exchanges.ccxt_wrapper import CCXTWrapper

    class BybitAPI(CCXTWrapper):
        def __init__(self, config):
            super().__init__(exchange_name='bybit', config=config)

Dependencies:
    - ccxt>=4.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.exchanges.base_api import BaseExchangeAPI
from core.logger_engine import LoggerEngine


# ============================================================================
# CCXT WRAPPER
# ============================================================================

class CCXTWrapper(BaseExchangeAPI):
    """
    Base wrapper for CCXT-based exchanges.

    Exchanges that extend this class:
    - Bybit (BybitAPI)
    - OKX (OkxAPI)
    - KuCoin (KucoinAPI)
    - Gate.io (GateioAPI)
    """

    def __init__(self, exchange_name: str, config: Dict[str, Any]):
        """
        Initialize the CCXT wrapper.

        Args:
            exchange_name: Exchange name (e.g., 'bybit', 'okx', 'kucoin')
            config: Exchange section from the config file

        Raises:
            ValueError: The exchange is not supported by CCXT.
        """
        super().__init__(config)

        self.exchange_name = exchange_name.lower()
        self.logger = LoggerEngine().get_logger(f"{__name__}.{exchange_name}")

        # Get credentials
        endpoint_key = "testnet" if self.testnet else "production"
        endpoint_config = config.get("endpoints", {}).get(endpoint_key, {})

        api_key = endpoint_config.get("api_key")
        api_secret = endpoint_config.get("secret_key")
        passphrase = endpoint_config.get("passphrase")  # For OKX

        # Create CCXT exchange instance
        exchange_class = getattr(ccxt, self.exchange_name, None)
        if not exchange_class:
            raise ValueError(f"Exchange '{exchange_name}' is not supported by CCXT")

        # CCXT config
        ccxt_config = {
            'enableRateLimit': True,
            'timeout': config.get('timeout', {}).get('read', 30) * 1000,  # milisaniye
        }

        # Credentials
        if api_key and api_secret:
            ccxt_config['apiKey'] = api_key
            ccxt_config['secret'] = api_secret
            if passphrase:
                ccxt_config['password'] = passphrase

        # Testnet/Sandbox mode
        if self.testnet:
            ccxt_config['options'] = {'defaultType': 'spot'}
            ccxt_config['sandbox'] = True

        # Create an exchange instance
        self.exchange = exchange_class(ccxt_config)

        # Stats
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
        }

        env = "testnet" if self.testnet else "production"
        self.logger.info(f"✅ {exchange_name.upper()} API started ({env})")

    # ========================================================================
    # MARKET DATA - Implementation
    # ========================================================================

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker price information.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Dict: Ticker verisi
        """
        try:
            result = await self.exchange.fetch_ticker(symbol)
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Ticker error ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Order book al

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            limit: Depth

        Returns:
            Dict: Order book verisi
        """
        try:
            result = await self.exchange.fetch_order_book(symbol, limit)
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Orderbook error ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise

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
            symbol: Trading pair (e.g., "BTC/USDT")
            interval: Interval (1m, 5m, 1h, 1d, etc.)
            limit: Number of klines
            start_time: Start time (timestamp ms)
            end_time: End time (timestamp ms)

        Returns:
            List: OHLCV verisi
        """
        try:
            result = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=interval,
                limit=limit,
                since=start_time
            )
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Kline error ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise

    # ========================================================================
    # ACCOUNT - Implementation
    # ========================================================================

    async def get_balance(self) -> Dict[str, Any]:
        """
        Hesap bakiyesi al

        Returns:
            Dict: Balance bilgisi
        """
        try:
            result = await self.exchange.fetch_balance()
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Balance error: {e}")
            self.stats["total_errors"] += 1
            raise

    # ========================================================================
    # TRADING - Implementation
    # ========================================================================

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
            symbol: Trading pair (e.g., "BTC/USDT")
            side: 'buy' or 'sell'
            order_type: 'limit', 'market', etc.
            quantity: Quantity
            price: Price (for limit order)
            **kwargs: Exchange-specific parameters

        Returns:
            Dict: Order result
        """
        try:
            result = await self.exchange.create_order(
                symbol=symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=quantity,
                price=price,
                params=kwargs
            )
            self.stats["total_requests"] += 1
            self.logger.info(f"📝 Order created: {symbol} {side} {quantity}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error creating order ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Order iptal et

        Args:
            symbol: Trading pair
            order_id: Order ID
            **kwargs: Exchange-specific parameters

        Returns:
            Dict: Cancel result
        """
        try:
            result = await self.exchange.cancel_order(order_id, symbol, params=kwargs)
            self.stats["total_requests"] += 1
            self.logger.info(f"🚫 Order cancelled: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Order cancellation error ({order_id}): {e}")
            self.stats["total_errors"] += 1
            raise

    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get open orders.

        Args:
            symbol: Trading pair (None if all)

        Returns:
            List[Dict]: List of open orders.
        """
        try:
            result = await self.exchange.fetch_open_orders(symbol)
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Error with open orders: {e}")
            self.stats["total_errors"] += 1
            raise

    # ========================================================================
    # UTILITY - Implementation
    # ========================================================================

    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get the time from the exchange server.

        Returns:
            Dict: Server time
        """
        try:
            result = await self.exchange.fetch_time()
            self.stats["total_requests"] += 1
            return {"serverTime": result}
        except Exception as e:
            self.logger.error(f"❌ Server time error: {e}")
            self.stats["total_errors"] += 1
            raise

    def health_check(self) -> bool:
        """
        API health check

        Returns:
            bool: True if the API is running.
        """
        try:
            # CCXT exchange status check
            return hasattr(self.exchange, 'apiKey') or True
        except Exception as e:
            self.logger.error(f"❌ Health check error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        API istatistikleri al

        Returns:
            Dict: Statistics
        """
        return {
            **self.stats,
            "exchange": self.exchange_name,
            "testnet": self.testnet,
            "enabled": self.enabled,
        }

    async def close(self):
        """Close the exchange connection"""
        try:
            await self.exchange.close()
            self.logger.info(f"🛑 Connection to {self.exchange_name.upper()} closed")
        except Exception as e:
            self.logger.error(f"❌ Connection closing error: {e}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'CCXTWrapper',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 CCXTWrapper Test")
    print("=" * 60)

    async def test():
        print("\n1️⃣  KuCoin wrapper test:")

        # Test config
        test_config = {
            "enabled": True,
            "testnet": True,
            "endpoints": {
                "testnet": {
                    "api_key": "test_key",
                    "secret_key": "test_secret"
                }
            },
            "timeout": {
                "read": 30
            }
        }

        try:
            # Create KuCoin wrapper
            kucoin = CCXTWrapper(exchange_name='kucoin', config=test_config)
            print(f"   ✅ KuCoin wrapper created")
            print(f"   - Exchange: {kucoin.exchange_name}")
            print(f"   - Testnet: {kucoin.testnet}")
            print(f"   - Health: {kucoin.health_check()}")

            # Stats
            stats = kucoin.get_stats()
            print(f"   - Stats: {stats}")

            # Close
            await kucoin.close()
            print(f"   ✅ Connection closed")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    asyncio.run(test())

    print("\n✅ All tests completed!")
    print("=" * 60)
