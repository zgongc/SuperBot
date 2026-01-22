#!/usr/bin/env python3
"""
components/exchanges/binance_api.py
SuperBot - Binance Exchange API
Author: SuperBot Team
Date: 2025-11-26
Versiyon: 2.0.0

Binance Exchange API Implementation
using the python-binance library with an async executor pattern.

Features:
- BaseExchangeAPI implementation
- python-binance library (native library)
- Async executor pattern (non-blocking)
- Config-driven credentials (testnet/production)
- Cache manager integration
- Stats tracking
- SPOT + FUTURES support

Usage:
    from components.exchanges import BinanceAPI
    from core.config_engine import ConfigEngine

    config = ConfigEngine().get('binance')
    binance = BinanceAPI(config=config)

    # Market data
    ticker = await binance.get_ticker('BTCUSDT')

    # Trading
    order = await binance.create_order(
        symbol='BTCUSDT',
        side='BUY',
        order_type='MARKET',
        quantity=0.001
    )

Dependencies:
    - python-binance>=1.0.17
    - asyncio
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.exchanges.base_api import BaseExchangeAPI
from core.logger_engine import LoggerEngine


# ============================================================================
# BINANCE API
# ============================================================================

class BinanceAPI(BaseExchangeAPI):
    """
    Binance Exchange API

    python-binance library kullanarak Binance API'yi wrap eder.
    All sync calls are wrapped with an async executor (non-blocking).

    Config source: config/connectors.yaml -> binance section
    """

    def __init__(
        self,
        config: Dict[str, Any],
        cache_manager: Optional[Any] = None
    ):
        """
        Initialize Binance API

        Args:
            config: Binance section from the configuration file
            cache_manager: Optional cache manager (for ticker and exchange info)

        Config Example:
            {
                "enabled": true,
                "testnet": true,
                "endpoints": {
                    "testnet": {
                        "api_key": "${BINANCE_TESTNET_API_KEY}",
                        "secret_key": "${BINANCE_TESTNET_API_SECRET}"
                    },
                    "production": {
                        "api_key": "${BINANCE_API_KEY}",
                        "secret_key": "${BINANCE_API_SECRET}"
                    }
                },
                "rate_limit": {...},
                "retry": {...},
                "features": {
                    "spot_trading": true,
                    "futures_trading": true
                }
            }
        """
        super().__init__(config)

        self.cache_manager = cache_manager
        self.logger = LoggerEngine().get_logger(__name__)

        # Get credentials based on testnet flag
        endpoint_key = "testnet" if self.testnet else "production"
        endpoint_config = config.get("endpoints", {}).get(endpoint_key, {})

        self.api_key = endpoint_config.get("api_key")
        self.api_secret = endpoint_config.get("secret_key")

        # Initialize Binance client
        self.client = Client(self.api_key, self.api_secret, ping=False)

        # CRITICAL: python-binance testnet parameter doesn't work with sync Client
        # Manual API_URL override required
        if self.testnet:
            # Testnet FUTURES endpoint
            self.client.API_URL = 'https://testnet.binancefuture.com'
        else:
            # Production FUTURES endpoint
            self.client.API_URL = 'https://fapi.binance.com'

        # Features
        self.features = config.get("features", {})
        self.spot_enabled = self.features.get("spot_trading", True)
        self.futures_enabled = self.features.get("futures_trading", True)

        # Rate limiting config
        self.rate_limit = config.get("rate_limit", {})
        self.rate_limit_weight = 0
        self.rate_limit_reset_time = 0

        # Stats
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Server time offset (adjusted with sync_server_time)
        self._server_time_offset: timedelta = timedelta(0)
        self._server_time_synced: bool = False

        env = "testnet" if self.testnet else "production - public endpoints"
        self.logger.info(f"✅ BinanceAPI started ({env})")

    # ========================================================================
    # MARKET DATA - Implementation
    # ========================================================================

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
        """
        try:
            # Cache control
            cache_key = f"ticker:{symbol}"
            if self.cache_manager:
                cached = self.cache_manager.get(cache_key)
                if cached:
                    self.stats["cache_hits"] += 1
                    self.logger.debug(f"📦 Cache hit: {cache_key}")
                    return cached
                self.stats["cache_misses"] += 1

            # Make the synchronous call non-blocking using an asynchronous executor.
            loop = asyncio.get_event_loop()

            # Use the FUTURES API
            result = await loop.run_in_executor(
                None,
                lambda: self.client.futures_symbol_ticker(symbol=symbol)
            )

            self.stats["total_requests"] += 1

            # Futures API returns {"symbol": "BTCUSDT", "price": "45000.00", ...}
            # Convert to match expected format: {"lastPrice": "45000.00", ...}
            if 'price' in result and 'lastPrice' not in result:
                result['lastPrice'] = result['price']

            # Cache'e kaydet (5 saniye TTL)
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, ttl=5)

            return result

        except BinanceAPIException as e:
            self.logger.error(f"❌ Ticker API error ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise
        except Exception as e:
            self.logger.error(f"❌ Ticker error ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Order book al

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Depth (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Dict: Order book verisi
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.get_order_book(symbol=symbol, limit=limit)
            )
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
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of klines (max 1000)
            start_time: Start time (timestamp ms)
            end_time: End time (timestamp ms)

        Returns:
            List: Kline verisi
        """
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time

            loop = asyncio.get_event_loop()

            # Use futures_klines for Futures API (based on API_URL)
            if 'fapi' in self.client.API_URL or 'future' in self.client.API_URL.lower():
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.futures_klines(**params)
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_klines(**params)
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
            Dict: Account bilgisi
        """
        try:
            loop = asyncio.get_event_loop()

            # For FUTURES account
            result = await loop.run_in_executor(
                None,
                self.client.futures_account
            )

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
            symbol: Trading pair (e.g., "BTCUSDT")
            side: BUY or SELL
            order_type: LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, etc.
            quantity: Quantity
            price: Price (required for LIMIT order)
            **kwargs: Binance-specific parameters (timeInForce, stopPrice, etc.)

        Returns:
            Dict: Order result
        """
        try:
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity
            }

            # price and timeInForce are required for a LIMIT order
            if order_type.upper() == "LIMIT":
                if price is None:
                    raise ValueError("Price is required for LIMIT order")
                params["price"] = price
                params["timeInForce"] = kwargs.get("timeInForce", "GTC")

            # Additional parameters
            params.update(kwargs)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.create_order(**params)
            )

            self.stats["total_requests"] += 1

            self.logger.info(f"📝 Order created: {symbol} {side} {quantity}")
            return result

        except BinanceAPIException as e:
            self.logger.error(f"❌ Order API error ({symbol}): {e}")
            self.stats["total_errors"] += 1
            raise
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
            **kwargs: Binance-specific parameters

        Returns:
            Dict: Cancel result
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.cancel_order(
                    symbol=symbol,
                    orderId=int(order_id)
                )
            )

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
            symbol: Trading pair (None if all symbols)

        Returns:
            List[Dict]: List of open orders.
        """
        try:
            loop = asyncio.get_event_loop()

            if symbol:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_open_orders(symbol=symbol)
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    self.client.get_open_orders
                )

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
        Get Binance server time.

        Returns:
            Dict: {"serverTime": timestamp}
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.client.get_server_time
            )
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Server time error: {e}")
            self.stats["total_errors"] += 1
            raise

    async def sync_server_time(self) -> timedelta:
        """
        Synchronize with the Binance server time.
        Calculates the offset between the local time and the server time.

        Returns:
            timedelta: Server time offset (server - local)
        """
        try:
            result = await self.get_server_time()
            server_time_ms = result.get('serverTime')

            if server_time_ms:
                server_dt = datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc)
                local_dt = datetime.now(timezone.utc)
                self._server_time_offset = server_dt - local_dt
                self._server_time_synced = True

                offset_ms = self._server_time_offset.total_seconds() * 1000
                self.logger.info(f"⏰ Server time synced (offset: {offset_ms:+.0f}ms)")

            return self._server_time_offset

        except Exception as e:
            self._server_time_offset = timedelta(0)
            self._server_time_synced = False
            self.logger.warning(f"⚠️ Server time sync failed: {e}, using local UTC")
            return self._server_time_offset

    def get_synced_server_time(self) -> datetime:
        """
        Get the synchronized server time (UTC).
        Returns local UTC if sync_server_time has not been called.

        Returns:
            datetime: Server time (UTC)
        """
        return datetime.now(timezone.utc) + self._server_time_offset

    @property
    def server_time_offset(self) -> timedelta:
        """Server time offset (readonly)"""
        return self._server_time_offset

    @property
    def is_time_synced(self) -> bool:
        """Is the server time synchronized?"""
        return self._server_time_synced

    def health_check(self) -> bool:
        """
        API health check (testing with server time)

        Returns:
            bool: True if the API is running.
        """
        try:
            result = self.client.get_server_time()
            return "serverTime" in result
        except Exception as e:
            self.logger.error(f"❌ Health check error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        API istatistikleri al

        Returns:
            Dict: Statistics
        """
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (self.stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0

        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "testnet": self.testnet,
            "enabled": self.enabled,
            "rate_limit_weight": self.rate_limit_weight,
        }

    # ========================================================================
    # ADDITIONAL METHODS (Binance-specific)
    # ========================================================================

    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Exchange bilgisi al (FUTURES)

        Returns:
            Dict: Futures exchange info
        """
        try:
            # Cache control
            cache_key = "exchange_info"
            if self.cache_manager:
                cached = self.cache_manager.get(cache_key)
                if cached:
                    self.stats["cache_hits"] += 1
                    return cached
                self.stats["cache_misses"] += 1

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.client.futures_exchange_info
            )

            self.stats["total_requests"] += 1

            # Save to cache (1 hour TTL)
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, ttl=3600)

            return result
        except Exception as e:
            self.logger.error(f"❌ Exchange info error: {e}")
            self.stats["total_errors"] += 1
            raise

    async def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Get all tickers.

        Returns:
            List[Dict]: List of tickers.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.client.get_all_tickers
            )
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Error with all tickers: {e}")
            self.stats["total_errors"] += 1
            raise

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Symbol bilgisi al

        Args:
            symbol: Trading pair

        Returns:
            Dict: Symbol information or None
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.get_symbol_info(symbol=symbol)
            )
            self.stats["total_requests"] += 1
            return result
        except Exception as e:
            self.logger.error(f"❌ Symbol info error ({symbol}): {e}")
            return None

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    async def close(self) -> None:
        """
        Close the API connection.

        Since python-binance Client is synchronous, session management is internal.
        This method is a placeholder for clean shutdown.
        """
        try:
            if hasattr(self.client, 'close_connection'):
                self.client.close_connection()

            self.logger.info("🛑 BinanceAPI closed")
        except Exception as e:
            self.logger.error(f"❌ BinanceAPI close error: {e}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BinanceAPI',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    print("=" * 60)
    print("🧪 BinanceAPI Test")
    print("=" * 60)

    async def test():
        print("\n1️⃣  Config test:")

        # Test config (testnet)
        test_config = {
            "enabled": True,
            "testnet": True,
            "endpoints": {
                "testnet": {
                    "api_key": "test_key",
                    "secret_key": "test_secret"
                }
            },
            "rate_limit": {
                "max_requests_per_minute": 1200
            },
            "features": {
                "spot_trading": True,
                "futures_trading": True
            }
        }

        try:
            # Create BinanceAPI
            binance = BinanceAPI(config=test_config)
            print(f"   ✅ BinanceAPI created")
            print(f"   - Testnet: {binance.testnet}")
            print(f"   - API URL: {binance.client.API_URL}")
            print(f"   - Enabled: {binance.enabled}")
            print(f"   - SPOT: {binance.spot_enabled}")
            print(f"   - FUTURES: {binance.futures_enabled}")

            # Stats
            print("\n2️⃣  Stats:")
            stats = binance.get_stats()
            print(f"   - Total requests: {stats['total_requests']}")
            print(f"   - Total errors: {stats['total_errors']}")
            print(f"   - Cache hit rate: {stats['cache_hit_rate']}")

            # Health check (will fail without real credentials)
            print("\n3️⃣  Health check:")
            try:
                health = binance.health_check()
                print(f"   - Health: {health}")
                if health:
                    server_time = await binance.get_server_time()
                    dt = datetime.fromtimestamp(server_time['serverTime']/1000)
                    print(f"   - Server time: {dt}")
            except Exception as e:
                print(f"   ⚠️  Health check failed as expected (test credentials): {type(e).__name__}")

            print("\n   ✅ Test configuration successful")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    asyncio.run(test())

    print("\n✅ All tests completed!")
    print("=" * 60)
