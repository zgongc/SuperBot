#!/usr/bin/env python3
"""
engines/connector_engine.py
SuperBot - Connector Engine
Author: SuperBot Team
Date: 2025-10-16
Versiyon: 1.0.0

Connector Engine - Exchange API wrapper

Features:
- Binance REST API wrapper
- Rate limiting
- Error handling
- Retry mechanism
- Testnet/Production support
- Connection pooling integration

Usage:
    from engines.connector_engine import ConnectorEngine
    
    connector = ConnectorEngine(config={
        "testnet": True,
        "api_key": "xxx",
        "api_secret": "yyy"
    })
    
    # Market data
    ticker = await connector.get_ticker("BTCUSDT")
    
    # Account
    balance = await connector.get_balance()

Dependencies:
    - python-binance
    - aiohttp
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


class ConnectorEngine:
    """
    Connector Engine - Exchange API wrapper
    
    Binance API'yi wrap eder
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        connection_pool: Optional[Any] = None,
        cache_manager: Optional[Any] = None
    ):
        """
        Initialize ConnectorEngine
        
        Args:
            config: Connector configuration
            connection_pool: ConnectionPoolManager instance
            cache_manager: CacheManager instance
        """
        self.config = config or {}
        self.connection_pool = connection_pool
        self.cache_manager = cache_manager
        
        # Testnet/Production
        self.testnet = self.config.get("testnet", True)

        # API credentials - Read from endpoints config based on testnet flag
        endpoint_key = "testnet" if self.testnet else "production"
        endpoint_config = self.config.get("endpoints", {}).get(endpoint_key, {})

        self.api_key = endpoint_config.get("api_key")
        self.api_secret = endpoint_config.get("secret_key")

        # Fallback to old style env vars if not in config
        if not self.api_key or not self.api_secret:
            if self.testnet:
                self.api_key = os.getenv("BINANCE_TESTNET_API_KEY") or self.config.get("api_key")
                self.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET") or self.config.get("api_secret")
            else:
                self.api_key = os.getenv("BINANCE_API_KEY") or self.config.get("api_key")
                self.api_secret = os.getenv("BINANCE_API_SECRET") or self.config.get("api_secret")
        
        # Binance client
        self.client = None
        self._init_client()
        
        # Rate limiting
        self.rate_limit_weight = 0
        self.rate_limit_reset_time = 0
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        env = "testnet" if self.testnet else "production"
        logger.info(f"ConnectorEngine started ({env})")
    
    def _init_client(self):
        """Initialize the Binance client"""
        try:
            # Create client (base_endpoint parameter doesn't work properly, we'll override manually)
            self.client = Client(
                self.api_key,
                self.api_secret,
                ping=False  # Don't ping during init, we'll set URL first
            )

            # Manually override API_URL for testnet
            # NOTE: python-binance's testnet=True parameter doesn't work with sync Client
            # The library still uses production URL (https://api.binance.com/api)
            # Workaround: Manual API_URL override after client creation
            # AsyncClient.create(testnet=True) works correctly but requires full async refactoring
            if self.testnet:
                self.client.API_URL = 'https://testnet.binance.vision/api'

            logger.info("Binance client started")

        except Exception as e:
            logger.error(f"Binance client initialization error: {e}")
            raise
    
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Server time al
        
        Returns:
            Dict: {"serverTime": timestamp}
        """
        try:
            result = self.client.get_server_time()
            self.stats["total_requests"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Server time error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ticker bilgisi al
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            Dict: Ticker data
        """
        try:
            # Cache control
            cache_key = f"ticker:{symbol}"
            if self.cache_manager:
                cached = self.cache_manager.get(cache_key)
                if cached:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached
                self.stats["cache_misses"] += 1
            
            # API call
            result = self.client.get_ticker(symbol=symbol)
            self.stats["total_requests"] += 1
            
            # Cache'e kaydet (5 saniye TTL)
            if self.cache_manager:
                self.cache_manager.set(cache_key, result, ttl=5)
            
            return result
            
        except BinanceAPIException as e:
            logger.error(f"Ticker API error: {e}")
            self.stats["total_errors"] += 1
            raise
        except Exception as e:
            logger.error(f"Ticker error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[List]:
        """
        Kline/Candlestick data al
        
        Args:
            symbol: Trading pair
            interval: Interval (1m, 5m, 1h, etc.)
            limit: Number of klines
            
        Returns:
            List: Kline data
        """
        try:
            result = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            self.stats["total_requests"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Kline error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Order book al
        
        Args:
            symbol: Trading pair
            limit: Depth (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dict: Order book
        """
        try:
            result = self.client.get_order_book(
                symbol=symbol,
                limit=limit
            )
            self.stats["total_requests"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Order book error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def get_balance(self) -> Dict[str, Any]:
        """
        Account balance al
        
        Returns:
            Dict: Balance info
        """
        try:
            result = self.client.get_account()
            self.stats["total_requests"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Balance error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create an order.
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: LIMIT, MARKET, etc.
            quantity: Quantity
            price: Price (required for LIMIT)
            
        Returns:
            Dict: Order result
        """
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity
            }
            
            if order_type == "LIMIT" and price:
                params["price"] = price
                params["timeInForce"] = "GTC"
            
            result = self.client.create_order(**params)
            self.stats["total_requests"] += 1

            logger.info(f"Order created: {symbol} {side} {quantity}")
            return result

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Order iptal et
        
        Args:
            symbol: Trading pair
            order_id: Order ID
            
        Returns:
            Dict: Cancel result
        """
        try:
            result = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            self.stats["total_requests"] += 1

            logger.info(f"Order cancelled: {order_id}")
            return result

        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open orders.
        
        Args:
            symbol: Trading pair (None if all)
            
        Returns:
            List: Open orders
        """
        try:
            if symbol:
                result = self.client.get_open_orders(symbol=symbol)
            else:
                result = self.client.get_open_orders()
            
            self.stats["total_requests"] += 1
            return result

        except Exception as e:
            logger.error(f"Open orders error: {e}")
            self.stats["total_errors"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns connector statistics"""
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (self.stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "testnet": self.testnet,
            "rate_limit_weight": self.rate_limit_weight
        }
    
    def health_check(self) -> bool:
        """Connector health check"""
        try:
            # Test with server time
            result = self.client.get_server_time()
            return "serverTime" in result

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 60)
        print("ConnectorEngine Test")
        print("=" * 60)

        # Create connector
        connector = ConnectorEngine(config={
            "testnet": True
        })

        # Test 1: Server time
        print("\n1. Server time:")
        server_time = await connector.get_server_time()
        print(f"   {datetime.fromtimestamp(server_time['serverTime']/1000)}")

        # Test 2: Ticker
        print("\n2. Ticker (BTCUSDT):")
        ticker = await connector.get_ticker("BTCUSDT")
        print(f"   Price: {ticker.get('lastPrice', 'N/A')}")

        # Test 3: Stats
        print("\n3. Stats:")
        stats = connector.get_stats()
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Total Errors: {stats['total_errors']}")

        # Test 4: Health check
        print(f"\n4. Health Check: {'OK' if connector.health_check() else 'FAIL'}")

        print("\nTest completed!")
        print("=" * 60)
    
    asyncio.run(test())