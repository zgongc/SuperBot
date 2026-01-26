#!/usr/bin/env python3
"""
modules/trading/price_feed.py
SuperBot - Price Feed
Author: SuperBot Team
Date: 2025-12-03
Versiyon: 2.0.0

Cached price feed for real-time price access

V2 Changes:
- bookTicker subscription added (fast price for SL/TP)
- Bid/Ask spread tracking
- Best bid/ask prices

Features:
- Price caching (TTL: 5s)
- EventBus ticker subscription
- bookTicker subscription (best bid/ask)
- Batch price fetch
- WebSocket integration

Usage:
    feed = PriceFeed()
    price = feed.get_price("BTCUSDT")
    prices = feed.get_prices(["BTCUSDT", "ETHUSDT"])
    bid, ask = feed.get_bid_ask("BTCUSDT")

Dependencies:
    - python>=3.12
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add project root to path for direct execution
import sys
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from core.logger_engine import get_logger
from core.cache_manager import get_cache
from core.event_bus import get_event_bus


class PriceFeed:
    """
    Cached Price Feed

    Caches prices received from the WebSocket ticker.
    Optimized for real-time price access.

    SORUMLULUKLAR:
    ✅ Price caching (short TTL)
    ✅ EventBus ticker subscription
    ✅ EventBus bookTicker subscription (bid/ask)
    ✅ Batch price queries
    ✅ Price change detection
    ✅ Bid/Ask spread tracking

    WHAT IT DOES NOT DO:
    ❌ WebSocket connection management (WebSocketEngine yapar)
    ❌ Historical data (DataManager yapar)
    ❌ Indicator calculation (IndicatorManager yapar)
    """

    # Cache TTL (saniye)
    CACHE_TTL = 5  # 5 seconds - prices change quickly

    # Price change threshold (%)
    SIGNIFICANT_CHANGE_PCT = 0.1  # %0.1

    def __init__(
        self,
        cache_manager: Any = None,
        event_bus: Any = None,
        logger: Any = None
    ):
        """
        Args:
            cache_manager: CacheManager instance
            event_bus: EventBus instance
            logger: Logger instance
        """
        self.cache = cache_manager or get_cache()
        self.event_bus = event_bus or get_event_bus()
        self.logger = logger or get_logger("modules.trading.price_feed")

        # Stats
        self.stats = {
            "total_updates": 0,
            "ticker_updates": 0,
            "book_ticker_updates": 0,
            "candle_updates": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Last known prices (cache miss durumunda fallback)
        self._last_prices: Dict[str, float] = {}

        # Best bid/ask prices (bookTicker'dan)
        self._bid_prices: Dict[str, float] = {}
        self._ask_prices: Dict[str, float] = {}

        # EventBus subscription
        self._setup_subscriptions()

        self.logger.info("PriceFeed V2 created (supports bookTicker)")

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price (from cache).

        Args:
            symbol: Symbol name (e.g., "BTCUSDT")

        Returns:
            float: Current price or None (if the price is not available)
        """
        cache_key = f"price:{symbol}"

        # Cache'den al
        price = self.cache.get(cache_key)

        if price is not None:
            self.stats["cache_hits"] += 1
            return float(price)

        # Cache miss - fallback to last known
        self.stats["cache_misses"] += 1

        if symbol in self._last_prices:
            return self._last_prices[symbol]

        return None

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get batch prices.

        Args:
            symbols: List of symbols

        Returns:
            {symbol: price} - Symbols without a price are not included.
        """
        result = {}

        for symbol in symbols:
            price = self.get_price(symbol)
            if price is not None:
                result[symbol] = price

        return result

    def get_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the best bid and ask prices (from bookTicker).

        Important for SL/TP control - actual buy/sell prices.

        Args:
            symbol: Symbol name

        Returns:
            (bid_price, ask_price) - None if not available
        """
        bid = self._bid_prices.get(symbol)
        ask = self._ask_prices.get(symbol)
        return (bid, ask)

    def get_spread(self, symbol: str) -> Optional[Dict]:
        """
        Bid/Ask spread hesapla

        Args:
            symbol: Symbol name

        Returns:
            {
                "bid": 95000.0,
                "ask": 95001.0,
                "spread": 1.0,
                "spread_pct": 0.00105
            }
        """
        bid, ask = self.get_bid_ask(symbol)

        if bid is None or ask is None:
            return None

        spread = ask - bid
        spread_pct = (spread / bid) * 100 if bid > 0 else 0

        return {
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_pct": round(spread_pct, 5)
        }

    def set_price(self, symbol: str, price: float):
        """
        Manually set the price (for testing or overriding).

        Args:
            symbol: Symbol name
            price: Price
        """
        cache_key = f"price:{symbol}"
        self.cache.set(cache_key, price, ttl=self.CACHE_TTL)
        self._last_prices[symbol] = price
        self.stats["total_updates"] += 1

    def has_price(self, symbol: str) -> bool:
        """
        Is the price available?

        Args:
            symbol: Symbol name

        Returns:
            bool: True = price exists
        """
        return self.get_price(symbol) is not None

    def get_price_change(self, symbol: str, previous_price: float) -> Optional[Dict]:
        """
        Calculate the price change.

        Args:
            symbol: Symbol name
            previous_price: Previous price

        Returns:
            {
                "current": 95100.0,
                "previous": 95000.0,
                "change": 100.0,
                "change_pct": 0.105,
                "direction": "UP" | "DOWN" | "FLAT"
            }
        """
        current = self.get_price(symbol)

        if current is None or previous_price <= 0:
            return None

        change = current - previous_price
        change_pct = (change / previous_price) * 100

        if change_pct > self.SIGNIFICANT_CHANGE_PCT:
            direction = "UP"
        elif change_pct < -self.SIGNIFICANT_CHANGE_PCT:
            direction = "DOWN"
        else:
            direction = "FLAT"

        return {
            "current": current,
            "previous": previous_price,
            "change": change,
            "change_pct": round(change_pct, 4),
            "direction": direction
        }

    def get_stats(self) -> Dict[str, Any]:
        """PriceFeed istatistikleri"""
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (self.stats["cache_hits"] / total * 100) if total > 0 else 0

        return {
            **self.stats,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "tracked_symbols": len(self._last_prices),
            "bid_ask_symbols": len(self._bid_prices)
        }

    # ═══════════════════════════════════════════════════════════════════════
    # EVENTBUS SUBSCRIPTION
    # ═══════════════════════════════════════════════════════════════════════

    def _setup_subscriptions(self):
        """EventBus'a ticker subscription"""
        if not self.event_bus:
            self.logger.warning("EventBus not found, subscription skipped")
            return

        # Ticker events (comes from WebSocketEngine)
        # Pattern: ticker.BTCUSDT, ticker.ETHUSDT, ...
        self.event_bus.subscribe("ticker.*", self._on_ticker)

        # Kline/Candle events (includes price information)
        # Pattern: candle.BTCUSDT.5m, candle.ETHUSDT.1m, ...
        self.event_bus.subscribe("candle.*.*", self._on_candle)

        # Price events (direct price update)
        self.event_bus.subscribe("price.*", self._on_price)

        # NEW: bookTicker events (best bid/ask - critical for SL/TP)
        # Pattern: bookTicker.BTCUSDT, bookTicker.ETHUSDT, ...
        # Binance bookTicker: the fastest price update
        self.event_bus.subscribe("bookTicker.*", self._on_book_ticker)

        self.logger.debug("EventBus subscriptions active (ticker, candle, price, bookTicker)")

    def _on_ticker(self, event):
        """
        Ticker event handler

        Processes WebSocket ticker messages.

        Args:
            event: Event object from EventBus
                event.data = {
                    "symbol": "BTCUSDT",
                    "price": "95000.50" or "lastPrice": "95000.50",
                    ...
                }
        """
        try:
            # EventBus sends Event object, data is in event.data
            data = event.data if hasattr(event, 'data') else event

            symbol = data.get("symbol") or data.get("s")

            # Price field (different formats)
            price = (
                data.get("price") or
                data.get("lastPrice") or
                data.get("p") or
                data.get("c")  # Close price
            )

            if symbol and price:
                self._update_price(symbol, float(price), source="ticker")
                self.stats["ticker_updates"] += 1

        except Exception as e:
            self.logger.debug(f"Ticker parsing error: {e}")

    def _on_candle(self, event):
        """
        Candle event handler

        Extracts price from Kline/Candle messages.

        Args:
            event: Event object from EventBus
                event.data = {
                    "symbol": "BTCUSDT",
                    "close": 95000.50,
                    ...
                }
        """
        try:
            # EventBus sends Event object, data is in event.data
            data = event.data if hasattr(event, 'data') else event

            symbol = data.get("symbol") or data.get("s")

            # Close price
            price = data.get("close") or data.get("c")

            if symbol and price:
                self._update_price(symbol, float(price), source="candle")
                self.stats["candle_updates"] += 1

        except Exception as e:
            self.logger.debug(f"Candle parse error: {e}")

    def _on_price(self, event):
        """
        Direct price event handler

        Args:
            event: Event object from EventBus
                event.data = {"symbol": "BTCUSDT", "price": 95000.50}
        """
        try:
            # EventBus sends Event object, data is in event.data
            data = event.data if hasattr(event, 'data') else event

            symbol = data.get("symbol")
            price = data.get("price")

            if symbol and price:
                self._update_price(symbol, float(price), source="price")

        except Exception as e:
            self.logger.debug(f"Price parsing error: {e}")

    def _on_book_ticker(self, event):
        """
        bookTicker event handler (V2 - NEW)

        Binance bookTicker stream - the fastest price update
        Critical for SL/TP control - actual buy/sell prices

        Args:
            event: Event object from EventBus
                event.data = {
                    "s": "BTCUSDT",      # Symbol
                    "b": "95000.00",     # Best bid price
                    "B": "1.5",          # Best bid qty
                    "a": "95001.00",     # Best ask price
                    "A": "2.0"           # Best ask qty
                }
        """
        try:
            data = event.data if hasattr(event, 'data') else event

            symbol = data.get("symbol") or data.get("s")

            # Best bid/ask
            bid_price = data.get("bidPrice") or data.get("b")
            ask_price = data.get("askPrice") or data.get("a")

            if symbol:
                # Update bid/ask
                if bid_price:
                    self._bid_prices[symbol] = float(bid_price)
                if ask_price:
                    self._ask_prices[symbol] = float(ask_price)

                # Update the main price as the mid price
                if bid_price and ask_price:
                    mid_price = (float(bid_price) + float(ask_price)) / 2
                    self._update_price(symbol, mid_price, source="bookTicker")

                self.stats["book_ticker_updates"] += 1

        except Exception as e:
            self.logger.debug(f"bookTicker parse error: {e}")

    def _update_price(self, symbol: str, price: float, source: str = "unknown"):
        """
        Internal price update

        Args:
            symbol: Symbol name
            price: New price
            source: Kaynak (ticker, candle, bookTicker, etc.)
        """
        if price <= 0:
            return

        cache_key = f"price:{symbol}"

        # Cache'e yaz
        self.cache.set(cache_key, price, ttl=self.CACHE_TTL)

        # Write to the last prices (for fallback)
        self._last_prices[symbol] = price

        self.stats["total_updates"] += 1

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear the price cache.

        Args:
            symbol: Specific symbol (None = all)
        """
        if symbol:
            cache_key = f"price:{symbol}"
            self.cache.delete(cache_key)
            if symbol in self._last_prices:
                del self._last_prices[symbol]
            if symbol in self._bid_prices:
                del self._bid_prices[symbol]
            if symbol in self._ask_prices:
                del self._ask_prices[symbol]
            self.logger.debug(f"{symbol} price cache cleaned")
        else:
            self._last_prices.clear()
            self._bid_prices.clear()
            self._ask_prices.clear()
            self.logger.debug("All price cache has been cleared")

    def get_all_prices(self) -> Dict[str, float]:
        """
        Returns all known prices.

        Returns:
            {symbol: price}
        """
        return dict(self._last_prices)


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PriceFeed V2 Test")
    print("=" * 60)

    print("\n1. Creating a PriceFeed:")
    feed = PriceFeed()
    print("   PriceFeed V2 created")

    print("\n2. Manuel price set:")
    feed.set_price("BTCUSDT", 95000.0)
    feed.set_price("ETHUSDT", 3200.0)
    feed.set_price("BNBUSDT", 620.0)

    price = feed.get_price("BTCUSDT")
    print(f"   BTCUSDT: ${price:,.2f}")

    print("\n3. Batch prices:")
    prices = feed.get_prices(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    for s, p in prices.items():
        print(f"   {s}: ${p:,.2f}")
    print(f"   SOLUSDT: (none - price not set)")

    print("\n4. Price change:")
    change = feed.get_price_change("BTCUSDT", 94500.0)
    if change:
        print(f"   Direction: {change['direction']}")
        print(f"   Change: ${change['change']:+,.2f} ({change['change_pct']:+.2f}%)")

    print("\n5. Bid/Ask simulation:")
    # Simulate bookTicker data
    feed._bid_prices["BTCUSDT"] = 94999.0
    feed._ask_prices["BTCUSDT"] = 95001.0

    bid, ask = feed.get_bid_ask("BTCUSDT")
    print(f"   Best Bid: ${bid:,.2f}")
    print(f"   Best Ask: ${ask:,.2f}")

    spread = feed.get_spread("BTCUSDT")
    if spread:
        print(f"   Spread: ${spread['spread']:.2f} ({spread['spread_pct']:.5f}%)")

    print("\n6. Stats:")
    stats = feed.get_stats()
    print(f"   Cache hit rate: {stats['cache_hit_rate']}")
    print(f"   Total updates: {stats['total_updates']}")
    print(f"   Tracked symbols: {stats['tracked_symbols']}")
    print(f"   Bid/Ask symbols: {stats['bid_ask_symbols']}")

    print("\n7. Has price check:")
    print(f"   BTCUSDT: {feed.has_price('BTCUSDT')}")
    print(f"   XYZUSDT: {feed.has_price('XYZUSDT')}")

    print("\nTest completed!")
    print("=" * 60)
