#!/usr/bin/env python3

"""
core/cache_manager.py

SuperBot - Cache Manager
Author: SuperBot Team
Date: 2025-11-12
Version: 1.0.0

Adaptive cache management (Memory / Redis) with TTL, LRU and statistics support.

Features:
- Memory cache (LRU eviction, TTL)
- Redis cache (optional)
- Cache warming
- Health check and statistics collection

Usage:
    from core.cache_manager import CacheManager
    cache = CacheManager()
    cache.set("price:BTCUSDT", 50000, ttl=60)
    value = cache.get("price:BTCUSDT")

Dependencies:
    - redis (optional)
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

# Cases when not running as package (python core/cache_manager.py)
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.config_engine import get_config
from core.logger_engine import get_logger

logger = get_logger("core.cache_manager")


@dataclass
class CacheEntry:
    """Memory cache entry structure."""

    key: str
    value: Any
    ttl: Optional[float]
    created_at: float
    accessed_count: int = 0

    def is_expired(self) -> bool:
        """Has entry expired?"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class MemoryCache:
    """
    Memory-based cache implementation.
    """

    def __init__(self, max_size: int = 1000, eviction_policy: str = "lru", compress: bool = False) -> None:
        self.max_size = max_size
        self.eviction_policy = eviction_policy.lower()
        self.compress = compress
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        entry = self.cache[key]

        if entry.is_expired():
            self.delete(key)
            self.stats["expired"] += 1
            self.stats["misses"] += 1
            return None

        # Update access order only for LRU
        if self.eviction_policy == "lru":
            self.cache.move_to_end(key)
            
        entry.accessed_count += 1
        self.stats["hits"] += 1
        
        value = entry.value
        if self.compress and isinstance(value, bytes):
            try:
                import zlib
                import pickle
                value = pickle.loads(zlib.decompress(value))
            except Exception:
                pass
                
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if self.compress:
             import zlib
             import pickle
             value = zlib.compress(pickle.dumps(value))

        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict based on policy
            last = False if self.eviction_policy == "fifo" else False 
            # OrderedDict: popitem(last=False) -> FIFO (pop first inserted)
            # OrderedDict: popitem(last=False) -> LRU (pop least recently used/inserted) if move_to_end used
            # For LRU: The oldest accessed item is at the beginning (if we use move_to_end on access)
            # For FIFO: The oldest inserted item is at the beginning (if we DONT use move_to_end on access)
            
            # Since we manage move_to_end logic in get(), popitem(last=False) works for both LRU (oldest used) and FIFO (oldest inserted)
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1

        entry = CacheEntry(key=key, value=value, ttl=ttl, created_at=time.time())

        if key in self.cache:
            # For LRU, update position. For FIFO, keep original insertion position? 
            # Usually FIFO doesn't update position on update, but let's stick to standard map behavior which is update in place.
            # If we delete and re-insert, it becomes "new". 
            if self.eviction_policy == "lru":
                self.cache.move_to_end(key)
            # If FIFO, we don't move it to end, keeping its original "insertion time" relative to others?
            # Or does 'set' count as a new insertion? Typically yes.
            elif self.eviction_policy == "fifo":
                 self.cache.move_to_end(key) 

        self.cache[key] = entry

    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0.0
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2f}%",
            "current_size": len(self.cache),
            "max_size": self.max_size,
        }


class RedisCache:
    """
    Redis-based cache implementation.
    """

    def __init__(self, host: str, port: int, db: int, password: Optional[str] = None, 
                 pool_size: int = 10, socket_timeout: int = 5, compress: bool = False) -> None:
        try:
            import redis

            pool = redis.ConnectionPool(
                host=host, 
                port=port, 
                db=db, 
                password=password, 
                max_connections=pool_size,
                decode_responses=not compress # If compressing, we need bytes
            )
            
            self.compress = compress
            self.redis = redis.Redis(
                connection_pool=pool,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout
            )
            self.redis.ping()
            logger.info(f"✅ Redis connection successful: {host}:{port}")
        except ImportError as exc:
            raise ImportError("Redis package missing: pip install redis") from exc
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(f"Redis connection error: {exc}") from exc

        self.stats = {"hits": 0, "misses": 0}

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.redis.get(key)
            if value is None:
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            
            if self.compress:
                import zlib
                import pickle
                return pickle.loads(zlib.decompress(value))
            
            return json.loads(value)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis get error: {exc}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            if self.compress:
                import zlib
                import pickle
                value_data = zlib.compress(pickle.dumps(value))
            else:
                value_data = json.dumps(value)
                
            if ttl:
                self.redis.setex(key, int(ttl), value_data)
            else:
                self.redis.set(key, value_data)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis set error: {exc}")

    def delete(self, key: str) -> bool:
        try:
            return bool(self.redis.delete(key))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis delete error: {exc}")
            return False

    def clear(self) -> None:
        try:
            self.redis.flushdb()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis clear error: {exc}")

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0.0
        try:
            info = self.redis.info("stats")
            return {
                **self.stats,
                "total_requests": total,
                "hit_rate": f"{hit_rate:.2f}%",
                "redis_hits": info.get("keyspace_hits", 0),
                "redis_misses": info.get("keyspace_misses", 0),
            }
        except Exception:  # noqa: BLE001
            return {**self.stats, "total_requests": total, "hit_rate": f"{hit_rate:.2f}%"}


class CacheManager:
    """
    Cache Manager - adaptive cache management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config_engine = get_config()
            cache_cfg = config_engine.get("cache", {})
        else:
            cache_cfg = config

        self.backend_type = cache_cfg.get("backend", "memory")

        compress = cache_cfg.get("compress", False)

        if self.backend_type == "redis":
            redis_cfg = cache_cfg.get("redis", {})
            try:
                self.backend = RedisCache(
                    host=redis_cfg.get("host", "localhost"),
                    port=int(redis_cfg.get("port", 6379)),
                    db=int(redis_cfg.get("db", 0)),
                    password=redis_cfg.get("password"),
                    pool_size=int(redis_cfg.get("pool_size", 10)),
                    socket_timeout=int(redis_cfg.get("socket_timeout", 5)),
                    compress=compress
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️  Redis initialization error, Memory backend will be used: {exc}")
                self.backend_type = "memory"
                memory_cfg = cache_cfg.get("memory", {})
                max_size = int(memory_cfg.get("max_size", cache_cfg.get("max_size", 1000)))
                eviction_policy = memory_cfg.get("eviction_policy", "lru")
                self.backend = MemoryCache(max_size=max_size, eviction_policy=eviction_policy, compress=compress)
        else:
            memory_cfg = cache_cfg.get("memory", {})
            max_size = int(memory_cfg.get("max_size", cache_cfg.get("max_size", 1000)))
            eviction_policy = memory_cfg.get("eviction_policy", "lru")
            self.backend = MemoryCache(max_size=max_size, eviction_policy=eviction_policy, compress=compress)
            logger.info(f"✅ CacheManager initialized with Memory backend (Policy: {eviction_policy}, Compress: {compress})")

        self.default_ttl = cache_cfg.get("default_ttl") or cache_cfg.get("ttl_default")
        self.warming_keys = cache_cfg.get("warming_keys", [])
        self.enable_warming = cache_cfg.get("enable_warming", False)

        if self.enable_warming and self.warming_keys:
            self.auto_warm()

    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        self.backend.set(key, value, ttl)

    def exists(self, key: str) -> bool:
        try:
            if self.backend_type == "redis":
                return bool(self.backend.redis.exists(key))  # type: ignore[attr-defined]
            return key in self.backend.cache  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Cache exists check error: {exc}")
            return False

    def delete(self, key: str) -> bool:
        return self.backend.delete(key)

    def clear(self) -> None:
        self.backend.clear()
        logger.info("🗑️  Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        stats = self.backend.get_stats()
        stats["backend"] = self.backend_type
        return stats

    def health_check(self) -> bool:
        try:
            test_key = "__cache_health__"
            self.set(test_key, "ok", ttl=5)
            result = self.get(test_key)
            self.delete(test_key)
            return result == "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Cache health check failed: {exc}")
            return False

    def warm_cache(self, keys: Optional[List[str]] = None) -> Dict[str, bool]:
        keys = keys or self.warming_keys
        if not keys:
            logger.info("⚠️  No keys found for cache warming")
            return {}

        logger.info(f"🔥 Starting cache warming: {len(keys)} keys")
        results: Dict[str, bool] = {}

        for key in keys:
            try:
                exists = self.exists(key)
                results[key] = exists
                if exists:
                    logger.debug(f"✅ Warmed: {key}")
                else:
                    logger.debug(f"⏭️  Skipped: {key}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"❌ Warming error ({key}): {exc}")
                results[key] = False

        success_count = sum(1 for val in results.values() if val)
        logger.info(f"✅ Cache warming completed: {success_count}/{len(keys)} successful")
        return results

    def auto_warm(self) -> None:
        if not self.enable_warming:
            return
        self.warm_cache(self.warming_keys)


# ============================================================================
# SINGLETON & HELPER FUNCTIONS
# ============================================================================


_cache_manager_instance: Optional[CacheManager] = None
_cache_lock = Lock()


def get_cache_manager() -> CacheManager:
    """
    Return CacheManager singleton instance.

    Returns:
        CacheManager: Singleton instance
    """

    global _cache_manager_instance
    if _cache_manager_instance is None:
        with _cache_lock:
            if _cache_manager_instance is None:
                _cache_manager_instance = CacheManager()
    return _cache_manager_instance


def get_cache() -> CacheManager:
    """
    Return CacheManager (backward compatibility).

    Returns:
        CacheManager: Cache manager instance
    """

    return get_cache_manager()


# ============================================================================
# TEST
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 CacheManager Test")
    print("=" * 60)

    cache = CacheManager(config={"backend": "memory", "max_size": 3})
    cache.set("price:BTC", 50000, ttl=1)
    cache.set("price:ETH", 1700)
    cache.set("price:SOL", 20)

    print(f"BTC Price: {cache.get('price:BTC')}")
    time.sleep(1.2)
    print(f"BTC Price (expired): {cache.get('price:BTC')}")

    cache.set("price:XRP", 0.5)
    print(f"ETH Price: {cache.get('price:ETH')}")

    print(f"Stats: {cache.get_stats()}")
    print(f"Health: {'✅' if cache.health_check() else '❌'}")

    print("=" * 60)

