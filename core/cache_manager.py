#!/usr/bin/env python3

"""
core/cache_manager.py

SuperBot - Cache Manager
Yazar: SuperBot Team
Tarih: 2025-11-12
Versiyon: 1.0.0

Adaptif cache yönetimi (Memory / Redis) ile TTL, LRU ve istatistik desteği.

Özellikler:
- Memory cache (LRU eviction, TTL)
- Redis cache (opsiyonel)
- Cache warming
- Health check ve istatistik toplama

Kullanım:
    from core.cache_manager import CacheManager
    cache = CacheManager()
    cache.set("price:BTCUSDT", 50000, ttl=60)
    value = cache.get("price:BTCUSDT")

Bağımlılıklar:
    - redis (opsiyonel)
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

# Paket olarak çalışmadığı durumlar (python core/cache_manager.py)
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
    """Memory cache entry yapısı."""

    key: str
    value: Any
    ttl: Optional[float]
    created_at: float
    accessed_count: int = 0

    def is_expired(self) -> bool:
        """Entry süresi doldu mu?"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class MemoryCache:
    """
    Memory tabanlı cache implementasyonu (LRU).
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
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

        self.cache.move_to_end(key)
        entry.accessed_count += 1
        self.stats["hits"] += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
            self.stats["evictions"] += 1

        entry = CacheEntry(key=key, value=value, ttl=ttl, created_at=time.time())

        if key in self.cache:
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
    Redis tabanlı cache implementasyonu.
    """

    def __init__(self, host: str, port: int, db: int, password: Optional[str] = None) -> None:
        try:
            import redis

            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            self.redis.ping()
            logger.info(f"✅ Redis bağlantısı başarılı: {host}:{port}")
        except ImportError as exc:
            raise ImportError("Redis package eksik: pip install redis") from exc
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(f"Redis bağlantı hatası: {exc}") from exc

        self.stats = {"hits": 0, "misses": 0}

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.redis.get(key)
            if value is None:
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            return json.loads(value)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis get hatası: {exc}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            value_str = json.dumps(value)
            if ttl:
                self.redis.setex(key, int(ttl), value_str)
            else:
                self.redis.set(key, value_str)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis set hatası: {exc}")

    def delete(self, key: str) -> bool:
        try:
            return bool(self.redis.delete(key))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis delete hatası: {exc}")
            return False

    def clear(self) -> None:
        try:
            self.redis.flushdb()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Redis clear hatası: {exc}")

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
    Cache Manager - adaptif cache yönetimi.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config_engine = get_config()
            cache_cfg = config_engine.get("cache", {})
        else:
            cache_cfg = config

        self.backend_type = cache_cfg.get("backend", "memory")

        if self.backend_type == "redis":
            redis_cfg = cache_cfg.get("redis", {})
            try:
                self.backend = RedisCache(
                    host=redis_cfg.get("host", "localhost"),
                    port=int(redis_cfg.get("port", 6379)),
                    db=int(redis_cfg.get("db", 0)),
                    password=redis_cfg.get("password"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️  Redis başlatma hatası, Memory backend kullanılacak: {exc}")
                self.backend_type = "memory"
                memory_cfg = cache_cfg.get("memory", {})
                max_size = int(memory_cfg.get("max_size", cache_cfg.get("max_size", 1000)))
                self.backend = MemoryCache(max_size=max_size)
        else:
            memory_cfg = cache_cfg.get("memory", {})
            max_size = int(memory_cfg.get("max_size", cache_cfg.get("max_size", 1000)))
            self.backend = MemoryCache(max_size=max_size)
            logger.info("✅ CacheManager Memory backend ile başlatıldı")

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
            logger.error(f"❌ Cache exists kontrol hatası: {exc}")
            return False

    def delete(self, key: str) -> bool:
        return self.backend.delete(key)

    def clear(self) -> None:
        self.backend.clear()
        logger.info("🗑️  Cache temizlendi")

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
            logger.error(f"❌ Cache health check başarısız: {exc}")
            return False

    def warm_cache(self, keys: Optional[List[str]] = None) -> Dict[str, bool]:
        keys = keys or self.warming_keys
        if not keys:
            logger.info("⚠️  Cache warming için key bulunamadı")
            return {}

        logger.info(f"🔥 Cache warming başlatılıyor: {len(keys)} key")
        results: Dict[str, bool] = {}

        for key in keys:
            try:
                exists = self.exists(key)
                results[key] = exists
                if exists:
                    logger.debug(f"✅ Warmed: {key}")
                else:
                    logger.debug(f"⏭️  Atlandı: {key}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"❌ Warming hatası ({key}): {exc}")
                results[key] = False

        success_count = sum(1 for val in results.values() if val)
        logger.info(f"✅ Cache warming tamamlandı: {success_count}/{len(keys)} başarılı")
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
    CacheManager singleton instance'ını döndür.

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
    Geriye CacheManager döndür (backward compatibility).

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

