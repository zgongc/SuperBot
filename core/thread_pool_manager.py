#!/usr/bin/env python3
"""
core/thread_pool_manager.py

SuperBot - Thread Pool Manager
Author: SuperBot Team
Date: 2025-11-12
Versiyon: 1.1.0

Module-based thread pool management. Creates a separate pool for each module.
priority, number of jobs, and metrics are tracked.

Features:
- Module-based thread pool management.
- Dynamic pool size adjustment.
- Statistics for running/failed jobs.
- Bulk shutdown for graceful shutdown.

Usage:
    from core.thread_pool_manager import get_thread_pool_manager

    manager = get_thread_pool_manager()
    future = manager.submit("trading", my_function, *args, **kwargs)

Dependencies:
    - python>=3.12
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.config_engine import get_config
from core.logger_engine import get_logger


@dataclass
class PoolStats:
    module: str
    max_workers: int
    priority: str = "normal"
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_resize: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ThreadPoolManager:
    """Module-based thread pool manager."""

    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None, logger=None):
        self.logger = logger or get_logger("core.thread_pool_manager")
        cfg_engine = get_config()
        default_cfg = cfg_engine.get("daemon.resource_allocation.thread_pools", {})
        infrastructure_cfg = cfg_engine.get("infrastructure.thread_pool", {})

        if config is not None:
            self.pool_config = config
        elif default_cfg:
            self.pool_config = default_cfg
        elif infrastructure_cfg:
            self.pool_config = infrastructure_cfg
        else:
            self.pool_config = {
                "default": {
                    "worker_threads": 4,
                    "priority": "normal",
                }
            }

        self._lock = threading.RLock()
        self._pools: Dict[str, ThreadPoolExecutor] = {}
        self._stats: Dict[str, PoolStats] = {}

        self._initialize_pools()

    def _initialize_pools(self) -> None:
        with self._lock:
            for module, cfg in self.pool_config.items():
                workers = int(cfg.get("worker_threads", cfg.get("max_workers", 2)))
                priority = cfg.get("priority", "normal")
                self._create_pool(module, workers, priority)

    def _create_pool(self, module: str, workers: int, priority: str) -> None:
        pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"{module}_pool",
        )
        self._pools[module] = pool
        self._stats[module] = PoolStats(
            module=module,
            max_workers=workers,
            priority=priority,
        )
        self.logger.info(
            f"✅ Thread pool created: module={module}, workers={workers}, priority={priority}"
        )

    def register_pool(
        self,
        module: str,
        worker_threads: int,
        priority: str = "normal",
    ) -> None:
        """Dynamically register a new pool."""
        with self._lock:
            if module in self._pools:
                self.logger.warning(f"⚠️ Thread pool is already registered: {module}")
                return
            self._create_pool(module, worker_threads, priority)

    def submit(
        self,
        module: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        """Sends a task to the pool of the specified module."""
        with self._lock:
            if module not in self._pools:
                default_workers = int(self.pool_config.get("default", {}).get("worker_threads", 2))
                self.logger.warning(
                    f"⚠️ Pool not found for '{module}', creating with default settings"
                )
                self._create_pool(module, default_workers, priority="normal")

            stats = self._stats[module]
            stats.active_tasks += 1
            pool = self._pools[module]

        future = pool.submit(fn, *args, **kwargs)

        def _on_done(fut: Future[Any]) -> None:
            with self._lock:
                stats.active_tasks -= 1
                if fut.cancelled():
                    self.logger.warning(f"⚠️ Task canceled: module={module}")
                elif fut.exception():
                    stats.failed_tasks += 1
                    self.logger.error(
                        f"❌ Task failed (module={module}): {fut.exception()}"
                    )
                else:
                    stats.completed_tasks += 1

        future.add_done_callback(_on_done)
        return future

    def resize_pool(self, module: str, new_size: int) -> None:
        """Resizes the pool."""
        with self._lock:
            if module not in self._pools:
                raise ValueError(f"Pool not found: {module}")

            old_pool = self._pools[module]
            old_pool.shutdown(wait=True, cancel_futures=False)

            self._pools[module] = ThreadPoolExecutor(
                max_workers=new_size,
                thread_name_prefix=f"{module}_pool",
            )
            stats = self._stats[module]
            stats.max_workers = new_size
            stats.last_resize = datetime.now(timezone.utc)
            self.logger.info(f"🔧 Thread pool resized: module={module}, workers={new_size}")

    def shutdown_pool(self, module: str, wait: bool = True) -> None:
        """Belirli bir pool'u kapat."""
        with self._lock:
            pool = self._pools.pop(module, None)
            stats = self._stats.pop(module, None)

        if not pool:
            self.logger.warning(f"⚠️ Thread pool not found: {module}")
            return

        self.logger.info(f"🛑 Thread pool is being shut down: module={module}")
        pool.shutdown(wait=wait, cancel_futures=True)
        if stats:
            self.logger.info(
                f"✅ Thread pool closed: module={module}, completed={stats.completed_tasks}, failed={stats.failed_tasks}"
            )

    def shutdown_all(self, wait: bool = True) -> None:
        """Closes all thread pools."""
        with self._lock:
            modules = list(self._pools.keys())
        for module in modules:
            self.shutdown_pool(module, wait=wait)

    def get_pool_stats(self, module: str) -> Optional[PoolStats]:
        """Returns the statistics of a specific pool."""
        with self._lock:
            stats = self._stats.get(module)
            if stats:
                return PoolStats(**stats.__dict__)
            return None

    def get_all_stats(self) -> Dict[str, PoolStats]:
        """Returns the statistics of all pools."""
        with self._lock:
            return {module: PoolStats(**stats.__dict__) for module, stats in self._stats.items()}


_thread_pool_manager_instance: Optional[ThreadPoolManager] = None
_thread_pool_manager_lock = threading.Lock()


def get_thread_pool_manager() -> ThreadPoolManager:
    """Returns the ThreadPoolManager singleton instance."""
    global _thread_pool_manager_instance
    with _thread_pool_manager_lock:
        if _thread_pool_manager_instance is None:
            _thread_pool_manager_instance = ThreadPoolManager()
    return _thread_pool_manager_instance


# ============================================================================
# TEST
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ThreadPoolManager Test")
    print("=" * 60)

    manager = ThreadPoolManager(
        config={
            "trading": {"worker_threads": 2, "priority": "high"},
            "backtest": {"worker_threads": 3, "priority": "normal"},
        }
    )

    def add(a: int, b: int) -> int:
        return a + b

    future1 = manager.submit("trading", add, 1, 2)
    future2 = manager.submit("backtest", add, 10, 20)
    print(f"Trading result: {future1.result(timeout=2)}")
    print(f"Backtest result: {future2.result(timeout=2)}")

    stats = manager.get_all_stats()
    for module, info in stats.items():
        print(f"{module}: completed={info.completed_tasks}, failed={info.failed_tasks}, active={info.active_tasks}")

    manager.shutdown_all()
    print("✅ Test completed!")
    print("=" * 60)

