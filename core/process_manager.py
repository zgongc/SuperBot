#!/usr/bin/env python3
"""
core/process_manager.py

SuperBot - Process Manager
Author: SuperBot Team
Date: 2025-11-12
Versiyon: 1.1.0

Central process manager that manages the module/engine lifecycle.

Features:
- Engine registration/deregistration
- Dependency-based startup order
- Automatic restart (crash recovery)
- Periyodik health check
- Async/sync callback support.

Usage:
    from core.process_manager import get_process_manager

    manager = get_process_manager()
    manager.register_engine(
        name="webui",
        start=start_webui,
        stop=stop_webui,
        health_check=webui_health,
        dependencies=["logger"],
    )
    manager.start_all()

Dependencies:
    - python>=3.12
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.config_engine import get_config
from core.logger_engine import get_logger


class EngineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    CRASHED = "crashed"
    RESTARTING = "restarting"


@dataclass
class EngineCallbacks:
    start: Optional[Callable[..., Any]] = None
    stop: Optional[Callable[..., Any]] = None
    health_check: Optional[Callable[..., bool]] = None


@dataclass
class EngineInfo:
    name: str
    callbacks: EngineCallbacks
    dependencies: List[str] = field(default_factory=list)
    auto_restart: bool = True
    state: EngineState = EngineState.STOPPED
    restart_count: int = 0
    max_restart_attempts: int = 3
    restart_delay: int = 5
    last_health_check: Optional[datetime] = None
    last_state_change: Optional[datetime] = None


class ProcessManager:
    """Engine lifecycle manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None) -> None:
        cfg_engine = get_config()
        daemon_cfg = cfg_engine.get("daemon", {})
        default_cfg = daemon_cfg.get("resource_allocation", {}).get("process", {})
        self.config = {**default_cfg, **(config or {})}

        self.logger = logger or get_logger("core.process_manager")
        self.health_check_interval = int(self.config.get("health_check_interval", 5))
        self.restart_delay = int(self.config.get("restart_delay", 5))

        self._lock = threading.RLock()
        self._engines: Dict[str, EngineInfo] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()

    # ------------------------------------------------------------------
    # Engine Registration Management
    # ------------------------------------------------------------------

    def register_engine(
        self,
        name: str,
        start: Optional[Callable[..., Any]] = None,
        stop: Optional[Callable[..., Any]] = None,
        health_check: Optional[Callable[..., bool]] = None,
        dependencies: Optional[List[str]] = None,
        auto_restart: bool = True,
        max_restart_attempts: Optional[int] = None,
        restart_delay: Optional[int] = None,
    ) -> None:
        """Engine kaydet."""
        with self._lock:
            if name in self._engines:
                self.logger.warning(f"⚠️ Engine is already registered: {name}")
                return

            info = EngineInfo(
                name=name,
                callbacks=EngineCallbacks(start=start, stop=stop, health_check=health_check),
                dependencies=dependencies or [],
                auto_restart=auto_restart,
                max_restart_attempts=max_restart_attempts or self.config.get("max_restart_attempts", 3),
                restart_delay=restart_delay or self.restart_delay,
            )
            info.last_state_change = datetime.now(timezone.utc)
            self._engines[name] = info
            self.logger.info(f"✅ Engine kaydedildi: {name}")

    def unregister_engine(self, name: str) -> None:
        with self._lock:
            info = self._engines.pop(name, None)
        if info:
            self.logger.info(f"🗑️ Engine record deleted: {name}")

    # ------------------------------------------------------------------
    # Engine Lifecycle Management
    # ------------------------------------------------------------------

    def start_engine(self, name: str) -> bool:
        info = self._get_engine(name)
        if not info:
            return False

        if info.state in {EngineState.RUNNING, EngineState.STARTING}:
            self.logger.debug(f"⚠️ Engine is already running: {name}")
            return True

        if not self._dependencies_ready(info):
            self.logger.warning(f"⚠️ Engine dependencies are not ready: {name}")
            return False

        info.state = EngineState.STARTING
        info.last_state_change = datetime.now(timezone.utc)
        self.logger.info(f"🚀 Engine is starting: {name}")
        try:
            self._execute_callback(info.callbacks.start, name)
            info.state = EngineState.RUNNING
            info.last_state_change = datetime.now(timezone.utc)
            info.restart_count = 0
            self.logger.info(f"✅ Engine is running: {name}")
            return True
        except Exception as exc:  # noqa: BLE001
            info.state = EngineState.CRASHED
            info.last_state_change = datetime.now(timezone.utc)
            self.logger.error(f"❌ Engine startup error {name}: {exc}")
            return False

    def stop_engine(self, name: str) -> bool:
        info = self._get_engine(name)
        if not info:
            return False

        if info.state in {EngineState.STOPPED, EngineState.STOPPING}:
            self.logger.debug(f"⚠️ Engine is already stopped: {name}")
            return True

        info.state = EngineState.STOPPING
        info.last_state_change = datetime.now(timezone.utc)
        self.logger.info(f"🛑 Engine durduruluyor: {name}")
        try:
            self._execute_callback(info.callbacks.stop, name)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"❌ Engine shutdown error {name}: {exc}")
        info.state = EngineState.STOPPED
        info.last_state_change = datetime.now(timezone.utc)
        self.logger.info(f"✅ Engine durdu: {name}")
        return True

    def restart_engine(self, name: str) -> bool:
        info = self._get_engine(name)
        if not info:
            return False

        info.state = EngineState.RESTARTING
        info.last_state_change = datetime.now(timezone.utc)
        self.logger.info(f"🔄 Engine is being restarted: {name}")
        self.stop_engine(name)
        time.sleep(info.restart_delay)
        info.restart_count += 1
        return self.start_engine(name)

    def start_all(self) -> None:
        for name in self._resolve_dependencies():
            self.start_engine(name)
        self.start_monitoring()

    def stop_all(self) -> None:
        self.stop_monitoring()
        for name in reversed(self._resolve_dependencies()):
            self.stop_engine(name)

    # ------------------------------------------------------------------
    # Monitoring & Health Check
    # ------------------------------------------------------------------

    def start_monitoring(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="process-monitor", daemon=True)
        self._monitor_thread.start()
        self.logger.info("🩺 Process monitoring started")

    def stop_monitoring(self) -> None:
        if not self._monitor_thread:
            return
        self._monitor_stop.set()
        self._monitor_thread.join(timeout=2)
        self._monitor_thread = None
        self.logger.info("🩺 Process monitoring durduruldu")

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.is_set():
            time.sleep(self.health_check_interval)
            for name, info in list(self._engines.items()):
                if info.state != EngineState.RUNNING or not info.callbacks.health_check:
                    continue
                try:
                    healthy = self._execute_callback(info.callbacks.health_check, name)
                    info.last_health_check = datetime.now(timezone.utc)
                    if not healthy:
                        self.logger.warning(f"⚠️ Health check failed: {name}")
                        if info.auto_restart and info.restart_count < info.max_restart_attempts:
                            self.restart_engine(name)
                        else:
                            info.state = EngineState.CRASHED
                    else:
                        self.logger.debug(f"🟢 Health check successful: {name}")
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"❌ Health check error {name}: {exc}")
                    if info.auto_restart and info.restart_count < info.max_restart_attempts:
                        self.restart_engine(name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execute_callback(self, callback: Optional[Callable[..., Any]], name: str) -> Any:
        if not callback:
            return None

        result = callback()
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    return asyncio.run_coroutine_threadsafe(result, loop).result()
            except RuntimeError:
                pass
            return asyncio.run(result)
        return result

    def _dependencies_ready(self, info: EngineInfo) -> bool:
        for dep in info.dependencies:
            dep_info = self._engines.get(dep)
            if not dep_info or dep_info.state != EngineState.RUNNING:
                return False
        return True

    def _resolve_dependencies(self) -> List[str]:
        order: List[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            engine = self._engines.get(name)
            if not engine:
                return
            for dep in engine.dependencies:
                visit(dep)
            order.append(name)

        for engine_name in list(self._engines.keys()):
            visit(engine_name)
        return order

    def _get_engine(self, name: str) -> Optional[EngineInfo]:
        info = self._engines.get(name)
        if not info:
            self.logger.error(f"❌ Engine not found: {name}")
        return info


_process_manager_instance: Optional[ProcessManager] = None
_process_manager_lock = threading.Lock()


def get_process_manager() -> ProcessManager:
    global _process_manager_instance
    with _process_manager_lock:
        if _process_manager_instance is None:
            _process_manager_instance = ProcessManager()
    return _process_manager_instance


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ProcessManager Test")
    print("=" * 60)

    manager = ProcessManager()

    def start_module():
        print("   -> module started")

    def stop_module():
        print("   -> module stopped")

    def health_check():
        return True

    manager.register_engine(
        name="test_engine",
        start=start_module,
        stop=stop_module,
        health_check=health_check,
    )

    manager.start_all()
    time.sleep(2)
    manager.stop_all()

    print("✅ Test completed!")
    print("=" * 60)


