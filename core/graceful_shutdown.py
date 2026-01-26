#!/usr/bin/env python3
"""
core/graceful_shutdown.py
SuperBot - Safe Shutdown System
Author: SuperBot Team
Date: 2025-10-16
Versiyon: 1.0.0

Features:
- Signal handling (SIGTERM, SIGINT, SIGBREAK)
- Ordered shutdown (based on dependency order)
- Timeout mechanism (force shutdown if stuck)
- State persistence (save the current state)
- Open position safety (warning for open positions)
- Buffer flush (empty all buffers)
- EventBus entegrasyonu

Usage:
    from core.graceful_shutdown import GracefulShutdown
    
    # Initialize
    shutdown = GracefulShutdown(
        process_manager=process_manager,
        event_bus=event_bus,
        logger_engine=logger_engine
    )
    
    # Register signal handlers
    shutdown.register_handlers()
    
    # Manuel shutdown
    shutdown.initiate(reason="User requested")

Dependencies:
    - core.process_manager
    - core.event_bus
    - core.logger_engine
"""

import signal
import sys
import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


@dataclass
class ShutdownConfig:
    """Shutdown configuration"""
    timeout: int = 30  # Maximum waiting time (seconds)
    force_after_timeout: bool = True  # Force close after timeout
    save_state: bool = True  # Durumu kaydet
    close_positions: bool = False  # Close open positions (default: warn)
    flush_buffers: bool = True  # Flush the buffers
    wait_for_pending_tasks: bool = True  # Complete pending tasks
    wait_for_tasks: bool = True  # Task'leri bekle (default: True)
    task_timeout: int = 30  # Task timeout (saniye)
    state_dir: Path = Path("state")  # State record directory


class GracefulShutdown:
    """
    Safe shutdown system - Async Version for Daemon

    It catches signals and gracefully shuts down the system.
    It prevents data loss and incomplete operations.
    """

    def __init__(
        self,
        logger,  # Logger instance (comes from daemon)
        config: Optional[Dict] = None  # Config dict from daemon.yaml
    ):
        """
        Start GracefulShutdown (Daemon compatible)

        Args:
            logger: Logger instance
            config: Shutdown configuration (dict)
        """
        self.logger = logger

        # Create a ShutdownConfig from the config.
        if config:
            self.config = ShutdownConfig(
                timeout=config.get('timeout', 30),
                force_after_timeout=config.get('force_after_timeout', True),
                save_state=config.get('save_state', True),
                close_positions=config.get('close_positions', False),
                flush_buffers=config.get('flush_buffers', True),
                wait_for_pending_tasks=config.get('wait_for_pending_tasks', True)
            )
        else:
            self.config = ShutdownConfig()

        # Shutdown state
        self.is_shutting_down = False
        self.shutdown_initiated_at: Optional[datetime] = None
        self.shutdown_reason: Optional[str] = None

        # Callbacks
        self.pre_shutdown_callbacks: list[Callable] = []
        self.post_shutdown_callbacks: list[Callable] = []

        self.logger.info("GracefulShutdown is ready")
    
    def register_handlers(self):
        """
        Register signal handlers.
        
        SIGTERM, SIGINT (Ctrl+C), SIGBREAK (Windows) are caught.
        """
        # Unix/Linux/Mac signals
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Windows signal
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._signal_handler)
        
        logger.info("Signal handler'lar kaydedildi")
    
    def _signal_handler(self, signum: int, frame):
        """
        Signal handler callback
        
        Args:
            signum: Signal number
            frame: Stack frame
        """
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT"
        }
        
        if sys.platform == "win32":
            signal_names[signal.SIGBREAK] = "SIGBREAK"
        
        signal_name = signal_names.get(signum, f"Signal {signum}")
        logger.warning(f"🛑 {signal_name} caught, safe shutdown is starting...")
        
        self.initiate(reason=f"{signal_name} received")
    
    def add_pre_shutdown_callback(self, callback: Callable):
        """
        Add a callback function to be executed before shutdown.
        
        Args:
            callback: Callback fonksiyon
        """
        self.pre_shutdown_callbacks.append(callback)
        logger.debug(f"Pre-shutdown callback eklendi: {callback.__name__}")
    
    def add_post_shutdown_callback(self, callback: Callable):
        """
        Add a callback function to be executed after shutdown.
        
        Args:
            callback: Callback fonksiyon
        """
        self.post_shutdown_callbacks.append(callback)
        logger.debug(f"Post-shutdown callback eklendi: {callback.__name__}")
    
    async def initiate(self, reason: str = "Unknown"):
        """
        Initiate safe shutdown (Async version)

        Args:
            reason: Kapanma sebebi
        """
        if self.is_shutting_down:
            self.logger.warning("Shutdown is already in progress")
            return

        self.is_shutting_down = True
        self.shutdown_initiated_at = datetime.now()
        self.shutdown_reason = reason

        self.logger.info(f"🛑 Safe shutdown started. Reason: {reason}")

        # Shutdown sequence (async)
        try:
            await self._execute_shutdown()
        except Exception as e:
            self.logger.critical(f"Critical error during shutdown: {e}")
            self._force_shutdown()
    
    async def _execute_shutdown(self):
        """Run the shutdown sequence (Async version - minimal for daemon)"""

        self.logger.info("=" * 60)
        self.logger.info("🛑 SHUTDOWN SEQUENCE STARTED")
        self.logger.info("=" * 60)

        # 1. Pre-shutdown callbacks
        self.logger.info("1️⃣ Pre-shutdown callbacks are being executed...")
        self._run_callbacks(self.pre_shutdown_callbacks)

        # 2. Post-shutdown callbacks
        self.logger.info("2️⃣ Post-shutdown callbacks are being executed...")
        self._run_callbacks(self.post_shutdown_callbacks)

        # 3. Final log
        self.logger.info("✅ Safe shutdown completed")
        self.logger.info("=" * 60)

        # NOTE: The stop() method in the daemon already performs all cleanup.
        # That's why we only run callbacks here
    
    def _run_callbacks(self, callbacks: list[Callable]):
        """Execute callbacks"""
        for callback in callbacks:
            try:
                logger.debug(f"   Running: {callback.__name__}")
                callback()
            except Exception as e:
                logger.error(f"   Callback error {callback.__name__}: {e}")
    
    def _stop_accepting_new_tasks(self):
        """Stop accepting new tasks"""
        # Stop the ProcessManager
        self.process_manager.running = False
        logger.info("   ✅ New task acceptance has been stopped")
    
    def _handle_open_positions(self):
        """
        Manage open positions (close or protect).

        Note: Requires TradeManager dependency.
        """
        try:
            logger.info("📊 Checking open positions...")

            # Get open positions from the position manager
            if not hasattr(self, 'position_manager') or self.position_manager is None:
                logger.warning("⚠️ PositionManager not found, position check skipped")
                return

            open_positions = self.position_manager.get_open_positions()

            if not open_positions:
                logger.info("✅ No open positions")
                return

            logger.info(f"📊 {len(open_positions)} open positions found")

            # Decide based on the shutdown policy.
            if self.config.close_positions:
                logger.warning("🛑 All positions are being closed...")

                for position in open_positions:
                    try:
                        logger.info(f"   💰 Closing position: {position.symbol}")
                        # Close via the position manager.
                        self.position_manager.close_position(
                            position.id,
                            reason="SHUTDOWN",
                            force=True
                        )
                        logger.info(f"   ✅ Position closed: {position.symbol}")

                    except Exception as e:
                        logger.error(f"   ❌ Position closing error ({position.symbol}): {e}")

                logger.info("✅ All positions have been closed")
            else:
                logger.info("📌 Positions are being left open (policy)")
                # Save positions to file (for crash recovery)
                self._save_positions_state(open_positions)

        except Exception as e:
            logger.error(f"❌ Position management error: {e}")

    def _save_positions_state(self, positions: List) -> None:
        """Save the position state (for crash recovery)"""
        try:
            import json
            state_file = self.config.state_dir / "open_positions.json"

            positions_data = [
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None
                }
                for p in positions
            ]

            with open(state_file, 'w') as f:
                json.dump(positions_data, f, indent=2)

            logger.info(f"💾 Position state saved: {state_file}")

        except Exception as e:
            logger.error(f"❌ Error saving position state: {e}")
    
    def _wait_for_pending_tasks(self):
        """
        Wait for or cancel pending tasks.

        Note: Requires the QueueManager dependency.
        """
        try:
            logger.info("⏳ Checking pending tasks...")

            # Get pending jobs from the queue manager
            if not hasattr(self, 'queue_manager') or self.queue_manager is None:
                logger.warning("⚠️ QueueManager not found, task check skipped")
                return

            pending_jobs = self.queue_manager.get_pending_jobs()
            running_jobs = self.queue_manager.get_running_jobs()

            total_jobs = len(pending_jobs) + len(running_jobs)

            if total_jobs == 0:
                logger.info("✅ No pending/running jobs")
                return

            logger.info(f"📊 {total_jobs} jobs exist (pending: {len(pending_jobs)}, running: {len(running_jobs)})")

            # Decide based on the shutdown policy.
            if self.config.wait_for_tasks:
                logger.info(f"⏳ Waiting for jobs to finish (max {self.config.task_timeout}s)...")

                # Wait for running jobs
                waited = 0
                check_interval = 2  # Check every 2 seconds

                while waited < self.config.task_timeout:
                    running_jobs = self.queue_manager.get_running_jobs()

                    if not running_jobs:
                        logger.info(f"✅ All jobs finished ({waited}s)")
                        break

                    logger.debug(f"⏳ {len(running_jobs)} job is still running... ({waited}s)")
                    time.sleep(check_interval)
                    waited += check_interval

                # Timeout oldu mu?
                if running_jobs:
                    logger.warning(f"⚠️ Timeout! {len(running_jobs)} job is still running, being forcibly stopped...")
                    for job in running_jobs:
                        try:
                            self.queue_manager.cancel_job(job.id, force=True)
                            logger.info(f"   🛑 Job cancelled: {job.id}")
                        except Exception as e:
                            logger.error(f"   ❌ Job cancellation error ({job.id}): {e}")

            else:
                # Task'leri beklemeden iptal et
                logger.warning("🛑 All jobs are being cancelled...")

                for job in pending_jobs + running_jobs:
                    try:
                        self.queue_manager.cancel_job(job.id, force=True)
                        logger.debug(f"   🛑 Job cancelled: {job.id}")
                    except Exception as e:
                        logger.error(f"   ❌ Job cancellation error ({job.id}): {e}")

                logger.info("✅ All jobs have been canceled")

        except Exception as e:
            logger.error(f"❌ Task management error: {e}")
    
    def _stop_engines(self):
        """Stops all engines"""
        logger.info("   Engine'ler durduruluyor...")
        
        # Use the stop_all method of ProcessManager
        success = self.process_manager.stop_all()
        
        if success:
            logger.info("   ✅ All engines have been stopped")
        else:
            logger.error("   ❌ Some engines could not be stopped")
    
    def _flush_buffers(self):
        """Empties all buffers"""
        # EventBus buffer flush
        if self.event_bus and hasattr(self.event_bus, 'event_history'):
            logger.info("   EventBus buffer is being emptied...")
            # The buffer should already be written to the log files.
        
        # Logger buffer flush
        if self.logger_engine:
            logger.info("   Logger buffer is being emptied...")
            logging.shutdown()
        
        logger.info("   ✅ Buffers have been emptied")
    
    def _save_state(self):
        """
        Save the application state (for crash recovery).
        """
        try:
            logger.info("💾 Saving application state...")

            # Use the state persistence engine
            if not hasattr(self, 'state_engine') or self.state_engine is None:
                logger.warning("⚠️ StatePersistenceEngine not found, state saving skipped")
                # At least create a manual entry.
                self._manual_state_save()
                return

            # Get the state of all engines through the Process Manager
            engines_state = {}

            if hasattr(self, 'process_manager') and self.process_manager:
                engines = self.process_manager.get_all_engines()

                for name, engine in engines.items():
                    try:
                        # Get the engine's state (if the get_state method exists)
                        if hasattr(engine, 'get_state'):
                            engines_state[name] = engine.get_state()
                            logger.debug(f"   💾 Engine state retrieved: {name}")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Error getting engine state ({name}): {e}")

            # State'i kaydet
            self.state_engine.save_state('shutdown', {
                'timestamp': datetime.now().isoformat(),
                'reason': 'graceful_shutdown',
                'engines': engines_state
            })

            logger.info("✅ State saved successfully")

        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")

    def _manual_state_save(self) -> None:
        """Manual state saving (fallback)"""
        try:
            import json
            state_file = self.config.state_dir / "shutdown_state.json"

            state_data = {
                'timestamp': datetime.now().isoformat(),
                'reason': 'graceful_shutdown',
                'note': 'Manual save (StatePersistenceEngine not available)'
            }

            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            logger.info(f"💾 Manuel state kaydedildi: {state_file}")

        except Exception as e:
            logger.error(f"❌ Manual state saving error: {e}")
    
    def _exit_gracefully(self):
        """Exit the program"""
        logger.info("👋 SuperBot is shutting down...")
        sys.exit(0)
    
    def _force_shutdown(self):
        """Zorla kapat (timeout durumunda)"""
        logger.critical("⚠️  FORCE SHUTDOWN - The system is being forcibly shut down!")
        
        # EventBus'a bildir
        if self.event_bus:
            try:
                self.event_bus.publish("system.shutdown.forced", {})
            except:
                pass
        
        # Direkt exit
        sys.exit(1)
    
    def is_shutdown_in_progress(self) -> bool:
        """Is the shutdown still in progress?"""
        return self.is_shutting_down
    
    def get_shutdown_info(self) -> Dict[str, Any]:
        """Returns shutdown information"""
        return {
            "is_shutting_down": self.is_shutting_down,
            "initiated_at": self.shutdown_initiated_at.isoformat() if self.shutdown_initiated_at else None,
            "reason": self.shutdown_reason,
            "elapsed_seconds": (
                (datetime.now() - self.shutdown_initiated_at).total_seconds()
                if self.shutdown_initiated_at else 0
            )
        }


# Test kodu
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 GracefulShutdown Test")
    print("=" * 60)
    
    # Mock ProcessManager
    class MockProcessManager:
        def __init__(self):
            self.running = True
        
        def stop_all(self):
            print("   [MockProcessManager] All engines are being stopped...")
            time.sleep(0.5)
            return True
        
        def get_status(self):
            return {"engines": [], "running": False}
    
    # Create GracefulShutdown
    process_manager = MockProcessManager()
    shutdown = GracefulShutdown(
        process_manager=process_manager,
        config=ShutdownConfig(timeout=5)
    )
    
    # Pre/Post callbacks
    def pre_callback():
        print("   [Callback] Pre-shutdown executed")
    
    def post_callback():
        print("   [Callback] Post-shutdown executed")
    
    shutdown.add_pre_shutdown_callback(pre_callback)
    shutdown.add_post_shutdown_callback(post_callback)
    
    # Register signal handlers
    shutdown.register_handlers()
    
    print("\n✅ GracefulShutdown ready")
    print("⌨️ Test with Ctrl+C...")
    print("⏳ 10 saniye bekleniyor...\n")
    
    # Wait for 10 seconds (Ctrl+C can be pressed during this time)
    try:
        for i in range(10, 0, -1):
            print(f"   {i}...", end="\r")
            time.sleep(1)
        
        print("\n\n⏰ Timeout - Manual shutdown is starting...")
        shutdown.initiate(reason="Test timeout")
        
    except KeyboardInterrupt:
        print("\n\n⌨️ Ctrl+C was caught!")
        # The signal handler will run automatically.
    
    print("\n" + "=" * 60)