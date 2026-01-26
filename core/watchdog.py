"""
Watchdog - Module Health Monitoring & Auto-Restart
===================================================

Monitors module health and automatically restarts crashed modules.

Features:
- Periodic health checks
- Process monitoring
- Auto-restart on crash
- Crash alerts
- Restart limits

Author: SuperBot Team
Date: 2025-11-07
"""

import asyncio
import time
from typing import Dict, Any, Optional
import psutil


class Watchdog:
    """
    Watchdog - Module Health Monitor

    Periodically checks module health and restarts crashed modules.
    """

    def __init__(self, logger, process_manager, event_bus, config: Dict):
        """
        Initialize Watchdog

        Args:
            logger: Logger instance
            process_manager: Process manager instance
            event_bus: Event bus for publishing alerts
            config: Watchdog configuration
        """
        self.logger = logger
        self.process_manager = process_manager
        self.event_bus = event_bus
        self.config = config

        # Configuration
        self.enabled = config.get('enabled', True)
        self.check_interval = config.get('check_interval', 30)
        self.auto_restart = config.get('auto_restart_on_crash', True)
        self.max_restart_attempts = config.get('max_restart_attempts', 3)
        self.restart_cooldown = config.get('restart_cooldown', 60)
        self.alert_on_restart = config.get('alert_on_restart', True)

        # State
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # Restart tracking
        self.restart_counts: Dict[str, int] = {}
        self.last_restart: Dict[str, float] = {}

    async def start(self):
        """Start watchdog monitoring"""
        if not self.enabled:
            self.logger.info("Watchdog is disabled")
            return

        self.running = True
        self.task = asyncio.create_task(self._monitor_loop())

        self.logger.info(
            f"Watchdog started (check_interval={self.check_interval}s, "
            f"auto_restart={self.auto_restart})"
        )

    async def stop(self):
        """Stop watchdog monitoring"""
        if not self.running:
            return

        self.running = False

        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info("Watchdog stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("Watchdog monitor loop started")

        try:
            while self.running:
                await self._check_all_modules()
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            self.logger.info("Watchdog monitor loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in watchdog monitor loop: {e}", exc_info=True)

    async def _check_all_modules(self):
        """Check all modules"""
        # Get all modules from daemon
        daemon = self.process_manager.daemon if hasattr(self.process_manager, 'daemon') else None

        if not daemon:
            # Fallback: get from process manager
            modules = {}
        else:
            modules = daemon.modules

        for module_name, module_info in modules.items():
            if module_info['status'] != 'running':
                continue

            # Check if module is healthy
            is_healthy = await self._check_module_health(module_name, module_info)

            if not is_healthy:
                await self._handle_unhealthy_module(module_name, module_info)

    async def _check_module_health(self, module_name: str, module_info: Dict) -> bool:
        """
        Check if module is healthy

        Args:
            module_name: Module name
            module_info: Module information

        Returns:
            True if healthy, False otherwise
        """
        pid = module_info.get('pid')

        if not pid:
            return False

        # Check if process exists
        try:
            process = psutil.Process(pid)

            # Check if process is running
            if process.status() == psutil.STATUS_ZOMBIE:
                self.logger.warning(f"Module '{module_name}' is zombie (PID: {pid})")
                return False

            # Process is alive
            return True

        except psutil.NoSuchProcess:
            self.logger.warning(f"Module '{module_name}' process not found (PID: {pid})")
            return False

        except psutil.AccessDenied:
            # Can't check, assume it's running
            return True

        except Exception as e:
            self.logger.error(f"Error checking module '{module_name}': {e}")
            return False

    async def _handle_unhealthy_module(self, module_name: str, module_info: Dict):
        """
        Handle unhealthy module

        Args:
            module_name: Module name
            module_info: Module information
        """
        self.logger.error(f"Module '{module_name}' is unhealthy!")

        # Publish crash event
        await self.event_bus.publish('module.crashed', {
            'module': module_name,
            'pid': module_info.get('pid'),
            'restart_count': self.restart_counts.get(module_name, 0)
        })

        # Update module status
        module_info['status'] = 'crashed'

        # Check if we should restart
        if not self.auto_restart:
            self.logger.info(f"Auto-restart is disabled for '{module_name}'")
            return

        # Check restart limits
        restart_count = self.restart_counts.get(module_name, 0)

        if restart_count >= self.max_restart_attempts:
            self.logger.error(
                f"Module '{module_name}' exceeded max restart attempts "
                f"({self.max_restart_attempts})"
            )
            await self._send_alert(module_name, "max_restarts_exceeded")
            return

        # Check restart cooldown
        last_restart = self.last_restart.get(module_name, 0)
        time_since_restart = time.time() - last_restart

        if time_since_restart < self.restart_cooldown:
            self.logger.warning(
                f"Module '{module_name}' in restart cooldown "
                f"({self.restart_cooldown - time_since_restart:.1f}s remaining)"
            )
            return

        # Attempt restart
        await self._restart_module(module_name)

    async def _restart_module(self, module_name: str):
        """
        Restart a module

        Args:
            module_name: Module name
        """
        self.logger.info(f"Attempting to restart module: {module_name}")

        try:
            # Get daemon
            daemon = self.process_manager.daemon if hasattr(self.process_manager, 'daemon') else None

            if not daemon:
                self.logger.error("Cannot restart: daemon not available")
                return

            # Restart module
            success = await daemon.restart_module(module_name)

            if success:
                # Update restart tracking
                self.restart_counts[module_name] = self.restart_counts.get(module_name, 0) + 1
                self.last_restart[module_name] = time.time()

                self.logger.info(
                    f"Module '{module_name}' restarted successfully "
                    f"(restart count: {self.restart_counts[module_name]})"
                )

                # Send alert if configured
                if self.alert_on_restart:
                    await self._send_alert(module_name, "restarted")

                # Publish restart event
                await self.event_bus.publish('module.restarted', {
                    'module': module_name,
                    'restart_count': self.restart_counts[module_name]
                })

            else:
                self.logger.error(f"Failed to restart module '{module_name}'")
                await self._send_alert(module_name, "restart_failed")

        except Exception as e:
            self.logger.error(f"Error restarting module '{module_name}': {e}")
            await self._send_alert(module_name, "restart_error")

    async def _send_alert(self, module_name: str, alert_type: str):
        """
        Send alert notification

        Args:
            module_name: Module name
            alert_type: Alert type (restarted, restart_failed, max_restarts_exceeded)
        """
        messages = {
            'restarted': f"⚠️ Module '{module_name}' crashed and was automatically restarted",
            'restart_failed': f"❌ Module '{module_name}' crashed and failed to restart",
            'max_restarts_exceeded': f"🚨 Module '{module_name}' exceeded max restart attempts",
            'restart_error': f"❌ Error restarting module '{module_name}'"
        }

        message = messages.get(alert_type, f"Alert for module '{module_name}': {alert_type}")

        # Publish alert event (notification system will pick it up)
        await self.event_bus.publish('system.alert', {
            'severity': 'critical' if 'failed' in alert_type else 'warning',
            'module': module_name,
            'type': alert_type,
            'message': message
        })

        self.logger.warning(f"Alert sent: {message}")

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check

        Returns:
            Health check results
        """
        health = {}

        # Get daemon
        daemon = self.process_manager.daemon if hasattr(self.process_manager, 'daemon') else None

        if not daemon:
            return {'status': 'error', 'message': 'Daemon not available'}

        # Check each module
        for module_name, module_info in daemon.modules.items():
            if module_info['status'] != 'running':
                health[module_name] = {
                    'status': 'stopped',
                    'message': f"Module is {module_info['status']}"
                }
                continue

            is_healthy = await self._check_module_health(module_name, module_info)

            health[module_name] = {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'pid': module_info.get('pid'),
                'restart_count': self.restart_counts.get(module_name, 0)
            }

        return health

    def reset_restart_count(self, module_name: str):
        """Reset restart count for module"""
        if module_name in self.restart_counts:
            self.restart_counts[module_name] = 0
            self.logger.info(f"Reset restart count for '{module_name}'")
