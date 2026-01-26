#!/usr/bin/env python3
"""
superbot.py
SuperBot - Master Daemon Orchestrator
Author: SuperBot Team
Date: 2025-11-07
Version: 1.0.0

SuperBot modules are managed by this central daemon (Trading, AI, WebUI, Backtest, etc.).
It provides process lifecycle management, resource allocation, health monitoring, and an IPC server.

Features:
- AI Module management (uvicorn FastAPI)
- WebUI Module management (Flask)
- Trading Module management (async Python)
- Backtest Module management (on-demand)
- IPC/RPC server (Unix socket / TCP)
- Watchdog (automatic restart)
- Task Scheduler (scheduled tasks)
- Thread pool management
- Graceful shutdown

Usage:
    python superbot.py                  # Start the daemon (foreground)
    python superbot-cli.py daemon start # Start the daemon (background)
    python superbot-cli.py daemon stop  # Stop the daemon
    python superbot-cli.py daemon status # Show the status of the daemon

Dependencies:
    - python>=3.10
    - uvicorn[standard]
    - waitress
    - aiohttp
    - click
    - pytz
"""

import sys
import os
import signal
import time
import asyncio
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import psutil

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Core imports
from core.config_engine import ConfigEngine
from core.logger_engine import LoggerEngine
from core.event_bus import EventBus
from core.cache_manager import CacheManager
from core.process_manager import ProcessManager
from core.graceful_shutdown import GracefulShutdown

# Component imports
from core.ipc_server import IPCServer
from core.module_launcher import ModuleLauncher
from core.thread_pool_manager import ThreadPoolManager
from core.watchdog import Watchdog
from core.scheduler import TaskScheduler

# Monitoring imports
from components.monitoring import ResourceMonitor


class SuperBotDaemon:
    """
    Master Daemon Orchestrator

    Manages lifecycle of all SuperBot modules:
    - AI Module (uvicorn FastAPI)
    - WebUI Module (Flask)
    - Trading Module (async Python)
    - Backtest Module (on-demand)
    - Optimizer Module (on-demand)
    - Monitoring Module (thread)
    """

    def __init__(self, config_path: str = "config"):
        """Start the daemon with the config path"""
        self.config_path = config_path
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Runtime paths
        self.runtime_dir = Path(".superbot")
        self.runtime_dir.mkdir(exist_ok=True)

        self.pid_file = self.runtime_dir / "daemon.pid"
        self.socket_path = "/tmp/superbot.sock"

        # CORE infrastructure (initialized in initialize_core)
        self.config: Optional[ConfigEngine] = None
        self.logger: Optional[LoggerEngine] = None
        self.event_bus: Optional[EventBus] = None
        self.cache: Optional[CacheManager] = None
        self.process_manager: Optional[ProcessManager] = None
        self.graceful_shutdown: Optional[GracefulShutdown] = None

        # Components specific to the daemon
        self.ipc_server: Optional[IPCServer] = None
        self.module_launcher: Optional[ModuleLauncher] = None
        self.thread_pool_manager: Optional[ThreadPoolManager] = None
        self.watchdog: Optional[Watchdog] = None
        self.scheduler: Optional[TaskScheduler] = None
        self.resource_monitor: Optional[ResourceMonitor] = None

        # Module registration log
        self.modules: Dict[str, Dict[str, Any]] = {}

        # Set up signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for safe shutdown"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._sighup_handler)

    def _signal_handler(self, signum, frame):
        """Manage shutdown signals"""
        signal_name = signal.Signals(signum).name
        if self.logger:
            self.logger.info(f"⚠️ {signal_name} received, initiating safe shutdown...")
        else:
            print(f"⚠️ {signal_name} received, initiating safe shutdown...")

        # Stop running flag (causes _run_forever to exit)
        self.running = False

    def _sighup_handler(self, signum, frame):
        """SIGHUP management (config reload)"""
        if self.logger:
            self.logger.info("📝 SIGHUP received, configuration is being reloaded...")

        try:
            self.reload_config()
            if self.logger:
                self.logger.info("✅ Configuration reloaded successfully")
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error reloading config: {e}")

    def check_already_running(self) -> bool:
        """Check if the daemon is already running"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if the process exists
            if psutil.pid_exists(pid):
                # Verify it's actually our daemon
                try:
                    proc = psutil.Process(pid)
                    if 'superbot.py' in ' '.join(proc.cmdline()):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Stale PID file, remove it
            self.pid_file.unlink()
            return False

        except (ValueError, FileNotFoundError):
            return False

    def write_pid_file(self):
        """Write current PID to file"""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))

    def remove_pid_file(self):
        """Remove PID file"""
        if self.pid_file.exists():
            self.pid_file.unlink()

    async def initialize_core(self):
        """Initializes the CORE infrastructure layer"""
        print("🚀 SuperBot Daemon is starting...")

        # 1. Config Engine (load all configurations)
        print("  ├─ Loading configuration...")
        self.config = ConfigEngine(base_path=self.config_path)
        self.config.load_all([
            "main.yaml",
            "connectors.yaml",
            "infrastructure.yaml",
            "daemon.yaml"
        ])

        # 2. Logger Engine
        print("  ├─ Logger is being initialized...")
        log_config = self.config.get('logging', {})
        logger_engine = LoggerEngine(config=log_config)
        self.logger = logger_engine.get_logger('daemon')
        self.logger.info("🚀 SuperBot Daemon is starting...")

        # 3. Event Bus
        self.logger.info("📡 Starting event bus...")
        eventbus_config = self.config.get('infrastructure.eventbus', {})
        self.event_bus = EventBus(config=eventbus_config)

        # 4. Cache Manager
        self.logger.info("💾 Cache manager is starting...")
        cache_config = self.config.get('infrastructure.cache', {})
        self.cache = CacheManager(config=cache_config)

        # 5. Process Manager
        self.logger.info("⚙️ Process manager is starting...")
        pm_config = self.config.get('performance', {})
        self.process_manager = ProcessManager(config=pm_config)

        # 6. Graceful Shutdown Handler
        shutdown_config = self.config.get('shutdown', {})
        self.graceful_shutdown = GracefulShutdown(
            logger=self.logger,
            config=shutdown_config
        )

        self.logger.info("✓ ✅ CORE infrastructure started")

    async def initialize_daemon_components(self):
        """Initialize daemon-specific components"""
        daemon_config = self.config.get('daemon', {})

        # 1. Module Launcher
        self.logger.info("🔧 Module launcher is starting...")
        self.module_launcher = ModuleLauncher(
            logger=self.logger,
            event_bus=self.event_bus,
            runtime_dir=self.runtime_dir
        )

        # 2. Thread Pool Manager
        self.logger.info("🧵 Thread pool manager is starting...")
        thread_config = self.config.get('daemon.resource_allocation.thread_pools', {})
        self.thread_pool_manager = ThreadPoolManager(
            logger=self.logger,
            config=thread_config
        )

        # 3. IPC Server
        self.logger.info("📡 Starting IPC server...")
        ipc_config = daemon_config.get('ipc', {})
        self.socket_path = ipc_config.get('socket_path', '/tmp/superbot.sock')
        self.tcp_host = ipc_config.get('tcp_host', '127.0.0.1')
        self.tcp_port = ipc_config.get('tcp_port', 9999)

        # Clean up the old socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.ipc_server = IPCServer(
            socket_path=self.socket_path,
            logger=self.logger,
            daemon=self  # Pass self for RPC method access
        )

        # Register RPC handlers
        self._register_rpc_handlers()

        # 4. Watchdog
        self.logger.info("🐕 Watchdog is starting...")
        watchdog_config = daemon_config.get('watchdog', {})
        self.watchdog = Watchdog(
            logger=self.logger,
            process_manager=self.process_manager,
            event_bus=self.event_bus,
            config=watchdog_config
        )

        # 5. Scheduler
        self.logger.info("⏰ Scheduler is starting...")
        schedule_config = daemon_config.get('schedule', {})
        self.scheduler = TaskScheduler(
            logger=self.logger,
            config=schedule_config,
            daemon=self  # For scheduled task access
        )

        # 6. Resource Monitor
        self.logger.info("📊 Resource monitor is starting...")
        monitoring_config = self.config.get('monitoring', {})
        self.resource_monitor = ResourceMonitor(
            config={
                'monitor_interval': monitoring_config.get('metrics', {}).get('collect_interval', 10),
                'cpu_threshold': 80,
                'memory_threshold': 80,
                'disk_threshold': 90
            },
            event_bus=self.event_bus
        )

        self.logger.info("✓ ✅ Daemon components started")

    def _register_rpc_handlers(self):
        """Registers RPC method handlers for the IPC server"""
        # Daemon control
        self.ipc_server.register_handler('daemon.status', self.rpc_daemon_status)
        self.ipc_server.register_handler('daemon.stop', self.rpc_daemon_stop)
        self.ipc_server.register_handler('daemon.reload_config', self.rpc_reload_config)

        # Module check
        self.ipc_server.register_handler('module.start', self.rpc_module_start)
        self.ipc_server.register_handler('module.stop', self.rpc_module_stop)
        self.ipc_server.register_handler('module.restart', self.rpc_module_restart)
        self.ipc_server.register_handler('module.status', self.rpc_module_status)
        self.ipc_server.register_handler('module.list', self.rpc_module_list)

        # Trading operations (proxy to the trading module)
        self.ipc_server.register_handler('trading.positions', self.rpc_trading_positions)
        self.ipc_server.register_handler('trading.orders', self.rpc_trading_orders)
        self.ipc_server.register_handler('trading.balance', self.rpc_trading_balance)

        # Monitoring
        self.ipc_server.register_handler('monitoring.health', self.rpc_health_check)
        self.ipc_server.register_handler('monitoring.metrics', self.rpc_get_metrics)
        self.ipc_server.register_handler('monitoring.resources', self.rpc_get_resources)

        # Logs
        self.ipc_server.register_handler('logs.stream', self.rpc_stream_logs)

        self.logger.info(f"Registered {len(self.ipc_server.handlers)} RPC handlers")

    async def load_module_definitions(self):
        """Load module definitions from the config."""
        modules_config = self.config.get('modules', {})

        for module_name, module_config in modules_config.items():
            if not module_config.get('enabled', False):
                continue

            self.modules[module_name] = {
                'name': module_name,
                'config': module_config,
                'status': 'stopped',
                'pid': None,
                'process': None,
                'start_time': None,
                'restart_count': 0
            }

        self.logger.info(f"Loaded {len(self.modules)} module definition")

    async def start_autostart_modules(self):
        """Starts modules marked for autostart"""
        autostart = self.config.get('daemon.autostart', [])

        if not autostart:
            self.logger.info("Autostart module is not configured")
            return

        self.logger.info(f"🚀 Autostart modules are being initialized: {autostart}")

        for module_name in autostart:
            if module_name not in self.modules:
                self.logger.warning(f"Autostart module '{module_name}' not found in configurations")
                continue

            try:
                await self.start_module(module_name)
            except Exception as e:
                self.logger.error(f"Error starting autostart module '{module_name}': {e}")

    async def start_module(self, module_name: str, params: Optional[Dict] = None) -> bool:
        """Module initialization"""
        if module_name not in self.modules:
            self.logger.error(f"❌ Unknown module: {module_name}")
            return False

        module = self.modules[module_name]

        if module['status'] == 'running':
            self.logger.warning(f"⚠️ Module '{module_name}' is already running")
            return True

        self.logger.info(f"🚀 Module is starting: {module_name}")

        try:
            # Merge parameters with the config
            config = module['config'].copy()
            if params:
                config.update(params)

            # Initialize the module
            process_info = await self.module_launcher.launch_module(module_name, config)

            # Update module status
            module['status'] = 'running'
            module['pid'] = process_info.get('pid')
            module['process'] = process_info.get('process')
            module['start_time'] = time.time()

            # Publish event
            self.event_bus.publish('module.started', {
                'module': module_name,
                'pid': module['pid']
            })

            self.logger.info(f"✓ Module '{module_name}' started (PID: {module['pid']})")
            return True

        except Exception as e:
            self.logger.error(f"Module initialization error '{module_name}': {e}")
            module['status'] = 'error'
            return False

    async def stop_module(self, module_name: str, graceful: bool = True) -> bool:
        """Module stop"""
        if module_name not in self.modules:
            self.logger.error(f"❌ Unknown module: {module_name}")
            return False

        module = self.modules[module_name]

        if module['status'] != 'running':
            self.logger.warning(f"⚠️ Module '{module_name}' is not working")
            return True

        self.logger.info(f"🛑 Module is being stopped: {module_name}")

        try:
            # Stop the module
            await self.module_launcher.stop_module(module_name, module['process'], graceful)

            # Update module status
            module['status'] = 'stopped'
            module['pid'] = None
            module['process'] = None

            # Publish event
            self.event_bus.publish('module.stopped', {
                'module': module_name
            })

            self.logger.info(f"✓ Module '{module_name}' stopped")
            return True

        except Exception as e:
            self.logger.error(f"Module shutdown error '{module_name}': {e}")
            return False

    async def restart_module(self, module_name: str) -> bool:
        """Module restart"""
        self.logger.info(f"🔄 Module restarting: {module_name}")

        await self.stop_module(module_name)
        await asyncio.sleep(2)  # Wait a bit
        return await self.start_module(module_name)

    def reload_config(self):
        """Reload configuration"""
        self.logger.info("📝 Configuration is being reloaded...")

        try:
            # Reload all configuration files
            self.config.load_all([
                "main.yaml",
                "connectors.yaml",
                "infrastructure.yaml",
                "daemon.yaml"
            ])

            # Publish event
            self.event_bus.publish('config.reloaded', {})

            self.logger.info("✓ ✅ Configuration reloaded")

        except Exception as e:
            self.logger.error(f"❌ Error reloading configuration: {e}")
            raise

    async def start(self):
        """Start the daemon"""
        try:
            # Check if it is already running.
            if self.check_already_running():
                print("❌ Daemon is already running!")
                print(f"   PID file: {self.pid_file}")
                sys.exit(1)

            # Write the PID file
            self.write_pid_file()

            # Start the core infrastructure
            await self.initialize_core()

            # Start the daemon components
            await self.initialize_daemon_components()

            # Load module definitions
            await self.load_module_definitions()

            # Start the IPC server
            self.logger.info("📡 Starting IPC server...")
            await self.ipc_server.start(tcp_host=self.tcp_host, tcp_port=self.tcp_port)
            self.logger.info(f"✓ IPC Server listening on {self.ipc_server.socket_path}")

            # Start the Watchdog.
            self.logger.info("🐕 Watchdog is starting...")
            await self.watchdog.start()

            # Start the scheduler
            self.logger.info("⏰ Scheduler is starting...")
            await self.scheduler.start()

            # Start resource monitor
            self.logger.info("📊 Resource monitor is starting...")
            await self.resource_monitor.start()

            # Initialize autostart modules
            await self.start_autostart_modules()

            # Mark as running
            self.running = True

            self.logger.info("=" * 60)
            self.logger.info("🎭 SuperBot Daemon IS RUNNING")
            self.logger.info(f"   PID: {os.getpid()}")
            self.logger.info(f"   IPC Socket: {self.socket_path}")
            self.logger.info(f"   Running Modules: {[name for name, m in self.modules.items() if m['status'] == 'running']}")
            self.logger.info("=" * 60)

            # Publish the system ready event
            self.event_bus.publish('system.ready', {
                'pid': os.getpid(),
                'modules': list(self.modules.keys())
            })

            # Continue execution
            await self._run_forever()

            # When _run_forever finishes (due to Ctrl+C or running=False), perform cleanup.
            await self.stop()

        except Exception as e:
            print(f"❌ ❌ Daemon startup error: {e}")
            if self.logger:
                self.logger.error(f"❌ Daemon startup error: {e}", exc_info=True)
            await self.stop()
            sys.exit(1)

    async def _run_forever(self):
        """Keep the daemon running"""
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("⚠️ Keyboard interrupt signal received")
            # Set running=False to trigger cleanup in start()
            self.running = False

    async def stop(self):
        """Safely stop the daemon"""
        if not self.running:
            return

        self.running = False

        self.logger.info("=" * 60)
        self.logger.info("🛑 SuperBot Daemon is stopping...")
        self.logger.info("=" * 60)

        try:
            # Trigger graceful shutdown
            if self.graceful_shutdown:
                await self.graceful_shutdown.initiate()

            # Stop scheduler
            if self.scheduler:
                self.logger.info("Stopping scheduler...")
                await self.scheduler.stop()

            # Stop watchdog
            if self.watchdog:
                self.logger.info("Stopping watchdog...")
                await self.watchdog.stop()

            # Stop resource monitor
            if self.resource_monitor:
                self.logger.info("Stopping resource monitor...")
                await self.resource_monitor.stop()

            # Stop all modules
            self.logger.info("Stopping all modules...")
            running_modules = [name for name, m in self.modules.items() if m['status'] == 'running']

            for module_name in running_modules:
                await self.stop_module(module_name)

            # Stop IPC server
            if self.ipc_server:
                self.logger.info("Stopping IPC server...")
                await self.ipc_server.stop()

            # Stop thread pools
            if self.thread_pool_manager:
                self.logger.info("Shutting down thread pools...")
                self.thread_pool_manager.shutdown_all()

            # Stop CORE components
            # Cache and EventBus might not have an async close method, check it.
            if self.cache and hasattr(self.cache, 'close'):
                self.logger.info("Closing cache connections...")
                if asyncio.iscoroutinefunction(self.cache.close):
                    await self.cache.close()
                else:
                    self.cache.close()

            if self.event_bus and hasattr(self.event_bus, 'close'):
                self.logger.info("Stopping event bus...")
                if asyncio.iscoroutinefunction(self.event_bus.close):
                    await self.event_bus.close()
                else:
                    self.event_bus.close()

            # Remove PID file
            self.remove_pid_file()

            # Remove socket
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            self.logger.info("=" * 60)
            self.logger.info("✅ SuperBot Daemon was stopped safely")
            self.logger.info("=" * 60)

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error during shutdown: {e}", exc_info=True)
            else:
                print(f"❌ Error during shutdown: {e}")

    # ============================================================
    # RPC METHOD HANDLERS
    # ============================================================

    async def rpc_daemon_status(self, params: Dict) -> Dict:
        """Get daemon status"""
        uptime = time.time() - psutil.Process(os.getpid()).create_time()

        return {
            'status': 'running' if self.running else 'stopped',
            'pid': os.getpid(),
            'uptime': int(uptime),
            'modules': {
                name: {
                    'status': module['status'],
                    'pid': module['pid'],
                    'uptime': int(time.time() - module['start_time']) if module['start_time'] else 0
                }
                for name, module in self.modules.items()
            }
        }

    async def rpc_daemon_stop(self, params: Dict) -> Dict:
        """Stop the daemon"""
        asyncio.create_task(self.stop())
        return {'status': 'stopping'}

    async def rpc_reload_config(self, params: Dict) -> Dict:
        """Reload configuration"""
        try:
            self.reload_config()
            return {'status': 'success', 'message': '✅ Configuration reloaded'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def rpc_module_start(self, params: Dict) -> Dict:
        """Module initialization"""
        module_name = params.get('module')
        module_params = params.get('params', {})

        success = await self.start_module(module_name, module_params)

        return {
            'status': 'success' if success else 'error',
            'module': module_name,
            'pid': self.modules[module_name]['pid'] if success else None
        }

    async def rpc_module_stop(self, params: Dict) -> Dict:
        """Module stop"""
        module_name = params.get('module')
        graceful = params.get('graceful', True)

        success = await self.stop_module(module_name, graceful)

        return {
            'status': 'success' if success else 'error',
            'module': module_name
        }

    async def rpc_module_restart(self, params: Dict) -> Dict:
        """Restart a module"""
        module_name = params.get('module')

        success = await self.restart_module(module_name)

        return {
            'status': 'success' if success else 'error',
            'module': module_name
        }

    async def rpc_module_status(self, params: Dict) -> Dict:
        """Get module status"""
        module_name = params.get('module')

        if module_name not in self.modules:
            return {'status': 'error', 'message': 'Module not found'}

        module = self.modules[module_name]

        return {
            'status': 'success',
            'module': module_name,
            'state': module['status'],
            'pid': module['pid'],
            'uptime': int(time.time() - module['start_time']) if module['start_time'] else 0,
            'restart_count': module['restart_count']
        }

    async def rpc_module_list(self, params: Dict) -> Dict:
        """List all modules"""
        return {
            'status': 'success',
            'modules': list(self.modules.keys())
        }

    async def rpc_trading_positions(self, params: Dict) -> Dict:
        """Acquire trading positions (proxy to the trading module)"""
        # TODO: Implement proxy to trading module via event bus or direct call
        return {'status': 'not_implemented'}

    async def rpc_trading_orders(self, params: Dict) -> Dict:
        """Receives trading orders (proxy to the trading module)"""
        # TODO: Implement proxy to trading module
        return {'status': 'not_implemented'}

    async def rpc_trading_balance(self, params: Dict) -> Dict:
        """Get account balance (proxy to the trading module)"""
        # TODO: Implement proxy to trading module
        return {'status': 'not_implemented'}

    async def rpc_health_check(self, params: Dict) -> Dict:
        """Perform a health check"""
        if self.watchdog:
            health = await self.watchdog.check_health()
            return {'status': 'success', 'health': health}
        return {'status': 'error', 'message': 'Watchdog has not been started'}

    async def rpc_get_metrics(self, params: Dict) -> Dict:
        """Get performance metrics"""
        # TODO: Collect metrics from all modules
        return {'status': 'not_implemented'}

    async def rpc_get_resources(self, params: Dict) -> Dict:
        """Get resource usage"""
        # Use resource monitor if available
        if self.resource_monitor:
            stats = self.resource_monitor.get_stats()
            process_info = self.resource_monitor.get_process_info()

            return {
                'status': 'success',
                'resources': {
                    'system': stats.get('current', {}),
                    'averages': stats.get('averages', {}),
                    'thresholds': stats.get('thresholds', {}),
                    'alerts': {
                        'cpu_alerts': stats.get('cpu_alerts', 0),
                        'memory_alerts': stats.get('memory_alerts', 0),
                        'disk_alerts': stats.get('disk_alerts', 0)
                    },
                    'process': process_info
                }
            }

        # Fallback to basic psutil
        process = psutil.Process(os.getpid())

        return {
            'status': 'success',
            'resources': {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'threads': process.num_threads(),
                'connections': len(process.connections()),
                'open_files': len(process.open_files())
            }
        }

    async def rpc_stream_logs(self, params: Dict) -> Dict:
        """Transfer logs (not implemented with RPC, use WebSocket)"""
        return {'status': 'not_implemented', 'message': 'Use WebSocket for log streaming'}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SuperBot Master Daemon')
    parser.add_argument('--daemon', action='store_true', help='Run as a daemon in the background')
    parser.add_argument('--stop', action='store_true', help='Stop the running daemon')
    parser.add_argument('--status', action='store_true', help='Check the daemon status')
    parser.add_argument('--config', default='config', help='Config directory path')

    args = parser.parse_args()

    daemon = SuperBotDaemon(config_path=args.config)

    if args.stop:
        # TODO: Add a stop feature with the IPC client.
        print("Usage: superbot-cli daemon stop")
        sys.exit(0)

    if args.status:
        # TODO: Add status check with the IPC client
        print("Usage: superbot-cli daemon status")
        sys.exit(0)

    if args.daemon:
        # TODO: Proper daemonization (forking in the background) will be added.
        print("⚠️ Background daemon mode has not been implemented yet")
        print("📺 Running in the foreground...")

    # Run the daemon
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
