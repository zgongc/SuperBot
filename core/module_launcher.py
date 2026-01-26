"""
Module Launcher - Multi-Type Module Process Manager
====================================================

Launches and manages different types of modules:
- uvicorn (FastAPI - AI Module)
- flask (WebUI Module)
- python (Async Python - Trading Module)
- thread (Background threads - Monitoring)

Author: SuperBot Team
Date: 2025-11-07
"""

import asyncio
import subprocess
import sys
import os
import signal
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import time


class ModuleLauncher:
    """
    Module Launcher - Handles different module types

    Supports:
    - uvicorn: FastAPI apps (AI Module)
    - flask: Flask apps (WebUI Module)
    - python: Python scripts/modules (Trading, Backtest)
    - thread: Background threads (Monitoring)
    """

    def __init__(self, logger, event_bus, runtime_dir: Path):
        """
        Initialize Module Launcher

        Args:
            logger: Logger instance
            event_bus: Event bus for publishing events
            runtime_dir: Runtime directory for PID files
        """
        self.logger = logger
        self.event_bus = event_bus
        self.runtime_dir = runtime_dir

        # Track launched processes/threads
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threads: Dict[str, threading.Thread] = {}

    async def launch_module(self, module_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Launch a module based on its type

        Args:
            module_name: Module name
            config: Module configuration

        Returns:
            Dict with process info (pid, process, etc.)
        """
        module_type = config.get('type', 'python')

        self.logger.info(f"Launching module '{module_name}' (type: {module_type})")

        if module_type == 'uvicorn':
            return await self._launch_uvicorn(module_name, config)
        elif module_type == 'flask':
            return await self._launch_flask(module_name, config)
        elif module_type == 'python':
            return await self._launch_python(module_name, config)
        elif module_type == 'thread':
            return await self._launch_thread(module_name, config)
        else:
            raise ValueError(f"Unknown module type: {module_type}")

    async def _launch_uvicorn(self, module_name: str, config: Dict) -> Dict:
        """Launch uvicorn FastAPI app"""
        app = config['app']
        host = config.get('host', '127.0.0.1')
        port = config.get('port', 8000)
        workers = config.get('workers', 1)
        debug = config.get('debug', False)

        # Build uvicorn command
        cmd = [
            sys.executable, '-m', 'uvicorn',
            app,
            '--host', host,
            '--port', str(port),
            '--log-level', 'debug' if debug else 'info'
        ]

        # Add debug-specific flags
        if debug:
            cmd.append('--reload')  # Auto-reload on code changes
            self.logger.info(f"🐛 Starting uvicorn in DEBUG mode (auto-reload enabled)")
        else:
            cmd.extend(['--workers', str(workers)])

        self.logger.info(f"Starting uvicorn: {' '.join(cmd)}")

        # Start process - redirect to log file (prevents PIPE buffer overflow issues)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        stdout_file = open(log_dir / f"{module_name}_stdout.log", "a")
        stderr_file = open(log_dir / f"{module_name}_stderr.log", "a")

        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=str(Path.cwd())
        )

        # Wait a bit to ensure it started
        await asyncio.sleep(2)

        # Check if still running
        if process.poll() is not None:
            # Read the last lines from the log file
            stderr_file.flush()
            stderr_log = log_dir / f"{module_name}_stderr.log"
            error_output = ""
            if stderr_log.exists():
                with open(stderr_log, 'r') as f:
                    lines = f.readlines()
                    error_output = ''.join(lines[-20:])  # Last 20 lines
            raise RuntimeError(
                f"uvicorn failed to start (exit code: {process.returncode}):\n"
                f"stderr (last lines): {error_output}"
            )

        # Store process and file handles for cleanup
        self.processes[module_name] = process

        # Write PID file
        pid_file = self.runtime_dir / f"{module_name}.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        self.logger.info(f"uvicorn started (PID: {process.pid})")

        return {
            'pid': process.pid,
            'process': process,
            'type': 'uvicorn',
            'port': port
        }

    async def _launch_flask(self, module_name: str, config: Dict) -> Dict:
        """Launch Flask app"""
        app = config['app']  # e.g., "modules.webui.app:app"
        host = config.get('host', '0.0.0.0')
        port = config.get('port', 8080)
        debug = config.get('debug', False)

        # Build flask command
        # Debug mode: Use Flask development server with auto-reload
        # Production mode: Use waitress (better than flask dev server)
        if debug:
            # Use Flask's built-in development server with debug mode
            # Set environment variables for Flask
            env = os.environ.copy()
            env['FLASK_APP'] = app
            env['FLASK_DEBUG'] = '1'

            cmd = [
                sys.executable, '-m', 'flask',
                'run',
                '--host=' + host,
                '--port=' + str(port),
                '--debug'
            ]
            self.logger.info(f"🐛 Starting Flask (development server) in DEBUG mode: {' '.join(cmd)}")
        else:
            # Production mode with waitress
            env = None
            cmd = [
                sys.executable, '-m', 'waitress',
                '--host=' + host,
                '--port=' + str(port),
                app  # module.path:app format
            ]
            self.logger.info(f"🍾 Starting Flask (waitress): {' '.join(cmd)}")

        # Start process
        # NOTE: If we connect stdout/stderr to PIPE, the subprocess will be blocked when the buffer is full!
        # When the daemon closes, there is no one left to read the pipes -> Flask freezing issue
        # Solution: Redirect to a log file or use DEVNULL
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        stdout_file = open(log_dir / f"{module_name}_stdout.log", "a")
        stderr_file = open(log_dir / f"{module_name}_stderr.log", "a")

        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=str(Path.cwd()),
            env=env  # Set Flask environment variables for debug mode
        )

        # Wait a bit
        await asyncio.sleep(2)

        # Check if still running
        if process.poll() is not None:
            # Read the last lines from the log file
            stderr_file.flush()
            stderr_log = log_dir / f"{module_name}_stderr.log"
            error_output = ""
            if stderr_log.exists():
                with open(stderr_log, 'r') as f:
                    lines = f.readlines()
                    error_output = ''.join(lines[-20:])  # Last 20 lines
            raise RuntimeError(
                f"Flask failed to start (exit code: {process.returncode}):\n"
                f"stderr (last lines): {error_output}"
            )

        # Store process
        self.processes[module_name] = process

        # Write PID file
        pid_file = self.runtime_dir / f"{module_name}.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        self.logger.info(f"Flask started (PID: {process.pid})")

        return {
            'pid': process.pid,
            'process': process,
            'type': 'flask',
            'port': port
        }

    async def _launch_python(self, module_name: str, config: Dict) -> Dict:
        """Launch Python module/script"""
        module = config['module']  # e.g., "modules.trading.engine"
        args = config.get('args', [])

        # Build python command
        cmd = [sys.executable, '-m', module] + args

        self.logger.info(f"Starting Python module: {' '.join(cmd)}")

        # Start process - redirect to log file (prevents PIPE buffer overflow issues)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        stdout_file = open(log_dir / f"{module_name}_stdout.log", "a")
        stderr_file = open(log_dir / f"{module_name}_stderr.log", "a")

        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=str(Path.cwd())
        )

        # Wait a bit
        await asyncio.sleep(1)

        # Check if still running
        if process.poll() is not None:
            # Read the last lines from the log file
            stderr_file.flush()
            stderr_log = log_dir / f"{module_name}_stderr.log"
            error_output = ""
            if stderr_log.exists():
                with open(stderr_log, 'r') as f:
                    lines = f.readlines()
                    error_output = ''.join(lines[-20:])  # Last 20 lines
            raise RuntimeError(
                f"Python module failed to start (exit code: {process.returncode}):\n"
                f"stderr (last lines): {error_output}"
            )

        # Store process
        self.processes[module_name] = process

        # Write PID file
        pid_file = self.runtime_dir / f"{module_name}.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        self.logger.info(f"Python module started (PID: {process.pid})")

        return {
            'pid': process.pid,
            'process': process,
            'type': 'python'
        }

    async def _launch_thread(self, module_name: str, config: Dict) -> Dict:
        """Launch background thread"""
        module = config['module']  # e.g., "modules.monitoring.health_monitor"

        self.logger.info(f"Starting thread: {module}")

        # Import module
        try:
            parts = module.rsplit('.', 1)
            if len(parts) == 2:
                module_path, func_name = parts
                imported_module = __import__(module_path, fromlist=[func_name])
                func = getattr(imported_module, func_name)
            else:
                raise ValueError(f"Invalid module format: {module}")
        except Exception as e:
            raise RuntimeError(f"Failed to import module '{module}': {e}")

        # Create and start thread
        thread = threading.Thread(
            target=func,
            name=module_name,
            daemon=True,
            kwargs={'config': config}
        )
        thread.start()

        # Store thread
        self.threads[module_name] = thread

        self.logger.info(f"Thread started: {module_name}")

        return {
            'pid': os.getpid(),  # Use daemon PID (threads share process)
            'process': None,
            'thread': thread,
            'type': 'thread'
        }

    async def stop_module(self, module_name: str, process_or_thread: Any, graceful: bool = True):
        """
        Stop a module

        Args:
            module_name: Module name
            process_or_thread: Process or thread object
            graceful: If True, send SIGTERM first, then SIGKILL after timeout
        """
        self.logger.info(f"Stopping module '{module_name}' (graceful={graceful})")

        # Check if it's a thread
        if module_name in self.threads:
            thread = self.threads[module_name]
            self.logger.info(f"Cannot forcefully stop thread '{module_name}' (daemon thread will exit with process)")
            del self.threads[module_name]
            return

        # It's a process
        if module_name not in self.processes:
            self.logger.warning(f"Module '{module_name}' process not found")
            return

        process = self.processes[module_name]

        try:
            if graceful:
                # Send SIGTERM
                self.logger.info(f"Sending SIGTERM to {module_name} (PID: {process.pid})")
                process.terminate()

                # Wait up to 3 seconds for graceful shutdown
                try:
                    process.wait(timeout=3)
                    self.logger.info(f"Module '{module_name}' stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Module '{module_name}' did not stop within 3s, killing...")
                    process.kill()
                    process.wait(timeout=2)
                    self.logger.info(f"Module '{module_name}' killed")

            else:
                # Force kill
                self.logger.info(f"Killing module '{module_name}' (PID: {process.pid})")
                process.kill()
                process.wait(timeout=2)
                self.logger.info(f"Module '{module_name}' killed")

        except Exception as e:
            self.logger.error(f"Error stopping module '{module_name}': {e}")

        finally:
            # Cleanup
            if module_name in self.processes:
                del self.processes[module_name]

            # Remove PID file
            pid_file = self.runtime_dir / f"{module_name}.pid"
            if pid_file.exists():
                pid_file.unlink()

    def is_module_running(self, module_name: str) -> bool:
        """Check if module is running"""
        # Check threads
        if module_name in self.threads:
            thread = self.threads[module_name]
            return thread.is_alive()

        # Check processes
        if module_name in self.processes:
            process = self.processes[module_name]
            return process.poll() is None

        return False

    def get_module_pid(self, module_name: str) -> Optional[int]:
        """Get module PID"""
        if module_name in self.threads:
            return os.getpid()  # Threads share daemon PID

        if module_name in self.processes:
            return self.processes[module_name].pid

        return None

    async def check_health(self, module_name: str, config: Dict) -> bool:
        """
        Check module health via health check endpoint

        Args:
            module_name: Module name
            config: Module config (contains healthcheck_endpoint)

        Returns:
            True if healthy, False otherwise
        """
        # Only for HTTP-based modules
        if config.get('type') not in ['uvicorn', 'flask']:
            # For python/thread, just check if process/thread is alive
            return self.is_module_running(module_name)

        # HTTP health check
        healthcheck_endpoint = config.get('healthcheck_endpoint')
        if not healthcheck_endpoint:
            return self.is_module_running(module_name)

        # Build health check URL
        host = config.get('host', '127.0.0.1')
        port = config.get('port', 8000)
        url = f"http://{host}:{port}{healthcheck_endpoint}"

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.warning(f"Health check failed for '{module_name}': {e}")
            return False
