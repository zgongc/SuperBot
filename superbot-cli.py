#!/usr/bin/env python3
"""
SuperBot Unified CLI Tool
==========================

Command-line interface for controlling SuperBot Daemon.
All operations communicate with the daemon via IPC/RPC.

Usage:
    superbot-cli daemon start           # Start daemon
    superbot-cli daemon stop            # Stop daemon
    superbot-cli daemon status          # Show status

    superbot-cli start trading          # Start trading module
    superbot-cli stop trading           # Stop trading module
    superbot-cli restart webui          # Restart WebUI

    superbot-cli status                 # Show all status
    superbot-cli logs --follow          # Stream logs
    superbot-cli positions              # Show positions
    superbot-cli balance                # Show balance

Author: SuperBot Team
Date: 2025-11-07
Version: 1.0.0
"""

import sys
import os
import socket
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
import yaml

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Console for rich output
console = Console()


def load_ipc_config() -> Dict[str, Any]:
    """Load IPC config from daemon.yaml"""
    config_path = ROOT_DIR / "config" / "daemon.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('daemon', {}).get('ipc', {})
    except Exception:
        return {}


class IPCClient:
    """IPC Client for communicating with SuperBot Daemon"""

    def __init__(self, socket_path: str = "/tmp/superbot.sock", timeout: int = 30):
        # Load config from daemon.yaml
        ipc_config = load_ipc_config()

        # Windows: use TCP, Unix: use socket
        if sys.platform == 'win32':
            self.use_tcp = True
            self.host = ipc_config.get('tcp_host', '127.0.0.1')
            self.port = ipc_config.get('tcp_port', 9999)
            self.socket_path = f"{self.host}:{self.port}"
        else:
            self.use_tcp = False
            self.socket_path = ipc_config.get('socket_path', socket_path)
            self.host = None
            self.port = None

        self.timeout = timeout
        self.request_id = 0

    def is_daemon_running(self) -> bool:
        """Check if daemon is running"""
        if self.use_tcp:
            # Windows: check if port is open
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                return result == 0
            except:
                return False
        else:
            # Unix: check if socket file exists
            return os.path.exists(self.socket_path)

    def call(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make JSON-RPC call to daemon"""
        if not self.is_daemon_running():
            raise ConnectionError("Daemon is not running. Start it with: superbot-cli daemon start")

        # Build JSON-RPC request
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }

        try:
            # Connect to socket (TCP on Windows, Unix socket on Linux/Mac)
            if self.use_tcp:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
            else:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect(self.socket_path)

            # Send request
            request_data = json.dumps(request).encode('utf-8')
            sock.sendall(request_data + b'\n')

            # Receive response
            response_data = b''
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b'\n' in response_data:
                    break

            sock.close()

            # Parse response
            response = json.loads(response_data.decode('utf-8'))

            if 'error' in response:
                raise Exception(response['error']['message'])

            return response.get('result', {})

        except socket.timeout:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to daemon: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid response from daemon: {e}")


# IPC Client instance
ipc = IPCClient()


# ============================================================
# CLI GROUPS & COMMANDS
# ============================================================

@click.group()
@click.version_option(version='1.0.0', prog_name='SuperBot CLI')
def cli():
    """SuperBot Unified CLI - Control your trading bot from command line"""
    pass


# ============================================================
# DAEMON COMMANDS
# ============================================================

@cli.group()
def daemon():
    """Daemon management commands"""
    pass


@daemon.command()
@click.option('--config', default='config', help='Config directory path')
def start(config):
    """Start SuperBot Daemon"""
    console.print("[bold cyan]> Starting SuperBot Daemon...[/bold cyan]")

    # Check if already running
    if ipc.is_daemon_running():
        console.print("[bold red]ERROR Daemon is already running![/bold red]")
        console.print(f"   Socket: {ipc.socket_path}")
        sys.exit(1)

    # Start daemon process
    daemon_script = ROOT_DIR / "superbot.py"

    try:
        # Start in background
        process = subprocess.Popen(
            [sys.executable, str(daemon_script), '--config', config],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        # Wait a bit for daemon to start
        time.sleep(2)

        # Check if started successfully
        if ipc.is_daemon_running():
            console.print("[bold green]OK Daemon started successfully![/bold green]")

            # Get status
            try:
                status = ipc.call('daemon.status')
                console.print(f"   PID: {status['pid']}")
                console.print(f"   Socket: {ipc.socket_path}")
            except:
                pass
        else:
            console.print("[bold red]ERROR Failed to start daemon[/bold red]")
            console.print("\nDaemon output:")
            console.print(process.stdout.read().decode())
            console.print(process.stderr.read().decode())
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]ERROR Error starting daemon: {e}[/bold red]")
        sys.exit(1)


@daemon.command()
def stop():
    """Stop SuperBot Daemon"""
    console.print("[bold cyan]> Stopping SuperBot Daemon...[/bold cyan]")

    try:
        result = ipc.call('daemon.stop')
        console.print("[bold green]OK Daemon stopping...[/bold green]")

        # Wait for shutdown
        for i in range(30):
            time.sleep(1)
            if not ipc.is_daemon_running():
                console.print("[bold green]OK Daemon stopped successfully![/bold green]")
                break
        else:
            console.print("[bold yellow]WARNING Daemon did not stop within 30s[/bold yellow]")

    except ConnectionError:
        console.print("[bold yellow]WARNING Daemon is not running[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@daemon.command()
def restart():
    """Restart SuperBot Daemon"""
    console.print("[bold cyan] Restarting SuperBot Daemon...[/bold cyan]")

    # Stop
    if ipc.is_daemon_running():
        try:
            ipc.call('daemon.stop')
            # Wait for shutdown
            for i in range(30):
                time.sleep(1)
                if not ipc.is_daemon_running():
                    break
        except:
            pass

    # Start
    time.sleep(2)
    start.invoke(click.Context(start))


@daemon.command(name='status')
def daemon_status():
    """Show daemon status"""
    try:
        status = ipc.call('daemon.status')

        # Create status table
        table = Table(title="SuperBot Daemon Status", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Status", "[green]RUNNING[/green]" if status['status'] == 'running' else "[red]STOPPED[/red]")
        table.add_row("PID", str(status['pid']))
        table.add_row("Uptime", f"{status['uptime'] // 3600}h {(status['uptime'] % 3600) // 60}m")
        table.add_row("Socket", ipc.socket_path)

        console.print(table)

        # Module status
        if status['modules']:
            console.print("\n[bold cyan]Modules:[/bold cyan]")
            module_table = Table(box=box.SIMPLE)
            module_table.add_column("Module", style="cyan")
            module_table.add_column("Status", style="green")
            module_table.add_column("PID", style="yellow")
            module_table.add_column("Uptime", style="blue")

            for name, module in status['modules'].items():
                status_color = "green" if module['status'] == 'running' else "red"
                status_text = f"[{status_color}]{module['status'].upper()}[/{status_color}]"
                pid_text = str(module['pid']) if module['pid'] else "-"
                uptime_text = f"{module['uptime'] // 60}m" if module['uptime'] else "-"

                module_table.add_row(name, status_text, pid_text, uptime_text)

            console.print(module_table)

    except ConnectionError:
        console.print("[bold red]ERROR Daemon is not running[/bold red]")
        console.print("   Start it with: [cyan]superbot-cli daemon start[/cyan]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


# ============================================================
# MODULE COMMANDS
# ============================================================

@cli.command()
@click.argument('module')
@click.option('--mode', default=None, help='Trading mode (paper/live)')
@click.option('--strategy', default=None, help='Strategy name')
def start(module, mode, strategy):
    """Start a module"""
    console.print(f"[bold cyan]> Starting module: {module}[/bold cyan]")

    params = {}
    if mode:
        params['mode'] = mode
    if strategy:
        params['strategy'] = strategy

    try:
        result = ipc.call('module.start', {'module': module, 'params': params})

        if result['status'] == 'success':
            console.print(f"[bold green]OK Module '{module}' started![/bold green]")
            if result.get('pid'):
                console.print(f"   PID: {result['pid']}")
        else:
            console.print(f"[bold red]ERROR Failed to start module '{module}'[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('module')
@click.option('--force', is_flag=True, help='Force stop (not graceful)')
def stop(module, force):
    """Stop a module"""
    console.print(f"[bold cyan]> Stopping module: {module}[/bold cyan]")

    try:
        result = ipc.call('module.stop', {'module': module, 'graceful': not force})

        if result['status'] == 'success':
            console.print(f"[bold green]OK Module '{module}' stopped![/bold green]")
        else:
            console.print(f"[bold red]ERROR Failed to stop module '{module}'[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('module')
def restart(module):
    """Restart a module"""
    console.print(f"[bold cyan] Restarting module: {module}[/bold cyan]")

    try:
        result = ipc.call('module.restart', {'module': module})

        if result['status'] == 'success':
            console.print(f"[bold green]OK Module '{module}' restarted![/bold green]")
        else:
            console.print(f"[bold red]ERROR Failed to restart module '{module}'[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('module', required=False)
def status(module):
    """Show module status (or all if no module specified)"""
    if not module:
        # Show all status (same as daemon status)
        daemon_status.invoke(click.Context(daemon_status))
        return

    try:
        result = ipc.call('module.status', {'module': module})

        if result['status'] == 'success':
            table = Table(title=f" Module: {module}", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            status_color = "green" if result['state'] == 'running' else "red"
            table.add_row("Status", f"[{status_color}]{result['state'].upper()}[/{status_color}]")
            table.add_row("PID", str(result['pid']) if result['pid'] else "-")
            table.add_row("Uptime", f"{result['uptime'] // 60}m" if result['uptime'] else "-")
            table.add_row("Restarts", str(result['restart_count']))

            console.print(table)
        else:
            console.print(f"[bold red]ERROR Module '{module}' not found[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


# ============================================================
# MONITORING COMMANDS
# ============================================================

@cli.command()
def health():
    """Show system health"""
    console.print("[bold cyan] System Health Check[/bold cyan]\n")

    try:
        result = ipc.call('monitoring.health')

        if result['status'] == 'success':
            health = result['health']

            for check_name, check_result in health.items():
                status_icon = "OK" if check_result['status'] == 'healthy' else ""
                status_color = "green" if check_result['status'] == 'healthy' else "red"

                console.print(f"[{status_color}]{status_icon}[/{status_color}] {check_name}: [{status_color}]{check_result['status']}[/{status_color}]")

        else:
            console.print("[bold red]ERROR Health check failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def resources():
    """Show resource usage"""
    console.print("[bold cyan]📊 Resource Usage[/bold cyan]\n")

    try:
        result = ipc.call('monitoring.resources')

        if result['status'] == 'success':
            res = result['resources']

            # Check if using new ResourceMonitor format
            if 'system' in res:
                # System Resources
                system = res.get('system', {})
                if system:
                    table = Table(title="System Resources", box=box.ROUNDED)
                    table.add_column("Metric", style="cyan")
                    table.add_column("Current", style="green")
                    table.add_column("Threshold", style="yellow")

                    thresholds = res.get('thresholds', {})
                    table.add_row("CPU", system.get('cpu_percent', '-'), thresholds.get('cpu', '-'))
                    table.add_row("Memory", system.get('memory_percent', '-'), thresholds.get('memory', '-'))
                    table.add_row("Disk", system.get('disk_percent', '-'), thresholds.get('disk', '-'))
                    table.add_row("Network Sent", system.get('network_sent_mb', '-'), "-")
                    table.add_row("Network Recv", system.get('network_recv_mb', '-'), "-")

                    console.print(table)

                # Averages
                averages = res.get('averages', {})
                if averages:
                    console.print(f"\n[dim]Averages (last 10): CPU={averages.get('cpu_percent', '-')} Memory={averages.get('memory_percent', '-')}[/dim]")

                # Alerts
                alerts = res.get('alerts', {})
                if any(alerts.values()):
                    console.print(f"\n[yellow]⚠️  Alerts: CPU={alerts.get('cpu_alerts', 0)} Memory={alerts.get('memory_alerts', 0)} Disk={alerts.get('disk_alerts', 0)}[/yellow]")

                # Process Info
                proc = res.get('process', {})
                if proc:
                    console.print(f"\n[bold]Process Info:[/bold]")
                    console.print(f"  PID: {proc.get('pid', '-')} | CPU: {proc.get('cpu_percent', '-')} | Memory: {proc.get('memory_mb', '-')} | Threads: {proc.get('num_threads', '-')}")

            else:
                # Legacy format (fallback)
                table = Table(box=box.ROUNDED)
                table.add_column("Resource", style="cyan")
                table.add_column("Usage", style="green")

                table.add_row("CPU", f"{res.get('cpu_percent', 0):.1f}%")
                table.add_row("Memory", f"{res.get('memory_mb', 0):.1f} MB")
                table.add_row("Threads", str(res.get('threads', '-')))
                table.add_row("Connections", str(res.get('connections', '-')))
                table.add_row("Open Files", str(res.get('open_files', '-')))

                console.print(table)

        else:
            console.print("[bold red]ERROR Failed to get resources[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--module', '-m', default=None, help='Filter by module')
@click.option('--tail', '-n', default=50, help='Number of lines to show')
def logs(follow, module, tail):
    """Stream logs"""
    if follow:
        console.print("[bold yellow]WARNING Log streaming not yet implemented[/bold yellow]")
        console.print("   Use: tail -f data/logs/modules.log")
    else:
        console.print(f"[bold cyan] Last {tail} log lines[/bold cyan]")
        console.print("[bold yellow]WARNING Not yet implemented[/bold yellow]")


# ============================================================
# TRADING COMMANDS
# ============================================================

@cli.command()
def positions():
    """Show open positions"""
    console.print("[bold cyan] Open Positions[/bold cyan]\n")

    try:
        result = ipc.call('trading.positions')

        if result['status'] == 'not_implemented':
            console.print("[bold yellow]WARNING Not yet implemented[/bold yellow]")
            console.print("   Will be available in next version")
        else:
            # TODO: Display positions table
            pass

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def orders():
    """Show active orders"""
    console.print("[bold cyan] Active Orders[/bold cyan]\n")

    try:
        result = ipc.call('trading.orders')

        if result['status'] == 'not_implemented':
            console.print("[bold yellow]WARNING Not yet implemented[/bold yellow]")
            console.print("   Will be available in next version")
        else:
            # TODO: Display orders table
            pass

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def balance():
    """Show account balance"""
    console.print("[bold cyan] Account Balance[/bold cyan]\n")

    try:
        result = ipc.call('trading.balance')

        if result['status'] == 'not_implemented':
            console.print("[bold yellow]WARNING Not yet implemented[/bold yellow]")
            console.print("   Will be available in next version")
        else:
            # TODO: Display balance
            pass

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


# ============================================================
# CONFIG COMMANDS
# ============================================================

@cli.group()
def config():
    """Configuration management"""
    pass


@config.command()
def reload():
    """Reload configuration"""
    console.print("[bold cyan] Reloading configuration...[/bold cyan]")

    try:
        result = ipc.call('daemon.reload_config')

        if result['status'] == 'success':
            console.print("[bold green]OK Configuration reloaded![/bold green]")
        else:
            console.print(f"[bold red]ERROR {result['message']}[/bold red]")

    except Exception as e:
        console.print(f"[bold red]ERROR Error: {e}[/bold red]")
        sys.exit(1)


@config.command()
def show():
    """Show current configuration"""
    console.print("[bold yellow]WARNING Not yet implemented[/bold yellow]")


@config.command()
def validate():
    """Validate configuration files"""
    console.print("[bold yellow]WARNING Not yet implemented[/bold yellow]")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted[/bold yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/bold red]")
        sys.exit(1)
