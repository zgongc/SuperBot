"""
Monitoring Service for WebUI
============================

Provides system resource monitoring directly using ResourceMonitor.
Works independently - no daemon required.

Author: SuperBot Team
"""

import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ResourceSnapshot:
    """Resource usage snapshot"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    disk_gb: float
    network_sent_mb: float
    network_recv_mb: float


class MonitoringService:
    """
    Monitoring service for WebUI.

    Uses psutil directly - no daemon dependency.
    Keeps history for charts.
    """

    def __init__(self, history_size: int = 60):
        """
        Initialize monitoring service.

        Args:
            history_size: Number of snapshots to keep (default 60 = 1 hour at 1/min)
        """
        self.history_size = history_size
        self.history: List[ResourceSnapshot] = []

        # Thresholds
        self.cpu_threshold = 80
        self.memory_threshold = 80
        self.disk_threshold = 90

        # Alert counters
        self.alerts = {
            'cpu': 0,
            'memory': 0,
            'disk': 0
        }

    def collect_snapshot(self) -> ResourceSnapshot:
        """Collect current resource snapshot"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)

        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_gb = disk.used / (1024 * 1024 * 1024)

        # Network
        net_io = psutil.net_io_counters()
        network_sent_mb = net_io.bytes_sent / (1024 * 1024)
        network_recv_mb = net_io.bytes_recv / (1024 * 1024)

        snapshot = ResourceSnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_percent=round(cpu_percent, 1),
            memory_percent=round(memory_percent, 1),
            memory_mb=round(memory_mb, 1),
            disk_percent=round(disk_percent, 1),
            disk_gb=round(disk_gb, 1),
            network_sent_mb=round(network_sent_mb, 1),
            network_recv_mb=round(network_recv_mb, 1)
        )

        # Check thresholds
        if cpu_percent > self.cpu_threshold:
            self.alerts['cpu'] += 1
        if memory_percent > self.memory_threshold:
            self.alerts['memory'] += 1
        if disk_percent > self.disk_threshold:
            self.alerts['disk'] += 1

        # Add to history
        self.history.append(snapshot)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return snapshot

    def get_current(self) -> Dict[str, Any]:
        """Get current resource usage"""
        snapshot = self.collect_snapshot()

        return {
            'status': 'success',
            'data': {
                'current': asdict(snapshot),
                'thresholds': {
                    'cpu': self.cpu_threshold,
                    'memory': self.memory_threshold,
                    'disk': self.disk_threshold
                },
                'alerts': self.alerts
            }
        }

    def get_history(self, limit: int = 60) -> Dict[str, Any]:
        """Get resource history for charts"""
        # Collect fresh snapshot first
        self.collect_snapshot()

        history_data = [asdict(s) for s in self.history[-limit:]]

        return {
            'status': 'success',
            'data': {
                'history': history_data,
                'count': len(history_data)
            }
        }

    def get_process_info(self) -> Dict[str, Any]:
        """Get current process info"""
        try:
            process = psutil.Process()

            return {
                'status': 'success',
                'data': {
                    'pid': process.pid,
                    'name': process.name(),
                    'cpu_percent': round(process.cpu_percent(), 1),
                    'memory_mb': round(process.memory_info().rss / (1024 * 1024), 1),
                    'threads': process.num_threads(),
                    'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'status': 'success',
                'data': {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                    'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 1),
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    'platform': psutil.sys.platform
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def reset_alerts(self):
        """Reset alert counters"""
        self.alerts = {'cpu': 0, 'memory': 0, 'disk': 0}
