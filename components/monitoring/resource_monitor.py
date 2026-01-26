#!/usr/bin/env python3
"""
engines/resource_monitor.py
SuperBot - Resource Monitor
Author: SuperBot Team
Date: 2025-10-16
Versiyon: 1.0.0

Resource Monitor - Monitors system resource usage.

Features:
- CPU usage
- RAM usage
- Disk usage
- Network I/O
- Process monitoring
- Threshold alerts
- EventBus entegrasyonu

Usage:
    from engines.resource_monitor import ResourceMonitor
    
    monitor = ResourceMonitor(config={...}, event_bus=event_bus)
    await monitor.start()
    
    # Stats
    stats = monitor.get_stats()

Dependencies:
    - psutil
"""

import asyncio
import psutil
import os
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


@dataclass
class ResourceSnapshot:
    """Resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    disk_gb: float
    network_sent_mb: float
    network_recv_mb: float


class ResourceMonitor:
    """
    Resource Monitor - Sistem kaynak izleme
    
    Monitors CPU, RAM, Disk, and Network usage.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[Any] = None
    ):
        """
        Initialize ResourceMonitor
        
        Args:
            config: Monitor configuration
            event_bus: EventBus instance
        """
        self.config = config or {}
        self.event_bus = event_bus
        
        # Monitoring settings
        self.monitor_interval = self.config.get("monitor_interval", 5)  # 5 saniye
        self.history_size = self.config.get("history_size", 100)
        
        # Threshold'lar
        self.cpu_threshold = self.config.get("cpu_threshold", 80)  # %80
        self.memory_threshold = self.config.get("memory_threshold", 80)  # %80
        self.disk_threshold = self.config.get("disk_threshold", 90)  # %90
        
        # History
        self.history: List[ResourceSnapshot] = []
        
        # Process info
        self.process = psutil.Process(os.getpid())
        
        # Monitoring task
        self.monitoring_task = None
        self.running = False
        
        # Stats
        self.stats = {
            "total_snapshots": 0,
            "cpu_alerts": 0,
            "memory_alerts": 0,
            "disk_alerts": 0
        }
        
        logger.info("ResourceMonitor started")
    
    async def start(self):
        """Start resource monitoring"""
        try:
            self.running = True
            
            # Start the monitoring task
            self.monitoring_task = asyncio.create_task(self._monitor_loop())
            
            logger.info("✅ Resource monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Error starting resource monitoring: {e}")
            raise
    
    async def stop(self):
        """Resource monitoring'i durdur"""
        try:
            self.running = False
            
            # Stop the monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("🛑 Resource monitoring durduruldu")
            
        except Exception as e:
            logger.error(f"❌ Error stopping resource monitoring: {e}")
    
    async def _monitor_loop(self):
        """Monitoring loop"""
        logger.info("📊 Resource monitoring cycle started")
        
        while self.running:
            try:
                await asyncio.sleep(self.monitor_interval)
                await self._collect_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
    
    async def _collect_metrics(self):
        """Kaynak metriklerini topla"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
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
            
            # Create snapshot
            snapshot = ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_percent=disk_percent,
                disk_gb=disk_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )
            
            # History'e ekle
            self.history.append(snapshot)
            if len(self.history) > self.history_size:
                self.history.pop(0)
            
            self.stats["total_snapshots"] += 1
            
            # Threshold check
            await self._check_thresholds(snapshot)
            
            logger.debug(f"📊 Metrics collected: CPU={cpu_percent:.1f}% RAM={memory_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Metric collection error: {e}")
    
    async def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Perform threshold check"""
        # CPU
        if snapshot.cpu_percent > self.cpu_threshold:
            self.stats["cpu_alerts"] += 1
            logger.warning(f"⚠️ CPU usage is high: {snapshot.cpu_percent:.1f}%")
            
            if self.event_bus:
                self.event_bus.publish(
                    topic="system.resource.cpu.high",
                    data={
                        "cpu_percent": snapshot.cpu_percent,
                        "threshold": self.cpu_threshold
                    },
                    source="ResourceMonitor"
                )
        
        # Memory
        if snapshot.memory_percent > self.memory_threshold:
            self.stats["memory_alerts"] += 1
            logger.warning(f"⚠️ High memory usage: {snapshot.memory_percent:.1f}%")
            
            if self.event_bus:
                self.event_bus.publish(
                    topic="system.resource.memory.high",
                    data={
                        "memory_percent": snapshot.memory_percent,
                        "memory_mb": snapshot.memory_mb,
                        "threshold": self.memory_threshold
                    },
                    source="ResourceMonitor"
                )
        
        # Disk
        if snapshot.disk_percent > self.disk_threshold:
            self.stats["disk_alerts"] += 1
            logger.warning(f"⚠️ Disk usage is high: {snapshot.disk_percent:.1f}%")
            
            if self.event_bus:
                self.event_bus.publish(
                    topic="system.resource.disk.high",
                    data={
                        "disk_percent": snapshot.disk_percent,
                        "disk_gb": snapshot.disk_gb,
                        "threshold": self.disk_threshold
                    },
                    source="ResourceMonitor"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns resource statistics"""
        if not self.history:
            return {
                **self.stats,
                "current": None
            }
        
        # Last snapshot
        current = self.history[-1]
        
        # Averages (last 10 snapshots)
        recent = self.history[-10:]
        avg_cpu = sum(s.cpu_percent for s in recent) / len(recent)
        avg_memory = sum(s.memory_percent for s in recent) / len(recent)
        
        return {
            **self.stats,
            "current": {
                "timestamp": current.timestamp.isoformat(),
                "cpu_percent": f"{current.cpu_percent:.1f}%",
                "memory_percent": f"{current.memory_percent:.1f}%",
                "memory_mb": f"{current.memory_mb:.1f} MB",
                "disk_percent": f"{current.disk_percent:.1f}%",
                "disk_gb": f"{current.disk_gb:.1f} GB",
                "network_sent_mb": f"{current.network_sent_mb:.1f} MB",
                "network_recv_mb": f"{current.network_recv_mb:.1f} MB"
            },
            "averages": {
                "cpu_percent": f"{avg_cpu:.1f}%",
                "memory_percent": f"{avg_memory:.1f}%"
            },
            "thresholds": {
                "cpu": f"{self.cpu_threshold}%",
                "memory": f"{self.memory_threshold}%",
                "disk": f"{self.disk_threshold}%"
            }
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """Return process information"""
        try:
            with self.process.oneshot():
                return {
                    "pid": self.process.pid,
                    "name": self.process.name(),
                    "cpu_percent": f"{self.process.cpu_percent():.1f}%",
                    "memory_mb": f"{self.process.memory_info().rss / (1024 * 1024):.1f} MB",
                    "num_threads": self.process.num_threads(),
                    "create_time": datetime.fromtimestamp(self.process.create_time()).isoformat()
                }
        except Exception as e:
            logger.error(f"❌ Process info error: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Resource monitor health check"""
        try:
            # Can CPU and memory information be retrieved?
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return cpu >= 0 and memory.percent >= 0
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("🧪 ResourceMonitor Test")
        print("=" * 60)
        
        # Create monitor
        monitor = ResourceMonitor(config={
            "monitor_interval": 2,
            "cpu_threshold": 50,
            "memory_threshold": 50
        })
        
        # Initialize
        await monitor.start()
        
        print("\n⏳ 10 saniye monitoring...")
        await asyncio.sleep(10)
        
        # Stats
        print("\n📊 Stats:")
        stats = monitor.get_stats()
        print(f"   Total Snapshots: {stats['total_snapshots']}")
        print(f"   Current CPU: {stats['current']['cpu_percent']}")
        print(f"   Current Memory: {stats['current']['memory_percent']}")
        print(f"   Avg CPU: {stats['averages']['cpu_percent']}")
        
        # Process info
        print("\n🔍 Process Info:")
        proc_info = monitor.get_process_info()
        print(f"   PID: {proc_info['pid']}")
        print(f"   CPU: {proc_info['cpu_percent']}")
        print(f"   Memory: {proc_info['memory_mb']}")
        
        # Health check
        print(f"\n✅ Health Check: {'OK' if monitor.health_check() else 'FAIL'}")
        
        # Durdur
        await monitor.stop()
        
        print("\n✅ Test completed!")
        print("=" * 60)
    
    asyncio.run(test())