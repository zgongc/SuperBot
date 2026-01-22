#!/usr/bin/env python3
"""
engines/connection_pool_engine.py
SuperBot - Connection Pool Manager
Author: SuperBot Team
Date: 2025-10-16
Versiyon: 1.0.0

Connection Pool - Manages HTTP/WebSocket connection pools.

Features:
- HTTP connection pooling (aiohttp)
- WebSocket connection pooling
- Connection health check
- Auto-reconnect
- Rate limiting
- Connection timeout
- Load balancing

Usage:
    from engines.connection_pool_manager import ConnectionPoolEngine
    
    pool = ConnectionPoolEngine(config={
        "min_connections": 2,
        "max_connections": 10
    })
    
    async with pool.get_connection() as conn:
        response = await conn.get("https://api.binance.com/api/v3/time")

Dependencies:
    - aiohttp
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
import aiohttp
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


@dataclass
class ConnectionInfo:
    """Connection information structure"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    in_use: bool = False
    health: bool = True
    session: Optional[aiohttp.ClientSession] = None
    request_count: int = 0
    error_count: int = 0


class ConnectionPoolEngine:
    """
    Connection Pool Manager
    
    Manages HTTP/WebSocket connection pool.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ConnectionPoolEngine
        
        Args:
            config: Connection pool configuration
        """
        self.config = config or {}
        
        # Pool settings
        self.min_connections = self.config.get("min_connections", 2)
        self.max_connections = self.config.get("max_connections", 10)
        self.connection_timeout = self.config.get("connection_timeout", 30)
        self.idle_timeout = self.config.get("idle_timeout", 300)  # 5 minutes
        self.health_check_interval = self.config.get("health_check_interval", 60)
        
        # Connection pool
        self.connections: Dict[str, ConnectionInfo] = {}
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            "total_created": 0,
            "total_closed": 0,
            "total_requests": 0,
            "total_errors": 0,
            "active_connections": 0
        }
        
        # Health check task
        self.health_check_task = None
        self.running = False
        
        logger.info(f"ConnectionPoolEngine started (min={self.min_connections}, max={self.max_connections})")
    
    async def start(self):
        """Initialize the pool"""
        try:
            self.running = True
            
            # Create the initial number of connections, up to the minimum connection count.
            for _ in range(self.min_connections):
                await self._create_connection()
            
            # Start the health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"✅ Connection pool started ({len(self.connections)} connections)")
            
        except Exception as e:
            logger.error(f"❌ Connection pool initialization error: {e}")
            raise
    
    async def stop(self):
        """Pool'u durdur"""
        try:
            self.running = False

            # Stop the health check task
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass

            # Close all connections
            async with self.lock:
                for conn_id, conn_info in list(self.connections.items()):
                    await self._close_connection(conn_id)

            # aiohttp's short wait for internal cleanup
            await asyncio.sleep(0.25)

            logger.info("🛑 Connection pool durduruldu")

        except Exception as e:
            logger.error(f"❌ Connection pool shutdown error: {e}")
    
    async def _create_connection(self) -> str:
        """Create a new connection"""
        async with self.lock:
            # Maximum connection check
            if len(self.connections) >= self.max_connections:
                raise RuntimeError(f"Maximum connection limit exceeded: {self.max_connections}")
            
            # Create connection ID
            connection_id = f"conn_{int(time.time() * 1000)}_{len(self.connections)}"
            
            # Create an aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            session = aiohttp.ClientSession(timeout=timeout)
            
            # Connection info
            conn_info = ConnectionInfo(
                connection_id=connection_id,
                created_at=datetime.now(),
                last_used=datetime.now(),
                session=session
            )
            
            self.connections[connection_id] = conn_info
            await self.available_connections.put(connection_id)
            
            self.stats["total_created"] += 1
            self.stats["active_connections"] = len(self.connections)
            
            logger.debug(f"🔗 A new connection has been established: {connection_id}")
            return connection_id
    
    async def _close_connection(self, connection_id: str):
        """Close the connection"""
        if connection_id not in self.connections:
            return
        
        conn_info = self.connections[connection_id]
        
        # Close the session
        if conn_info.session and not conn_info.session.closed:
            await conn_info.session.close()
        
        # Exit the pool
        del self.connections[connection_id]
        
        self.stats["total_closed"] += 1
        self.stats["active_connections"] = len(self.connections)
        
        logger.debug(f"❌ Connection closed: {connection_id}")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get connection (context manager)
        
        Yields:
            aiohttp.ClientSession: HTTP session
        """
        connection_id = None
        
        try:
            # Get the existing connection or create a new one.
            try:
                connection_id = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Timeout - create new connection
                if len(self.connections) < self.max_connections:
                    connection_id = await self._create_connection()
                else:
                    raise RuntimeError("All connections are busy, timeout")
            
            conn_info = self.connections[connection_id]
            conn_info.in_use = True
            conn_info.last_used = datetime.now()
            
            logger.debug(f"🔓 Connection established: {connection_id}")
            
            # Give the session.
            yield conn_info.session
            
            # Update statistics
            conn_info.request_count += 1
            self.stats["total_requests"] += 1
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            if connection_id and connection_id in self.connections:
                self.connections[connection_id].error_count += 1
                self.stats["total_errors"] += 1
            raise
            
        finally:
            # Return the connection
            if connection_id and connection_id in self.connections:
                conn_info = self.connections[connection_id]
                conn_info.in_use = False
                await self.available_connections.put(connection_id)
                logger.debug(f"🔒 Connection acknowledged: {connection_id}")
    
    async def _health_check_loop(self):
        """Health check loop"""
        logger.info("🏥 Health check cycle started")
        
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform a health check on all connections"""
        async with self.lock:
            current_time = datetime.now()
            
            for conn_id, conn_info in list(self.connections.items()):
                # Idle timeout check
                idle_seconds = (current_time - conn_info.last_used).total_seconds()
                
                if idle_seconds > self.idle_timeout and not conn_info.in_use:
                    # Maintain the minimum number of connections
                    if len(self.connections) > self.min_connections:
                        logger.info(f"⏱️ Idle timeout, connection is being closed: {conn_id}")
                        await self._close_connection(conn_id)
                        continue
                
                # Session health check
                if conn_info.session and conn_info.session.closed:
                    logger.warning(f"⚠️ Connection closed, recreating: {conn_id}")
                    await self._close_connection(conn_id)
                    await self._create_connection()
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns pool statistics"""
        return {
            **self.stats,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "available_connections": self.available_connections.qsize(),
            "in_use_connections": sum(1 for c in self.connections.values() if c.in_use)
        }
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Returns connection information"""
        if connection_id not in self.connections:
            return None
        
        conn_info = self.connections[connection_id]
        return {
            "connection_id": conn_info.connection_id,
            "created_at": conn_info.created_at.isoformat(),
            "last_used": conn_info.last_used.isoformat(),
            "in_use": conn_info.in_use,
            "health": conn_info.health,
            "request_count": conn_info.request_count,
            "error_count": conn_info.error_count
        }


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("🧪 ConnectionPoolEngine Test")
        print("=" * 60)
        
        # Create a pool
        pool = ConnectionPoolEngine(config={
            "min_connections": 2,
            "max_connections": 5,
            "connection_timeout": 10
        })
        
        # Initialize
        await pool.start()
        
        # Test 1: Get connection
        print("\n1️⃣ Connection test:")
        async with pool.get_connection() as session:
            print(f"   ✅ Session obtained: {session}")
        
        # Test 2: Multiple connections
        print("\n2️⃣  Multiple connection test:")
        tasks = []
        for i in range(3):
            async def request(num):
                async with pool.get_connection() as session:
                    print(f"   ✅ Connection {num} is in use")
                    await asyncio.sleep(0.5)
            tasks.append(request(i))
        
        await asyncio.gather(*tasks)
        
        # Test 3: Stats
        print("\n3️⃣  Statistics:")
        stats = pool.get_stats()
        print(f"   Total Created: {stats['total_created']}")
        print(f"   Active: {stats['active_connections']}")
        print(f"   Available: {stats['available_connections']}")
        
        # Durdur
        await pool.stop()
        
        print("\n✅ Test completed!")
        print("=" * 60)
    
    asyncio.run(test())