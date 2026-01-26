#!/usr/bin/env python3
"""
engines/websocket_engine.py
SuperBot - WebSocket Manager
Author: SuperBot Team
Date: 2025-10-16
Versiyon: 1.0.0

WebSocket Manager - Manages Binance WebSocket connections.

Features:
- Multi-stream WebSocket
- Auto-reconnect
- Heartbeat/ping-pong
- Message buffer
- Symbol subscription
- Stream multiplexing

Usage:
    from engines.websocket_manager import WebSocketEngine
    
    ws = WebSocketEngine(config={...}, event_bus=event_bus)
    await ws.start()
    await ws.subscribe("BTCUSDT", ["trade", "kline_1m"])

Dependencies:
    - websockets
    - python-binance
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import websockets
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


@dataclass
class StreamInfo:
    """WebSocket stream bilgisi"""
    stream_id: str
    url: str
    symbols: List[str]
    channels: List[str]
    websocket: Optional[Any] = None
    connected: bool = False
    last_message: Optional[datetime] = None
    message_count: int = 0
    reconnect_count: int = 0


class WebSocketEngine:
    """
    WebSocket Manager
    
    Manages Binance WebSocket connections.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[Any] = None
    ):
        """
        Initialize WebSocketEngine
        
        Args:
            config: WebSocket configuration
            event_bus: EventBus instance
        """
        self.config = config or {}
        self.event_bus = event_bus
        
        # WebSocket settings
        # Default: FUTURES production (fallback if not in config)
        self.base_url = self.config.get("base_url", "wss://fstream.binance.com")  # FUTURES production (USDT-M)
        # OLD SPOT production: "wss://stream.binance.com:9443"
        self.testnet = self.config.get("testnet", True)
        if self.testnet:
            # OLD SPOT URLs (commented for reference):
            # self.base_url = "wss://testnet.binance.vision"  # SPOT testnet

            # NEW FUTURES URLs:
            self.base_url = "wss://fstream.binancefuture.com"  # FUTURES testnet (USDT-M)
        
        self.ping_interval = self.config.get("ping_interval", 180)  # 3 minutes
        self.ping_timeout = self.config.get("ping_timeout", 10)
        self.reconnect_delay = self.config.get("reconnect_delay", 5)
        self.max_reconnect_attempts = self.config.get("max_reconnect_attempts", 10)
        
        # Message buffer
        self.buffer_size = self.config.get("buffer_size", 1000)
        self.message_buffer: deque = deque(maxlen=self.buffer_size)
        
        # Streams
        self.streams: Dict[str, StreamInfo] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Stats
        self.stats = {
            "total_messages": 0,
            "total_reconnects": 0,
            "total_errors": 0,
            "active_streams": 0
        }
        
        logger.info(f"WebSocketEngine started ({'testnet' if self.testnet else 'production'})")
    
    async def start(self):
        """Initialize the WebSocket manager"""
        try:
            self.running = True
            logger.info("✅ WebSocket manager started")
            
        except Exception as e:
            logger.error(f"❌ WebSocket manager startup error: {e}")
            raise
    
    async def stop(self):
        """Stops the WebSocket manager"""
        try:
            self.running = False

            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Close all streams
            for stream_id in list(self.streams.keys()):
                await self._close_stream(stream_id)

            # Short wait for WebSocket cleanup
            await asyncio.sleep(0.25)

            logger.info("🛑 WebSocket manager durduruldu")

        except Exception as e:
            logger.error(f"❌ WebSocket manager shutdown error: {e}")
    
    async def subscribe(
        self,
        symbols: List[str],
        channels: List[str]
    ) -> str:
        """
        Symbol'lere abone ol
        
        Args:
            symbols: List of symbols (e.g., ["BTCUSDT", "ETHUSDT"])
            channels: List of channels (e.g., ["trade", "kline_1m"])
            
        Returns:
            str: Stream ID
        """
        try:
            # Create stream ID
            stream_id = f"stream_{int(time.time() * 1000)}"
            
            # Create stream URL
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                for channel in channels:
                    if channel.startswith("kline"):
                        # Kline: btcusdt@kline_1m
                        streams.append(f"{symbol_lower}@{channel}")
                    else:
                        # Trade, depth, etc: btcusdt@trade
                        streams.append(f"{symbol_lower}@{channel}")
            
            stream_path = "/".join(streams)
            url = f"{self.base_url}/stream?streams={stream_path}"
            
            # Stream info
            stream_info = StreamInfo(
                stream_id=stream_id,
                url=url,
                symbols=symbols,
                channels=channels
            )
            
            self.streams[stream_id] = stream_info
            
            # Start the WebSocket connection
            task = asyncio.create_task(self._stream_handler(stream_id))
            self.tasks.append(task)
            
            self.stats["active_streams"] = len(self.streams)
            
            logger.info(f"✅ Subscribe: {symbols} - {channels} (stream_id: {stream_id})")
            return stream_id
            
        except Exception as e:
            logger.error(f"❌ Subscription error: {e}")
            raise
    
    async def unsubscribe(self, stream_id: str):
        """
        Cancel the stream subscription.
        
        Args:
            stream_id: Stream ID
        """
        try:
            if stream_id not in self.streams:
                logger.warning(f"⚠️ Stream not found: {stream_id}")
                return
            
            await self._close_stream(stream_id)
            
            self.stats["active_streams"] = len(self.streams)
            
            logger.info(f"✅ Unsubscribe: {stream_id}")
            
        except Exception as e:
            logger.error(f"❌ Unsubscribe error: {e}")
    
    async def _stream_handler(self, stream_id: str):
        """
        Stream handler - Listen to WebSocket messages.

        Args:
            stream_id: Stream ID
        """
        stream_info = self.streams[stream_id]
        reconnect_count = 0

        while self.running and stream_id in self.streams:
            try:
                # Open WebSocket connection
                async with websockets.connect(
                    stream_info.url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=10,
                    open_timeout=30  # 15 -> 30 seconds (to prevent timeouts when the network is slow)
                ) as websocket:
                    stream_info.websocket = websocket
                    stream_info.connected = True
                    reconnect_count = 0

                    logger.info(f"🔗 WebSocket connected: {len(stream_info.symbols)} symbols")

                    # Listen for messages
                    async for message in websocket:
                        if not self.running:
                            break

                        await self._process_message(stream_id, message)
                        
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  WebSocket timeout: {stream_id}")
                stream_info.connected = False
                reconnect_count += 1

                # Timeout durumunda reconnect
                if reconnect_count < self.max_reconnect_attempts:
                    logger.info(f"🔄 Reconnecting ({reconnect_count}/{self.max_reconnect_attempts})...")
                    await asyncio.sleep(self.reconnect_delay)
                    stream_info.reconnect_count += 1
                    self.stats["total_reconnects"] += 1
                else:
                    logger.error(f"❌ Maximum reconnect attempts exceeded: {stream_id}")
                    break

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"⚠️ WebSocket connection closed: {stream_id} - {e}")
                stream_info.connected = False
                reconnect_count += 1

                # Reconnect
                if reconnect_count < self.max_reconnect_attempts:
                    logger.info(f"🔄 Reconnecting ({reconnect_count}/{self.max_reconnect_attempts})...")
                    await asyncio.sleep(self.reconnect_delay)
                    stream_info.reconnect_count += 1
                    self.stats["total_reconnects"] += 1
                else:
                    logger.error(f"❌ Maximum reconnect attempts exceeded: {stream_id}")
                    break

            except Exception as e:
                # For other errors, detailed logs are only available in debug mode.
                logger.warning(f"⚠️ Stream error: {type(e).__name__}: {e}")
                if logger.level <= 10:  # DEBUG level
                    import traceback
                    logger.debug(f"   Traceback: {traceback.format_exc()}")

                stream_info.connected = False
                self.stats["total_errors"] += 1
                reconnect_count += 1

                # Try to reconnect
                if reconnect_count < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
                    stream_info.reconnect_count += 1
                    self.stats["total_reconnects"] += 1
                else:
                    logger.error(f"❌ Maximum reconnect attempts exceeded: {stream_id}")
                    break
        
        # Cleanup
        stream_info.connected = False
        logger.info(f"🛑 Stream handler finished: {stream_id}")
    
    async def _process_message(self, stream_id: str, message: str):
        """
        Process the WebSocket message.

        Args:
            stream_id: Stream ID
            message: JSON message
        """
        try:
            # JSON parse
            data = json.loads(message)

            # Update stream information
            if stream_id in self.streams:
                stream_info = self.streams[stream_id]
                stream_info.last_message = datetime.now()
                stream_info.message_count += 1

            # Update statistics
            self.stats["total_messages"] += 1

            # Log message details (DEBUG level - not spamming)
            stream = data.get("stream", "unknown")
            logger.debug(f"Received WebSocket message: {stream} (stream_id: {stream_id})")

            # Add to buffer
            self.message_buffer.append({
                "stream_id": stream_id,
                "timestamp": time.time(),
                "data": data
            })

            # Publish to the EventBus
            if self.event_bus:
                event_type = self._get_event_type(stream)

                if event_type:
                    # For kline events, extract candle data
                    event_data = data.get("data", data)

                    if event_type.startswith("candle."):
                        # Extract kline data from Binance format
                        kline = event_data.get("k", {})
                        # Extract symbol and timeframe from event_type
                        # event_type = "candle.BTCUSDT.5m"
                        parts = event_type.split(".")
                        symbol = parts[1] if len(parts) > 1 else ""  # BTCUSDT
                        timeframe = parts[2] if len(parts) > 2 else ""  # 5m
                        event_data = {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'timestamp': kline.get('t', 0),
                            'open': float(kline.get('o', 0)),
                            'high': float(kline.get('h', 0)),
                            'low': float(kline.get('l', 0)),
                            'close': float(kline.get('c', 0)),
                            'volume': float(kline.get('v', 0)),
                            'is_closed': kline.get('x', False)  # Candle closed?
                        }
                        #logger.info(f"Candle data: {event_data}")

                    # Publish to EventBus (no logging - too verbose)
                    self.event_bus.publish(
                        topic=event_type,
                        data=event_data,
                        source="WebSocketEngine"
                    )

            logger.debug(f"📨 Message processed: {stream_id}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            self.stats["total_errors"] += 1
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
            self.stats["total_errors"] += 1
    
    def _get_event_type(self, stream: str) -> Optional[str]:
        """
        Stream'den event type belirle

        Args:
            stream: Stream name (e.g., "btcusdt@kline_5m")

        Returns:
            str: Event type
                - Kline: "candle.BTCUSDT.5m"
                - Trade: "trade.BTCUSDT"
                - Depth: "depth.BTCUSDT"
        """
        try:
            parts = stream.split("@")
            if len(parts) != 2:
                return None

            symbol = parts[0].upper()
            channel = parts[1]

            if channel == "trade":
                return f"trade.{symbol}"
            elif channel.startswith("kline"):
                # Extract interval: "kline_5m" -> "5m"
                interval = channel.split("_")[1]
                return f"candle.{symbol}.{interval}"
            elif channel == "depth":
                return f"depth.{symbol}"
            else:
                # Generic format
                return f"{channel}.{symbol}"

        except Exception as e:
            logger.error(f"❌ Error determining event type: {e}")
            return None
    
    async def _close_stream(self, stream_id: str):
        """Close the stream"""
        if stream_id not in self.streams:
            return
        
        stream_info = self.streams[stream_id]
        
        # Close WebSocket
        if stream_info.websocket:
            try:
                await stream_info.websocket.close()
            except:
                pass
        
        # Delete the stream
        del self.streams[stream_id]
        
        logger.info(f"❌ Stream closed: {stream_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns WebSocket statistics"""
        return {
            **self.stats,
            "buffer_size": len(self.message_buffer),
            "streams": {
                stream_id: {
                    "symbols": info.symbols,
                    "channels": info.channels,
                    "connected": info.connected,
                    "message_count": info.message_count,
                    "reconnect_count": info.reconnect_count
                }
                for stream_id, info in self.streams.items()
            }
        }


# Test
if __name__ == "__main__":
    from core.event_bus import EventBus
    
    async def test():
        print("=" * 60)
        print("🧪 WebSocketEngine Test")
        print("=" * 60)
        
        # Create EventBus
        event_bus = EventBus()
        
        # Event handler
        def on_trade(event):
            print(f"   📊 Trade: {event.data}")
        
        event_bus.subscribe("price.*.trade", on_trade)
        
        # WebSocket manager
        ws = WebSocketEngine(
            config={"testnet": True},
            event_bus=event_bus
        )
        
        await ws.start()
        
        # Subscribe
        stream_id = await ws.subscribe(
            symbols=["BTCUSDT"],
            channels=["trade"]
        )
        
        print(f"\n✅ Subscribed: {stream_id}")
        print("⏳ 10 saniye mesaj dinleniyor...\n")
        
        # 10 saniye bekle
        await asyncio.sleep(10)
        
        # Stats
        stats = ws.get_stats()
        print(f"\n📊 Stats:")
        print(f"   Total Messages: {stats['total_messages']}")
        print(f"   Active Streams: {stats['active_streams']}")
        
        # Stop
        await ws.stop()
        
        print("\n✅ Test completed!")
        print("=" * 60)
    
    asyncio.run(test())