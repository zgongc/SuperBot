#!/usr/bin/env python3
"""
core/event_bus.py

SuperBot - Event Bus
Author: SuperBot Team
Date: 2025-10-16
Version: 1.0.0

Event Bus - Inter-module event-driven communication

Features:
- Pub/Sub pattern
- Topic-based routing
- Memory backend (default)
- Redis backend (optional)
- Event history
- Dead letter queue
- Async support

Usage:
    from core.event_bus import EventBus
    
    bus = EventBus()

Dependencies:
    - python>=3.10
    - redis (optional)
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.config_engine import get_config
from core.logger_engine import get_logger


@dataclass
class Event:
    """Event data structure"""
    topic: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (JSON serializable)"""
        return {
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "event_id": self.event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create Event from dict"""
        return cls(
            topic=data["topic"],
            data=data["data"],
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source"),
            event_id=data.get("event_id")
        )


class EventBus:
    """
    Event Bus - Pub/Sub messaging
    
    Provides inter-module event-driven communication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        """
        Initialize EventBus
        
        Args:
            config: EventBus configuration
        """
        config_engine = get_config()
        default_cfg = config_engine.get("infrastructure.eventbus", {})
        self.config = config or default_cfg or {}
        self.logger = logger or get_logger("core.event_bus")
        self.backend = self.config.get("backend", "memory")
        
        # Subscribers: topic -> [callbacks]
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.lock = Lock()
        
        # Event history (last N events)
        self.history_size = self.config.get("history_size", 1000)
        self.event_history: deque = deque(maxlen=self.history_size)
        
        # Dead letter queue
        self.dead_letter_enabled = self.config.get("dead_letter_enabled", True)
        self.dead_letter_queue: deque = deque(maxlen=1000)
        
        # Stats
        self.stats = {
            "total_published": 0,
            "total_delivered": 0,
            "total_failed": 0
        }
        
        # Redis backend (optional)
        self.redis_client = None
        if self.backend == "redis":
            try:
                import redis
                redis_config = self.config.get("redis", {})
                self.redis_client = redis.Redis(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    db=redis_config.get("db", 0),
                    decode_responses=True
                )
                self.logger.info("✅ EventBus initialized with Redis backend")
            except Exception as e:
                self.logger.warning(f"❌ Redis connection error, using Memory backend: {e}")
                self.backend = "memory"
        
        if self.backend == "memory":
            self.logger.info("✅ EventBus initialized with Memory backend")
    
    def subscribe(self, topic: str, callback: Callable) -> bool:
        """
        Subscribe to topic
        
        Args:
            topic: Topic pattern (e.g., "price.*.update" or "price.BTCUSDT.update")
            callback: Callback function (event: Event)
            
        Returns:
            bool: True if successful
        """
        try:
            with self.lock:
                if callback not in self.subscribers[topic]:
                    self.subscribers[topic].append(callback)
                    self.logger.debug(f"✅ Subscribed to topic: {topic}")
                    return True
                else:
                    self.logger.warning(f"❌ Callback already subscribed: {topic}")
                    return False
        except Exception as e:
            self.logger.error(f"❌ Subscription error {topic}: {e}")
            return False
    
    def unsubscribe(self, topic: str, callback: Callable) -> bool:
        """
        Unsubscribe from topic
        
        Args:
            topic: Topic pattern
            callback: Callback function
            
        Returns:
            bool: True if successful
        """
        try:
            with self.lock:
                if callback in self.subscribers[topic]:
                    self.subscribers[topic].remove(callback)
                    self.logger.debug(f"✅ Unsubscribed from topic: {topic}")
                    return True
                else:
                    self.logger.warning(f"❌ Callback not found: {topic}")
                    return False
        except Exception as e:
            self.logger.error(f"❌ Unsubscription error {topic}: {e}")
            return False
    
    def publish(
        self,
        topic: str,
        data: Any,
        source: Optional[str] = None
    ) -> bool:
        """
        Publish event
        
        Args:
            topic: Topic
            data: Event data (JSON serializable)
            source: Event source (e.g., "WebSocketManager")
            
        Returns:
            bool: True if successful
        """
        try:
            # Create event
            event = Event(
                topic=topic,
                data=data,
                source=source,
                event_id=f"{topic}_{int(time.time() * 1000)}"
            )
            
            # Add to history
            self.event_history.append(event)
            self.stats["total_published"] += 1
            
            if self.redis_client:
                try:
                    self.redis_client.publish(topic, json.dumps(event.to_dict()))
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"❌ Redis publish error {topic}: {exc}")
            
            # Send to subscribers
            delivered_count = self._deliver_to_subscribers(event)
            self.stats["total_delivered"] += delivered_count
            
            self.logger.debug(f"✅ Event published: {topic} (delivered to {delivered_count} subscribers)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Event publishing error {topic}: {e}")
            self.stats["total_failed"] += 1
            return False

    async def publish_async(
        self,
        topic: str,
        data: Any,
        source: Optional[str] = None,
    ) -> bool:
        """
        Publish event asynchronously.
        """

        try:
            return await asyncio.to_thread(self.publish, topic, data, source)
        except RuntimeError:  # fallback if no asyncio loop
            return self.publish(topic, data, source)
    
    def _deliver_to_subscribers(self, event: Event) -> int:
        """
        Deliver event to subscribers
        
        Args:
            event: Event
            
        Returns:
            int: Number of delivered subscribers
        """
        delivered_count = 0
        
        # Exact match
        if event.topic in self.subscribers:
            for callback in self.subscribers[event.topic]:
                if self._invoke_callback(callback, event):
                    delivered_count += 1
        
        # Wildcard match (simple implementation)
        # Support patterns like "price.*.update"
        for pattern, callbacks in self.subscribers.items():
            if "*" in pattern and self._match_pattern(event.topic, pattern):
                for callback in callbacks:
                    if self._invoke_callback(callback, event, pattern=pattern):
                        delivered_count += 1
        
        return delivered_count
    
    def _invoke_callback(self, callback: Callable, event: Event, pattern: Optional[str] = None) -> bool:
        """
        Invoke callback safely.
        """

        try:
            if asyncio.iscoroutinefunction(callback):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(callback(event))
                except RuntimeError:
                    asyncio.run(callback(event))
            else:
                callback(event)
            return True
        except Exception as exc:  # noqa: BLE001
            topic = pattern or event.topic
            self.logger.error(f"Callback error ({topic}): {exc}")
            if self.dead_letter_enabled:
                self.dead_letter_queue.append(
                    {
                        "event": event.to_dict(),
                        "error": str(exc),
                        "timestamp": time.time(),
                    }
                )
            return False
    
    def _match_pattern(self, topic: str, pattern: str) -> bool:
        """
        Topic pattern matching (simple)
        
        Args:
            topic: Actual topic
            pattern: Pattern (* wildcard)
            
        Returns:
            bool: True if matches
        """
        return fnmatch(topic, pattern)
    
    def get_history(self, topic: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history
        
        Args:
            topic: Specific topic (None for all)
            limit: Max number of events
            
        Returns:
            List[Event]: List of events
        """
        if topic:
            return [e for e in list(self.event_history)[-limit:] if e.topic == topic]
        else:
            return list(self.event_history)[-limit:]
    
    def get_dead_letter_queue(self, limit: int = 100) -> List[Dict]:
        """
        Get dead letter queue
        
        Args:
            limit: Max number of events
            
        Returns:
            List[Dict]: List of failed events
        """
        return list(self.dead_letter_queue)[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get EventBus statistics
        
        Returns:
            Dict: Stats
        """
        return {
            **self.stats,
            "total_subscribers": sum(len(callbacks) for callbacks in self.subscribers.values()),
            "total_topics": len(self.subscribers),
            "history_size": len(self.event_history),
            "dead_letter_size": len(self.dead_letter_queue),
            "backend": self.backend
        }
    
    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        self.logger.info("✅ Event history cleared")
    
    def clear_dead_letter_queue(self):
        """Clear dead letter queue"""
        self.dead_letter_queue.clear()
        self.logger.info("✅ Dead letter queue cleared")

_event_bus_instance: Optional[EventBus] = None
_event_bus_lock = Lock()


def get_event_bus() -> EventBus:
    """
    Return EventBus singleton instance.
    """

    global _event_bus_instance
    if _event_bus_instance is None:
        with _event_bus_lock:
            if _event_bus_instance is None:
                _event_bus_instance = EventBus()
    return _event_bus_instance


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 EventBus Test")
    print("=" * 60)
    
    # Create EventBus
    bus = EventBus()
    
    # Test 1: Subscribe
    print("\n1️⃣ Subscribe test:")
    
    def on_price_update(event: Event):
        print(f"   📈 Price Update: {event.data}")
    
    def on_trade_executed(event: Event):
        print(f"   💰 Trade: {event.data}")
    
    bus.subscribe("price.BTCUSDT.update", on_price_update)
    bus.subscribe("trade.executed", on_trade_executed)
    print("   ✅ Subscribed to 2 topics")
    
    # Test 2: Publish
    print("\n2️⃣ Publish test:")
    bus.publish("price.BTCUSDT.update", {"symbol": "BTCUSDT", "price": 50000})
    bus.publish("trade.executed", {"symbol": "ETHUSDT", "side": "BUY", "amount": 1.5})
    
    # Test 3: Wildcard subscription
    print("\n3️⃣ Wildcard test:")
    
    def on_any_price(event: Event):
        print(f"   🌐 Any Price: {event.topic} = {event.data}")
    
    bus.subscribe("price.*.update", on_any_price)
    bus.publish("price.ETHUSDT.update", {"symbol": "ETHUSDT", "price": 3000})
    
    # Test 4: Stats
    print("\n4️⃣ Stats:")
    stats = bus.get_stats()
    print(f"   Total Published: {stats['total_published']}")
    print(f"   Total Delivered: {stats['total_delivered']}")
    print(f"   Total Subscribers: {stats['total_subscribers']}")
    print(f"   Backend: {stats['backend']}")
    
    # Test 5: History
    print("\n5️⃣ History:")
    history = bus.get_history(limit=3)
    for event in history:
        print(f"   - {event.topic}: {event.data}")
    
    # Test 6: Unsubscribe
    print("\n6️⃣ Unsubscribe test:")
    bus.unsubscribe("price.BTCUSDT.update", on_price_update)
    bus.publish("price.BTCUSDT.update", {"symbol": "BTCUSDT", "price": 51000})
    print("   ✅ Unsubscribed (no output above)")
    
    print("\n✅ EventBus test completed!")
    print("=" * 60)