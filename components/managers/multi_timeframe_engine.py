#!/usr/bin/env python3
"""
engines/multi_timeframe_engine.py
SuperBot - Multi-Timeframe Manager
Author: SuperBot Team
Date: 2025-10-16
Versiyon: 1.0.0

Multi-Timeframe Manager - Synchronously manages multiple timeframes.

Features:
- Multi-timeframe candle tracking
- Lower → Higher timeframe aggregation
- Timestamp alignment
- HTF (Higher Timeframe) trend confirmation
- Signal validation across timeframes
- Memory efficient storage

Desteklenen Timeframes:
- Intraday: 1m, 3m, 5m, 15m, 30m
- Hours: 1h, 2h, 4h, 6h, 12h
- Days+: 1d, 3d, 1w, 1M

Usage:
    from engines.multi_timeframe_manager import MultiTimeframeEngine
    
    mtf = MultiTimeframeEngine(
        symbol="BTCUSDT",
        timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
        primary_timeframe="1m",
        websocket_manager=ws,
        cache_manager=cache
    )
    
    await mtf.start()
    
    # Get current candle
    candle_1m = mtf.get_candle("1m")
    
    # HTF confirmation
    if mtf.is_trend_aligned(["1m", "15m", "1h"]):
        print("✅ HTF confirmed!")

Dependencies:
    - pandas
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import time
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


@dataclass
class Candle:
    """Candlestick data structure"""
    timestamp: int  # Unix timestamp (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    is_closed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe,
            "is_closed": self.is_closed
        }


@dataclass
class TimeframeInfo:
    """Timeframe information structure"""
    timeframe: str
    interval_ms: int  # Milliseconds
    candles: deque = field(default_factory=lambda: deque(maxlen=500))
    current_candle: Optional[Candle] = None
    last_update: Optional[datetime] = None


class MultiTimeframeEngine:
    """
    Multi-Timeframe Manager
    
    Manages multiple timeframes synchronously.
    """
    
    # Supported timeframes and their corresponding milliseconds
    TIMEFRAME_MAP = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000  # Approximately
    }
    
    MINIMUM_TIMEFRAMES = 3  # Minimum number of timeframes (was 6, reduced to support SMC_Volume strategy)
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        primary_timeframe: str = "1m",
        websocket_manager: Optional[Any] = None,
        connector_engine: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        indicator_manager: Optional[Any] = None,
        on_candle_closed: Optional[Any] = None,
        warmup_period: int = 500,
        verbose: bool = False
    ):
        """
        Initialize MultiTimeframeEngine

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframes: List of timeframes (minimum 6)
            primary_timeframe: Main timeframe (for trading)
            websocket_manager: WebSocketManager instance
            connector_engine: ConnectorEngine instance
            cache_manager: CacheManager instance
            event_bus: EventBus instance
            indicator_manager: IndicatorManager instance (for warmup after data load)
            on_candle_closed: Callback function when candle closes (for strategy trigger)
            warmup_period: Number of historical candles to load (from strategy.warmup_period)
            verbose: Enable verbose logging (default: False)
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.websocket_manager = websocket_manager
        self.connector_engine = connector_engine
        self.cache_manager = cache_manager
        self.event_bus = event_bus
        self.indicator_manager = indicator_manager  # NEW: For warmup
        self.on_candle_closed = on_candle_closed  # NEW: Callback for strategy
        self.verbose = verbose  # Verbose logging flag

        # Ensure minimum warmup for complex indicators (ADX needs 28+)
        self.warmup_period = max(warmup_period, 50)  # Minimum 50 candles for safety

        # Timeframe validation - minimum requirement removed (user request)
        # The user can provide any number of timeframes (1+).
        if len(timeframes) < 1:
            raise ValueError("At least 1 timeframe is required!")
        
        for tf in timeframes:
            if tf not in self.TIMEFRAME_MAP:
                raise ValueError(f"Desteklenmeyen timeframe: {tf}")
        
        if primary_timeframe not in timeframes:
            raise ValueError(
                f"Primary timeframe ({primary_timeframe}) must be in the timeframes list!"
            )
        
        # Sort the timeframes (from smallest to largest)
        self.timeframes = sorted(
            timeframes,
            key=lambda x: self.TIMEFRAME_MAP[x]
        )
        
        # Timeframe info dict
        self.tf_info: Dict[str, TimeframeInfo] = {}
        for tf in self.timeframes:
            self.tf_info[tf] = TimeframeInfo(
                timeframe=tf,
                interval_ms=self.TIMEFRAME_MAP[tf]
            )
        
        # WebSocket stream ID
        self.stream_id = None
        
        # Aggregation task
        self.aggregation_task = None
        self.running = False
        
        # Stats
        self.stats = {
            "total_candles": 0,
            "total_updates": 0,
            "total_aggregations": 0,
            "missed_candles": 0
        }

        # Active tasks tracking (for callback exception handling)
        self._active_tasks = set()

        if self.verbose:
            logger.info(
                f"MultiTimeframeEngine started: {symbol} - "
                f"Timeframes: {self.timeframes} (Primary: {primary_timeframe})"
            )
    
    async def start(self):
        """Start multi-timeframe tracking"""
        try:
            self.running = True

            # Verbose: Log connector status
            if self.verbose:
                logger.info(f"🔧 MTF Engine starting for {self.symbol}")
                logger.info(f"   Connector: {'✅ Available' if self.connector_engine else '❌ None'}")
                logger.info(f"   WebSocket: {'✅ Available' if self.websocket_manager else '❌ None'}")

            # Load historical data (if a connector exists)
            if self.connector_engine:
                if self.verbose:
                    logger.info(f"📊 Loading historical data for {self.symbol}...")
                await self._load_historical_data()
            else:
                if self.verbose:
                    logger.warning(f"⚠️  No connector, skipping historical data for {self.symbol}")

            # WebSocket subscribe - DISABLED
            # TradingEngine now subscribes all symbols together in ONE WebSocket connection
            # This is more efficient than creating separate connections per symbol
            # MTF engines still subscribe to EventBus to receive messages
            # if self.websocket_manager:
            #     await self._subscribe_websocket()

            # Subscribe to EventBus (to receive WebSocket messages)
            # V2: Subscribe to all timeframes (for MTF support)
            if self.event_bus:
                for tf in self.timeframes:
                    topic = f"candle.{self.symbol}.{tf}"
                    self.event_bus.subscribe(topic, self._on_kline_update)
                    if self.verbose:
                        logger.info(f"📡 EventBus subscribed: {topic}")

            # Start aggregation task
            self.aggregation_task = asyncio.create_task(self._aggregation_loop())
            if self.verbose:
                logger.info(f"✅ Multi-timeframe tracking started: {self.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe start error: {e}")
            raise
    
    async def stop(self):
        """Stops multi-timeframe tracking"""
        try:
            self.running = False
            
            # Stop the aggregation task
            if self.aggregation_task:
                self.aggregation_task.cancel()
                try:
                    await self.aggregation_task
                except asyncio.CancelledError:
                    pass
            
            # WebSocket unsubscribe
            if self.websocket_manager and self.stream_id:
                await self.websocket_manager.unsubscribe(self.stream_id)
            
            logger.info(f"🛑 Multi-timeframe tracking durduruldu: {self.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe stop error: {e}")
    
    async def _load_historical_data(self):
        """Load historical candle data - Fetch separately for each timeframe"""
        try:
            if self.verbose:
                logger.info(f"📊 Loading historical data: {self.symbol}")
                logger.info(f"📊 Using warmup_period: {self.warmup_period} candles")

            # Fetch historical data separately for each timeframe
            # (Instead of aggregation - more accurate and reliable)
            for tf in self.timeframes:
                if self.verbose:
                    logger.info(f"📊 Loading {self.warmup_period} candles for {self.symbol} {tf}...")

                # Binance'den historical candles al
                klines = await self.connector_engine.get_klines(
                    symbol=self.symbol,
                    interval=tf,
                    limit=self.warmup_period
                )

                # Parse the candles
                candle_count = 0
                for kline in klines:
                    candle = Candle(
                        timestamp=int(kline[0]),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5]),
                        timeframe=tf,
                        is_closed=True
                    )

                    self.tf_info[tf].candles.append(candle)
                    self.stats["total_candles"] += 1
                    candle_count += 1

                if self.verbose:
                    logger.info(f"✅ {self.symbol} {tf}: Loaded {candle_count} historical candles")

            if self.verbose:
                logger.info(f"✅ All historical data loaded: {self.stats['total_candles']} total candles")

            # Warmup indicator buffers with primary timeframe data
            if self.indicator_manager:
                await self._warmup_indicator_buffers()

        except Exception as e:
            logger.error(f"❌ Error loading historical data: {e}")

    async def _warmup_indicator_buffers(self):
        """
        Warmup indicator buffers using loaded historical data

        V2: Create separate calculators for each timeframe.
        Calculator key format: "ema_50_5m", "ema_50_15m" (HER ZAMAN suffix)
        """
        try:
            import pandas as pd

            # V2: Separate warmup for each timeframe
            for tf in self.timeframes:
                candles = self.tf_info[tf].candles

                if not candles:
                    if self.verbose:
                        logger.warning(f"⚠️  No {tf} data for {self.symbol}, skipping indicator warmup")
                    continue

                # Convert candles to DataFrame for indicator warmup
                candle_dicts = [
                    {
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    }
                    for c in candles
                ]

                historical_data = pd.DataFrame(candle_dicts)

                # V2: Create separate calculators for each timeframe
                # Calculator key: "ema_50_5m", "ema_50_15m" (HER ZAMAN suffix)
                self.indicator_manager.setup_realtime(
                    symbol=self.symbol,
                    data=historical_data,
                    timeframe=tf  # V2: Timeframe parametresi
                )

                if self.verbose:
                    logger.info(f"   🔥 Indicator buffers warmed up for {self.symbol} {tf} ({len(candle_dicts)} candles)")

        except Exception as e:
            logger.error(f"❌ Indicator warmup error for {self.symbol}: {e}")

    async def _subscribe_websocket(self):
        """WebSocket'e subscribe ol"""
        try:
            # Only subscribe to the primary timeframe
            # The others will be created with aggregation.
            channels = [f"kline_{self.primary_timeframe}"]
            
            self.stream_id = await self.websocket_manager.subscribe(
                symbols=[self.symbol],
                channels=channels
            )
            
            # Listen for WebSocket messages (via EventBus)
            if self.event_bus:
                # FIXED: Topic format must match WebSocketEngine's _get_event_type()
                topic = f"candle.{self.symbol}.{self.primary_timeframe}"
                self.event_bus.subscribe(topic, self._on_kline_update)
            
            if self.verbose:
                logger.info(f"✅ WebSocket subscribed: {self.symbol} {channels}")
            
        except Exception as e:
            logger.error(f"❌ WebSocket subscription error: {e}")
    
    async def _safe_callback(self, symbol: str, timeframe: str):
        """
        Safe wrapper for on_candle_closed callback with exception logging

        Prevents silent failures in asyncio.create_task()
        """
        try:
            await self.on_candle_closed(symbol, timeframe)
        except Exception as e:
            logger.error(f"❌ Candle closed callback error: {symbol} {timeframe}")
            logger.error(f"   Error: {e}", exc_info=True)

    def _on_kline_update(self, event):
        """WebSocket kline update callback (V2: timeframe-aware)"""
        try:
            kline_data = event.data

            # V2: Extract the timeframe from the event topic: "candle.BTCUSDT.15m" -> "15m"
            timeframe = self.primary_timeframe  # Default
            if hasattr(event, 'topic') and event.topic:
                parts = event.topic.split('.')
                if len(parts) >= 3:
                    timeframe = parts[-1]

            # Alternative: If there is a timeframe in kline_data, use it.
            timeframe = kline_data.get('timeframe', timeframe)

            # Invalid timeframe check
            if timeframe not in self.tf_info:
                logger.debug(f"⚠️ Unknown timeframe in event: {timeframe}")
                return

            # DEBUG: Log to verify WebSocket is receiving kline data
            is_closed_flag = kline_data.get("is_closed", False)
            logger.debug(f"🔍 KLINE UPDATE: {self.symbol} {timeframe} is_closed={is_closed_flag}, close={kline_data.get('close', 'N/A')}")

            candle = Candle(
                timestamp=int(kline_data["timestamp"]),
                open=float(kline_data["open"]),
                high=float(kline_data["high"]),
                low=float(kline_data["low"]),
                close=float(kline_data["close"]),
                volume=float(kline_data["volume"]),
                timeframe=timeframe,  # V2: Dynamic timeframe
                is_closed=kline_data["is_closed"]
            )

            # Update current candle (V2: Update the correct timeframe)
            tf_info = self.tf_info[timeframe]

            if candle.is_closed:
                # Candle closed, add to history
                tf_info.candles.append(candle)
                tf_info.current_candle = None
                self.stats["total_candles"] += 1

                # DEBUG: Log when an MTF candle closes (for those not in the primary).
                if timeframe != self.primary_timeframe:
                    logger.info(f"📊 MTF Candle Closed: {self.symbol} {timeframe} close={candle.close:.2f}")

                # V2: Update realtime indicators with closed candle (timeframe-aware)
                if self.indicator_manager:
                    candle_dict = {
                        'timestamp': candle.timestamp,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume
                    }
                    # V2: Add timeframe parameter
                    self.indicator_manager.update_realtime(
                        symbol=self.symbol,
                        new_candle=candle_dict,
                        timeframe=timeframe  # V2: Each timeframe updates its own calculators
                    )

                # Trigger for higher timeframes (only for the primary one)
                if timeframe == self.primary_timeframe:
                    asyncio.create_task(self._trigger_aggregation())

                # TRIGGER STRATEGY: Trigger the strategy when the candle closes.
                if self.on_candle_closed:
                    # Create task with exception handling wrapper
                    task = asyncio.create_task(self._safe_callback(self.symbol, self.primary_timeframe))
                    # Track task to prevent silent failure
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)

            else:
                # Candle is still open, update
                tf_info.current_candle = candle

            tf_info.last_update = datetime.now()
            self.stats["total_updates"] += 1

            # Save to cache
            if self.cache_manager:
                cache_key = f"mtf:{self.symbol}:{self.primary_timeframe}:current"
                self.cache_manager.set(cache_key, candle.to_dict(), ttl=60)

        except Exception as e:
            logger.error(f"❌ Kline update error: {e}", exc_info=True)
            logger.error(f"   Event data: {event.data if hasattr(event, 'data') else 'N/A'}")
    
    async def _aggregation_loop(self):
        """Aggregation loop"""
        if self.verbose:
            logger.info("🔄 Aggregation loop started")
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every 1 minute
                
                # Check aggregation for each timeframe
                await self._trigger_aggregation()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Aggregation loop error: {e}")
    
    async def _trigger_aggregation(self):
        """Aggregation trigger"""
        try:
            # Aggregate from the primary timeframe to higher timeframes
            base_tf = self.primary_timeframe
            
            for tf in self.timeframes:
                if tf == base_tf:
                    continue
                
                await self._aggregate_timeframe(base_tf, tf)
            
        except Exception as e:
            logger.error(f"❌ Aggregation trigger error: {e}")
    
    async def _aggregate_timeframe(self, base_tf: str, target_tf: str):
        """
        Create a higher timeframe from a lower timeframe.
        
        Args:
            base_tf: Base timeframe (e.g., "1m")
            target_tf: Target timeframe (e.g., "5m")
        """
        try:
            base_info = self.tf_info[base_tf]
            target_info = self.tf_info[target_tf]
            
            if not base_info.candles:
                return
            
            # Target timeframe interval
            target_interval = self.TIMEFRAME_MAP[target_tf]
            base_interval = self.TIMEFRAME_MAP[base_tf]
            
            # How many base candles are needed?
            ratio = target_interval // base_interval
            
            # Get the last N candles
            recent_candles = list(base_info.candles)[-int(ratio):]
            
            if len(recent_candles) < ratio:
                return  # Not enough candles
            
            # Is the timestamp of the first candle aligned with the target timeframe?
            first_ts = recent_candles[0].timestamp
            if first_ts % target_interval != 0:
                return  # No alignment
            
            # Create aggregate candle
            agg_candle = Candle(
                timestamp=first_ts,
                open=recent_candles[0].open,
                high=max(c.high for c in recent_candles),
                low=min(c.low for c in recent_candles),
                close=recent_candles[-1].close,
                volume=sum(c.volume for c in recent_candles),
                timeframe=target_tf,
                is_closed=recent_candles[-1].is_closed
            )
            
            # Add to the target timeframe
            # If the last candle has the same timestamp, update it.
            if (target_info.candles and 
                target_info.candles[-1].timestamp == agg_candle.timestamp):
                target_info.candles[-1] = agg_candle
            else:
                target_info.candles.append(agg_candle)
                self.stats["total_aggregations"] += 1
            
            target_info.last_update = datetime.now()
            
            logger.debug(f"🔄 Aggregated: {base_tf} → {target_tf}")
            
        except Exception as e:
            logger.error(f"❌ Aggregation error ({base_tf} -> {target_tf}): {e}")
    
    def get_candle(
        self,
        timeframe: str,
        index: int = -1
    ) -> Optional[Candle]:
        """
        Candle al (real-time updates dahil)

        Args:
            timeframe: Timeframe
            index: Index (-1 = latest, -2 = previous, etc.)

        Returns:
            Candle: Candle object or None

        Note:
            index=-1 returns current_candle if available (WebSocket real-time data),
            otherwise returns last closed candle from history
        """
        if timeframe not in self.tf_info:
            logger.warning(f"⚠️ Invalid timeframe: {timeframe}")
            return None

        tf_info = self.tf_info[timeframe]

        # If requesting latest candle (index=-1), return current_candle if available (WebSocket real-time)
        if index == -1 and tf_info.current_candle is not None:
            return tf_info.current_candle

        # Otherwise return from history
        if not tf_info.candles:
            return None

        try:
            return tf_info.candles[index]
        except IndexError:
            return None
    
    def get_candles(
        self,
        timeframe: str,
        limit: int = 100
    ) -> List[Candle]:
        """
        Get multiple candles (including the real-time current_candle).

        Args:
            timeframe: Timeframe
            limit: Number of candles

        Returns:
            List[Candle]: List of candles (the last candle is the current_candle, which represents real-time data).
        """
        if timeframe not in self.tf_info:
            return []

        tf_info = self.tf_info[timeframe]

        # Get historical candles
        candles = list(tf_info.candles)[-limit:]

        # Add current_candle if exists (WebSocket real-time data)
        if tf_info.current_candle is not None:
            candles.append(tf_info.current_candle)

        return candles
    
    def is_trend_aligned(
        self,
        timeframes: Optional[List[str]] = None,
        lookback: int = 5
    ) -> bool:
        """
        Check if the trend is aligned within the timeframes.
        
        Args:
            timeframes: Timeframes to check (None means all)
            lookback: How many candles to look back
            
        Returns:
            bool: True if all timeframes have the same trend.
        """
        if timeframes is None:
            timeframes = self.timeframes
        
        trends = []
        
        for tf in timeframes:
            candles = self.get_candles(tf, limit=lookback + 1)
            
            if len(candles) < lookback + 1:
                return False  # Not enough data
            
            # Simple trend: last candle vs. first candle
            first = candles[0]
            last = candles[-1]
            
            trend = "UP" if last.close > first.close else "DOWN"
            trends.append(trend)
        
        # Are all trends the same?
        return len(set(trends)) == 1
    
    def get_htf_confirmation(
        self,
        signal_timeframe: str,
        htf_timeframes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Higher Timeframe confirmation
        
        Args:
            signal_timeframe: Signal timeframe
            htf_timeframes: HTF timeframes (if None, uses larger ones)
            
        Returns:
            Dict: Confirmation bilgisi
        """
        if htf_timeframes is None:
            # Get the ones that are greater than the signal timeframe
            signal_idx = self.timeframes.index(signal_timeframe)
            htf_timeframes = self.timeframes[signal_idx + 1:]
        
        if not htf_timeframes:
            return {"confirmed": False, "reason": "No HTF timeframes"}
        
        # Check trend for each HTF
        htf_trends = {}
        
        for tf in htf_timeframes:
            candles = self.get_candles(tf, limit=10)
            
            if len(candles) < 2:
                htf_trends[tf] = "UNKNOWN"
                continue
            
            # Basit trend
            if candles[-1].close > candles[0].close:
                htf_trends[tf] = "UP"
            else:
                htf_trends[tf] = "DOWN"
        
        # Are all HTFs in the same trend?
        unique_trends = set(
            t for t in htf_trends.values() if t != "UNKNOWN"
        )
        
        confirmed = len(unique_trends) == 1
        
        return {
            "confirmed": confirmed,
            "htf_trends": htf_trends,
            "signal_timeframe": signal_timeframe
        }
    
    # ========================================================================
    # DATA TRANSFORMATION (moved from TradingEngine)
    # ========================================================================

    def build_dataframe(
        self,
        indicator_manager: Optional[Any] = None,
        use_previous_candle: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Convert MTF candle data to a DataFrame dictionary for SignalValidator.

        V8: Each timeframe has its own DataFrame (OHLCV + indicators)
        Calculator key format: "ema_50_5m", "ema_50_15m" (HER ZAMAN suffix)

        Bu method TradingEngine._build_dataframe_from_mtf()'nin yerine gecer.

        Args:
            indicator_manager: IndicatorManager instance (for calculators)
            use_previous_candle: True = use the values of the closed candle (for entry)
                                False = use the current values (for monitoring)

        Returns:
            Dict[str, pd.DataFrame]: {timeframe: DataFrame} or None
            Her DataFrame: OHLCV + tum indicator degerleri (suffix'siz)

        Example:
            >>> data = mtf_engine.build_dataframe(indicator_manager)
            >>> data['5m']   # 5m OHLCV + 5m indicators
            >>> data['15m']  # 15m OHLCV + 15m indicators
        """
        import pandas as pd

        try:
            result = {}

            # Known timeframe suffixes for stripping
            known_tf_suffixes = ["_1m", "_3m", "_5m", "_15m", "_30m", "_1h", "_2h", "_4h", "_6h", "_12h", "_1d"]

            # 1. Create a base DataFrame for all timeframes
            for tf, tf_info in self.tf_info.items():
                candles = list(tf_info.candles)
                if not candles:
                    continue

                # Candle'lari dict listesine cevir
                candle_dicts = []
                for candle in candles:
                    candle_dicts.append({
                        'timestamp': candle.timestamp,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume
                    })

                df = pd.DataFrame(candle_dicts)
                result[tf] = df

            # Primary DataFrame yoksa don
            if self.primary_timeframe not in result:
                return None

            # 2. Add indicator values to the relevant DataFrames.
            if indicator_manager:
                calculators = indicator_manager.calculators.get(self.symbol, {})

                if not calculators:
                    logger.warning(f"⚠️ {self.symbol}: calculators bos! Warmup yapilmamis olabilir.")

                # Add each calculator to the DataFrame of its corresponding timeframe.
                for calc_key, calculator in calculators.items():
                    # calc_key: "ema_50_5m", "rsi_14_15m", etc.
                    if use_previous_candle and calculator.previous_result:
                        value = calculator.previous_result.value
                    elif calculator.last_result:
                        value = calculator.last_result.value
                    else:
                        continue

                    # Separate the timeframe and indicator name from calc_key
                    # "ema_50_5m" -> indicator="ema_50", tf="5m"
                    target_tf = None
                    indicator_name = calc_key

                    for tf_suffix in known_tf_suffixes:
                        if calc_key.endswith(tf_suffix):
                            target_tf = tf_suffix[1:]  # "_5m" -> "5m"
                            indicator_name = calc_key[:-len(tf_suffix)]  # "ema_50_5m" -> "ema_50"
                            break

                    # Find the target DataFrame
                    if target_tf and target_tf in result:
                        target_df = result[target_tf]
                    else:
                        # If there is no suffix or the suffix is unknown, add it to the primary.
                        target_df = result.get(self.primary_timeframe)
                        if target_df is None:
                            continue

                    # Add the value to the DataFrame (without suffix!)
                    self._add_indicator_value_to_df(target_df, indicator_name, value)

            # Also add the primary timeframe by default (backward compat)
            result['default'] = result.get(self.primary_timeframe)

            # Log indicator keys (only the first time)
            self._log_indicator_keys(result)

            return result if result else None

        except Exception as e:
            logger.error(f"❌ {self.symbol}: DataFrame build error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    # Class-level flag - single log for all instances
    _global_indicators_logged = False

    def _log_indicator_keys(self, result: Dict[str, Any]) -> None:
        """
        Log indicator keys (ONLY ONCE FOR ALL SYMBOLS)

        Useful for debugging - to disable:
        return  # Add this line
        """
        # return  # <- Uncomment this line to disable it

        if not MultiTimeframeEngine._global_indicators_logged:
            MultiTimeframeEngine._global_indicators_logged = True
            for tf, df in result.items():
                if tf != 'default' and df is not None:
                    logger.info(f"   ✅ {tf}: {len(df.columns)} indicators calculated")
                    logger.info(f"      📋 Keys: {list(df.columns)}")

    def _add_indicator_value_to_df(
        self,
        df: Any,
        indicator_name: str,
        value: Any
    ) -> None:
        """
        Add the indicator value to the DataFrame (helper method).

        Args:
            df: Hedef DataFrame
            indicator_name: Indicator adi (suffix'siz)
            value: Indicator degeri (scalar, dict, Series, etc.)

        Note:
            Empty lists/tuples are converted to None to avoid DataFrame length mismatch errors.
            This can happen during warmup period when indicators don't have enough data.
        """
        if isinstance(value, dict):
            # Multi-output indicators: {'lower': 100, 'middle': 105, 'upper': 110}
            for key, val in value.items():
                col_name = f"{indicator_name}_{key}"
                if val is None:
                    df[col_name] = None
                elif isinstance(val, (int, float)):
                    df[col_name] = val
                elif hasattr(val, 'iloc') and len(val) > 0:
                    df[col_name] = val.iloc[-1]
                elif isinstance(val, (list, tuple)):
                    # Handle empty list/tuple - convert to None
                    df[col_name] = val[-1] if len(val) > 0 else None
                else:
                    df[col_name] = val
            # Add the main indicator name as well (initial value)
            if value:
                first_val = list(value.values())[0]
                if first_val is None:
                    df[indicator_name] = None
                elif isinstance(first_val, (int, float)):
                    df[indicator_name] = first_val
                elif hasattr(first_val, 'iloc') and len(first_val) > 0:
                    df[indicator_name] = first_val.iloc[-1]
        elif value is None:
            df[indicator_name] = None
        elif isinstance(value, (int, float)):
            df[indicator_name] = value
        elif hasattr(value, 'iloc'):
            df[indicator_name] = value.iloc[-1] if len(value) > 0 else None
        elif isinstance(value, (list, tuple)):
            # Handle empty list/tuple - convert to None
            df[indicator_name] = value[-1] if len(value) > 0 else None
        else:
            df[indicator_name] = value

    def is_new_candle_closed(self, last_processed_timestamp: int) -> Tuple[bool, int]:
        """
        Check if a new candle has closed in the primary timeframe.

        Bu method TradingEngine._is_new_candle_closed()'in yerine gecer.

        Args:
            last_processed_timestamp: The timestamp (in milliseconds) of the last processed candle.

        Returns:
            Tuple[bool, int]: (is_new_candle, current_timestamp)
            - is_new_candle: Did a new candle close?
            - current_timestamp: Su anki candle timestamp'i

        Example:
            >>> is_new, current_ts = mtf_engine.is_new_candle_closed(last_ts)
            >>> if is_new:
            ...     # Process new candle
            ...     last_ts = current_ts
        """
        tf_info = self.tf_info.get(self.primary_timeframe)
        if not tf_info or not tf_info.candles:
            return False, 0

        # The timestamp of the last closed candle.
        current_ts = tf_info.candles[-1].timestamp

        # Is this a new candle?
        is_new = current_ts > last_processed_timestamp

        return is_new, current_ts

    def get_stats(self) -> Dict[str, Any]:
        """Return multi-timeframe statistics"""
        return {
            **self.stats,
            "symbol": self.symbol,
            "timeframes": self.timeframes,
            "primary_timeframe": self.primary_timeframe,
            "timeframe_candle_counts": {
                tf: len(info.candles)
                for tf, info in self.tf_info.items()
            }
        }


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("🧪 MultiTimeframeEngine Test")
        print("=" * 60)
        
        # Mock connector
        class MockConnector:
            async def get_klines(self, symbol, interval, limit):
                # Dummy kline data
                base_ts = int(time.time() * 1000) - (limit * 60 * 1000)
                return [
                    [
                        base_ts + (i * 60 * 1000),  # timestamp
                        50000 + i * 10,  # open
                        50000 + i * 10 + 50,  # high
                        50000 + i * 10 - 30,  # low
                        50000 + i * 10 + 20,  # close
                        100.0  # volume
                    ]
                    for i in range(limit)
                ]
        
        # Create MTF
        mtf = MultiTimeframeEngine(
            symbol="BTCUSDT",
            timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
            primary_timeframe="1m",
            connector_engine=MockConnector()
        )
        
        # Initialize
        await mtf.start()
        
        print("\n📊 Candle Counts:")
        for tf in mtf.timeframes:
            count = len(mtf.get_candles(tf))
            print(f"   {tf}: {count} candles")
        
        print("\n📈 Latest Candles:")
        for tf in ["1m", "5m", "1h"]:
            candle = mtf.get_candle(tf)
            if candle:
                print(f"   {tf}: Close=${candle.close:.2f}")
        
        print("\n🔍 Trend Alignment:")
        aligned = mtf.is_trend_aligned(["1m", "5m", "15m"])
        print(f"   Aligned: {'✅ Yes' if aligned else '❌ No'}")
        
        print("\n📊 Stats:")
        stats = mtf.get_stats()
        print(f"   Total Candles: {stats['total_candles']}")
        print(f"   Total Aggregations: {stats['total_aggregations']}")
        
        # Stop
        await mtf.stop()
        
        print("\n✅ Test completed!")
        print("=" * 60)
    
    asyncio.run(test())